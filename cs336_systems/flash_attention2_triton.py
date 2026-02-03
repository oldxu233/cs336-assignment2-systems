import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    O_block, l_i, m_i, Q_block, K_block_ptr, V_block_ptr,
    block_index_query, softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr, offs_q: tl.constexpr, offs_kv: tl.constexpr, SEQ_LEN: tl.constexpr,
):
    if STAGE == 1:
        # 从0到对角线左侧
        lo, hi = 0, block_index_query * BLOCK_SIZE_Q
    elif STAGE == 2:
        # 仅用于包含非掩码键到掩码键过渡的块
        lo, hi = block_index_query * BLOCK_SIZE_Q, (block_index_query + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # 仅用于非因果注意力
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # compute QK
        K_block = tl.load(K_block_ptr)
        S = tl.dot(Q_block, K_block) 

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            S = S * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(S, 1))
            S -= m_ij[:, None]
        else:
            # 计算qk的最大值 或 保留旧的最大值
            m_ij = tl.maximum(m_i, tl.max(S, 1) * softmax_scale)
            S = S * softmax_scale - m_ij[:, None]
        
        P_block = tl.math.exp(S)
        
        l_ij = tl.sum(P_block, axis=1)
        
        # 修正因子
        alpha = tl.math.exp(m_i - m_ij)

        l_i = l_i * alpha + l_ij
        
        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)

        O_block =  O_block * alpha[:, None]
        O_block =  tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        # move to the net block of K AND V
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
    
    return O_block, l_i, m_i

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    softmax_scale,
    O_ptr, L_ptr,
    stride_qb, stride_qh, stride_q_seq, stride_qd,
    stride_kb, stride_kh, stride_k_seq, stride_kd,
    stride_vb, stride_vh, stride_v_seq, stride_vd,
    stride_ob, stride_oh, stride_o_seq, stride_od,
    stride_lb, stride_lq,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    # tl.static_assert(N_KEYS <= K_TILE_SIZE)
    
    # 程序索引
    block_index_query = tl.program_id(0)      # this indicate which block in the seq_len to process.
    index_batch_head = tl.program_id(1)       # this indicate which head and batch to process.
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    # get the (seq_len, head_dim) block in the Q, K, V by seleting indexing it by batch and head
    qvk_offset = (index_batch.to(tl.int64) * stride_qb + index_head.to(tl.int64) * stride_qh)

    # 为每个指针添加批次索引对应的偏移量（批次索引 × 每个张量的批次步长）
    Q_block_ptr = tl.make_block_ptr(
        base = Q_ptr + qvk_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_q_seq, stride_qd),
        offsets = (block_index_query * BLOCK_SIZE_Q, 0), # Q[batch_size, num_heads, block_index_query * BLOCK_SIZE_Q, :]
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base = V_ptr + qvk_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_v_seq, stride_vd),
        offsets = (0, 0),                           # V[batch_size, num_heads, :, :]
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order = (1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base = K_ptr + qvk_offset, 
        shape = (HEAD_DIM, SEQ_LEN),
        strides = (stride_kd, stride_k_seq), # we invert the strides, so we transpose the matrix
        offsets = (0, 0),
        block_shape = (HEAD_DIM, BLOCK_SIZE_KV),
        order = (0, 1),
    )

    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + qvk_offset, 
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_o_seq, stride_od),
        offsets = (block_index_query * BLOCK_SIZE_Q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0),
    )

    # the offsets for the tokens in Q to process
    offs_q = block_index_query * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    # the offsets for the tokens in K and V to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i : the running maximum. we have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    
    # l_i: the running sum. we have one for each query
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # acc: the accumulator for output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    
    # load the blocks of Q: 它将全程保留在 SRAM 中
    Q_block = tl.load(Q_block_ptr)

    # stage: 3 if causal, else 1
    if STAGE == 1 or STAGE == 3:
        # this step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block, l_i, m_i, Q_block, K_block_ptr, V_block_ptr, 
            block_index_query, softmax_scale, BLOCK_SIZE_Q, BLOCK_SIZE_KV,
            4 - STAGE, offs_q, offs_kv, SEQ_LEN, 
        )

    if STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block, l_i, m_i, Q_block, K_block_ptr, V_block_ptr, 
            block_index_query, softmax_scale, BLOCK_SIZE_Q, BLOCK_SIZE_KV,
            2, offs_q, offs_kv, SEQ_LEN, 
        )

    m_i += tl.math.log(l_i) # this is needed for backwards pass
    O_block = O_block / l_i[:, None]
    l_ptrs = L_ptr + index_batch_head * SEQ_LEN + offs_q
    tl.store(l_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O_ptr.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
    block_index_q = tl.program_id(0)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM)
    
    # load a single block of BLOCK_SIZE_Q rows of O
    O_block = tl.load(
        O + index_batch_head * HEAD_DIM * SEQ_LEN + offs_q[:, None] * HEAD_DIM + offs_dim[None, :]
    ) #(BLOCK_SIZE_Q, HEAD_DIM)
    
    # load a single block of BLOCK_SIZE_Q rows of dO
    dO_block = tl.load(
        dO + index_batch_head * HEAD_DIM * SEQ_LEN + offs_q[:, None] * HEAD_DIM + offs_dim[None, :]
    ).to(tl.float32)

    D_block = tl.sum(dO_block * O_block, axis=1) # shape: (BLOCK_SIZE_Q, )
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)


@triton.jit
def _attn_bwd_dk_dv(
    Q, K, V, softmax_scale, dO, dQ, dK, dV, 
    M, D, stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # 确保指针相对于 batch 和 head 处于正确的位置。
    # 我们不通过 make_block_ptr 访问块的原因是，我们需要使用偏移量范围来应用掩码。 
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # 确保指针相对于 batch  head 和 sequence 处于正确的位置。
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner Loop
    K_block = tl.load(K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim) # shape: (BLOCK_KV, HEAD_DIM)
    V_block = tl.load(V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim) # shape: (BLOCK_KV, HEAD_DIM)

    offs_q = tl.arange(0, BLOCK_Q)

    # 我们将 Q 视为一个转置后的数组进行访问，因此我们将 offs_q 视为列向量，offs_dim 视为行向量。
    # 这等价于执行以下操作：
    # q_ptrs = Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    # qT_ptrs = tl.trans(q_ptrs)
    # 我们指向 Q 的前 BLOCK_Q 行（针对 qT 和 d0 指针），在 for 循环内部，每次迭代我们将向前移动 BLOCK_Q 行。
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    # 沿着query的 seq 维度迭代
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        qT_block = tl.load(qT_ptrs)
        # load the logsumexp values for queries in the current block
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)
        P_T_block = tl.math.exp(QK_T_block - m[None, :])

        if STAGE == 3:
            # mask is true for all values that do not need to be masked
            mask_block = (offs_q[None, :] >= offs_kv[:, None]) # shape: (BLOCK_KV1, BLOCK_Q1)
            P_T_block = tl.where(mask_block, P_T_block, 0.0)
        
        dO_block = tl.load(dO_ptrs)
        # 根据公式, dV_new = dV_old + P^T x dO , x 是矩阵乘法
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        # Delta = rowsum(O * dO) * 是 element-wise 乘法
        Di = tl.load(D + offs_q)

        # dP = dO x V^T 所以 dP^T = V x dO^T
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
        # dS = P * (dP - Delta) 所以 dS^T = P^T * (dP^T - Delta^T)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)

        # dK_new = dK_old + dS^T x Q
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))

        # move to the next block
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq

    # write back dV
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)

    # write back dK
    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)

@triton.jit
def _attn_bwd_dq(
    Q, K, V, softmax_scale, dO, dQ, dK, dV, 
    M, D, stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    # 确保指针相对于 batch 和 head 处于正确的位置。
    # 我们不通过 make_block_ptr 访问块的原因是，我们需要使用偏移量范围来应用掩码。 
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    # 确保指针相对于 batch  head 和 sequence 处于正确的位置。
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_q = tl.program_id(0)
    start_q = index_block_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)

    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]

    offs_kv = tl.arange(0, BLOCK_KV)

    KT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    VT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    Di = tl.load(D + offs_q)
    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    for blk_idx in range(num_steps):
        K_T_block = tl.load(KT_ptrs)
        V_T_block = tl.load(VT_ptrs)

        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block)

        if STAGE == 3:
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)

        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))

        curr_kv += BLOCK_KV
        KT_ptrs += BLOCK_KV * stride_seq
        VT_ptrs += BLOCK_KV * stride_seq
    
    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)

class FlashAttention2(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal=False):
        stage = 3 if causal else 1

        # d_v = value.shape[-1]
        batch_size, num_heads, seq_len, d_k = query.shape
        softmax_scale = 1 / (d_k**0.5)
        
        O = torch.empty_like(query)

        # 并行程序的数量: batch_size * num_heads * NUM_BLOCKS_Q
        grid = lambda args: (triton.cdiv(seq_len, args["BLOCK_SIZE_Q"]), batch_size * num_heads, 1)
        
        # L is the logsumexp for the backward pass, one for each query
        L = torch.empty((batch_size, num_heads, seq_len), device=query.device, dtype=torch.float32)

        flash_fwd_kernel[grid](
            Q_ptr=query, K_ptr=key, V_ptr=value,
            softmax_scale=softmax_scale,
            O_ptr=O, L_ptr=L,
            stride_qb=query.stride(0), stride_qh=query.stride(1), stride_q_seq=query.stride(2), stride_qd=query.stride(3),
            stride_kb=key.stride(0), stride_kh=key.stride(1), stride_k_seq=key.stride(2), stride_kd=key.stride(3),
            stride_vb=value.stride(0), stride_vh=value.stride(1), stride_v_seq=value.stride(2), stride_vd=value.stride(3),
            stride_ob=O.stride(0), stride_oh=O.stride(1), stride_o_seq=O.stride(2), stride_od=O.stride(3),
            stride_lb=L.stride(0), stride_lq=L.stride(2),
            BATCH_SIZE=query.shape[0],
            NUM_HEADS=query.shape[1],
            SEQ_LEN=query.shape[2],
            HEAD_DIM=d_k,
            STAGE=stage,
        )

        ctx.save_for_backward(query, key, value, O, L)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.head_dim = d_k
        ctx.causal = causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors

        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, _ = Q.shape
        NUM_WARPS, NUM_STAGES = 4, 3 
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128

        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M) # shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.head_dim,
        )

        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1,  BATCH_SIZE * NUM_HEADS)
        stage = 3 if ctx.causal else 1

        # fix KV and iterate through all the Q blocks 
        _attn_bwd_dk_dv[grid](
            Q=Q, K=K, V=V, softmax_scale=ctx.softmax_scale, dO=dO, dQ=dQ, dK=dK, dV=dV, 
            M=M, D=D, stride_batch=Q.stride(0), stride_head=Q.stride(1), stride_seq=Q.stride(2), stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.head_dim,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        # fix Q and iterate through all the KV blocks
        _attn_bwd_dq[grid](
            Q=Q, K=K, V=V, softmax_scale=ctx.softmax_scale, dO=dO, dQ=dQ, dK=dK, dV=dV, 
            M=M, D=D, stride_batch=Q.stride(0), stride_head=Q.stride(1), stride_seq=Q.stride(2), stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.head_dim,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        return dQ, dK, dV, None, None

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    softmax_scale = 1 / (HEAD_DIM**0.5)
    dO = torch.rand_like(Q) # 用于反向传播
    
    # 参考实现
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # triton实现
    tri_out = FlashAttention2.apply(Q, K, V, causal).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    #对比
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dQ, tri_dQ, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dK, tri_dK, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dV, tri_dV, rtol=rtol, atol=atol)

if __name__ == "__main__":
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=1024, HEAD_DIM=64, causal=True)
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=1024, HEAD_DIM=64, causal=False)
    print("All tests passed!")