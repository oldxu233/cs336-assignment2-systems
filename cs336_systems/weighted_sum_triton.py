import triton
import triton.language as tl
import torch
from einops import rearrange

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,  # 输入指针
    output_ptr,         # 输出指针
    x_stride_row, x_stride_dim,  # 步长定义了在张量各维度上移动一个元素的偏移量
    weight_stride_dim,  # 通常为1
    output_stride_row,  # 通常为1
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,  # 分块大小必须在编译时确定
):
    # 每个实例将计算x的一个行分块的加权和
    # `tl.program_id` 用于获取当前运行的线程块索引
    row_tile_idx = tl.program_id(0)

    # 块指针用于从内存的ND区域中选择数据, 并可灵活移动选择范围
    # 块指针必须包含以下信息：
    # - 张量第一个元素的指针
    # - 张量的整体形状（用于处理越界访问）
    # - 各维度的步长（用于正确适配内存布局）
    # - 起始块的ND坐标（即“偏移量”）
    # - 每次加载/存储的块形状
    # - 内存中维度的主次顺序（通过对步长排序得到，axes(=np.argsort(strides))
    #   有助于H100等硬件的优化）

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 初始化输出缓冲区
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载当前块数据
        # 由于ROWS_TILE_SIZE可能无法整除ROWS，D_TILE_SIZE可能无法整除D，
        # 因此需要对两个维度都进行边界检查
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)

        # 计算行的加权和
        output += tl.sum(row * weight[None, :], axis=1)

        # 将指针推进到下一个分块
        # 以下为（行、列）坐标增量
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))  # 沿最后一个维度移动D_TILE_SIZE
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))  # 移动D_TILE_SIZE

    # 将输出写入输出块指针（每行对应一个标量）
    # 由于ROWS_TILE_SIZE可能无法整除ROWS，需要进行边界检查
    tl.store(output_block_ptr, output, boundary_check=(0,))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # 缓存 x 和 weight，用于反向传播（反向传播时仅接收输出张量的梯度，
        # 需计算 x 和 weight 对应的梯度）
        D, output_dims = x.shape[-1], x.shape[:-1]

        # 将输入张量重塑为 2D 形状
        input_shape = x.shape
        x = rearrange(x, "...d -> (...)d")
        # x = x.unsqueeze(-1)

        # 保存反向传播需要的张量
        ctx.save_for_backward(x, weight)

        # 维度校验
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        # 设置 Triton 核函数的 tile 尺寸
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # 对嵌入维度约循环16次
        ctx.ROWS_TILE_SIZE = 16  # 每个线程一次处理16个批次元素
        ctx.input_shape = input_shape

        # 初始化空的结果张量（注意：这些元素不一定初始化为0！）
        y = torch.empty(output_dims, device=x.device)

        # 启动 Triton 核函数：1D grid，每个实例处理 ROWS_TILE_SIZE 行
        n_rows = y.numel()
        weighted_sum_fwd[(triton.div(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        # 将输出重塑为原始输入的除最后一维外的形状
        return y.view(input_shape[:-1])
    
    @staticmethod
    def backward(ctx, grad_out):
        # 恢复正向传播保存的张量
        x, weight = ctx.saved_tensors
        # 恢复分块大小（行/列分块可不同）
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        # 获取输入张量维度
        n_rows, D = x.shape

        # 策略：先让每个线程块写入分片梯度缓冲区，再归约得到最终梯度
        # 初始化w的分片梯度缓冲区（n_row_tiles × D）
        partial_grad_weight = torch.empty(
            (triton.cdiv(n_rows, ROWS_TILE_SIZE), D),
            device=x.device,
            dtype=x.dtype
        )
        # 初始化x的梯度张量（与x形状、设备、 dtype一致）
        grad_x = torch.empty_like(x)

        # 调用Triton反向传播核函数
        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            # 输入张量
            x, weight,
            grad_out,
            # 输出梯度张量
            grad_x, partial_grad_weight,
            # 步长参数（x的行/列步长）
            x.stride(0), x.stride(1),
            # weight的步长
            weight.stride(0),
            # grad_out的步长
            grad_out.stride(0),
            # grad_x的行/列步长
            grad_x.stride(0), grad_x.stride(1),
            # partial_grad_weight的行/列步长
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            # 全局维度参数
            NUM_ROWS=n_rows, D=D,
            # 编译期分块大小常量
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )

        # 归约分片梯度：对所有行分块的结果求和，得到最终∇w
        grad_weight = partial_grad_weight.sum(axis=0)

        # 返回梯度（与forward输入参数一一对应）
        return grad_x, grad_weight

    

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,        # 输入：x矩阵指针、权重w向量指针
    grad_output_ptr,          # 输入：损失对输出f(x,w)的梯度 ∇_{f(x,w)}ℒ 指针
    grad_x_ptr, partial_grad_weight_ptr,  # 输出：∇xℒ指针、w的分片梯度指针
    stride_xr, stride_xd,     # x的行/列步长 (x: NUM_ROWS×D)
    stride_wd,                # w的列步长 (w: D×1)
    stride_gr,                # grad_output的行步长 (grad_output: NUM_ROWS×1)
    stride_gxr, stride_gxd,   # grad_x的行/列步长 (grad_x: NUM_ROWS×D)
    stride_gwb, stride_gwd,   # partial_grad_weight的块/列步长 (n_row_tiles×D)
    NUM_ROWS, D,              # 全局维度：总行数、总列数
    ROWS_TILE_SIZE: tl.constexpr,  # 编译期常量：行分块大小
    D_TILE_SIZE: tl.constexpr,     # 编译期常量：列分块大小
):
    # 1. 获取当前程序（分块）的行索引
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)  # 总行分块数

    # 2. 定义所有输入/输出的块指针（Block Pointer）
    # 2.1 定义∇_{f(x,w)}ℒ的块指针（维度：ROWS_TILE_SIZE×1）
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # 2.2 定义输入x的块指针（维度：ROWS_TILE_SIZE×D_TILE_SIZE）
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),  # 先列后行，适配内存布局提升访存效率
    )

    # 2.3 定义权重w的块指针（维度：D_TILE_SIZE×1）
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,),
        offsets=(0,), block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    # 2.4 定义∇xℒ的输出块指针（维度：ROWS_TILE_SIZE×D_TILE_SIZE）
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D,), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # 2.5 定义w的分片梯度块指针（维度：1×D_TILE_SIZE）
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D,), strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    # 3. 按列分块遍历，逐块计算梯度
    for i in range(tl.cdiv(D, D_TILE_SIZE)):  # tl.cdiv：向上取整除法，遍历所有列分块
        # 3.1 加载当前分块的∇_{f(x,w)}ℒ（形状：(ROWS_TILE_SIZE,)）
        grad_output = tl.load(
            grad_output_block_ptr,
            boundary_check=(0,),  # 行维度边界检查
            padding_option="zero"  # 越界部分填充0
        )

        # 3.2 计算∇xℒ：w与∇_{f(x,w)}ℒ的外积（公式2）
        # 加载当前分块的权重w（形状：(D_TILE_SIZE,)）
        weight = tl.load(
            weight_block_ptr,
            boundary_check=(0,),  # 列维度边界检查
            padding_option="zero"
        )
        # 外积计算：(ROWS_TILE_SIZE,1) × (1,D_TILE_SIZE) → (ROWS_TILE_SIZE,D_TILE_SIZE)
        grad_x_row = grad_output[:, None] * weight[None, :]
        # 将∇xℒ写入输出指针
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # 3.3 计算w的分片梯度∇wℒ（公式3）
        # 加载当前分块的x（形状：(ROWS_TILE_SIZE, D_TILE_SIZE)）
        row = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),  # 行+列维度边界检查
            padding_option="zero"
        )
        # 按行求和：(ROWS_TILE_SIZE,D_TILE_SIZE) × (ROWS_TILE_SIZE,1) → (1,D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        # 将分片梯度写入输出指针（dim0无越界，仅检查dim1）
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))

        # 3.4 移动块指针到下一个列分块
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


