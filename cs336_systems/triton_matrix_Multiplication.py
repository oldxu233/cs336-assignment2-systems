# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

'''## 矩阵乘法（Matrix Multiplication）

在本教程中，你将编写一个非常简短但高性能的 FP16（半精度浮点）矩阵乘法 kernel，其性能可与 cuBLAS（NVIDIA）或 rocBLAS（AMD）相媲美。

你将具体学习以下内容：

- **块级（Block-level）矩阵乘法**  
- **多维指针算术（Multi-dimensional pointer arithmetic）**  
- **程序重排序以提高 L2 缓存命中率（Program re-ordering for improved L2 cache hit rate）**  
- **自动性能调优（Automatic performance tuning）**

---

### 动机（Motivations）

矩阵乘法是现代高性能计算系统中最核心的构建模块之一。然而，它极难优化，因此通常由硬件厂商在所谓的“内核库”（kernel libraries，例如 cuBLAS）中提供高度优化的实现。遗憾的是，这些库往往是闭源的，难以根据现代深度学习工作负载的需求进行定制（例如融合激活函数等操作）。

在本教程中，你将学习如何使用 Triton 自行实现高效的矩阵乘法——这种方法不仅易于定制，也便于扩展。

---

粗略来说，我们将编写的 kernel 会实现以下**分块算法（blocked algorithm）**，用于计算一个形状为 (M, K) 的矩阵与一个形状为 (K, N) 的矩阵的乘积：


# 并行执行（由不同 Triton 程序实例处理）
for m in range(0, M, BLOCK_SIZE_M):
    # 并行执行
    for n in range(0, N, BLOCK_SIZE_N):
        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
        for k in range(0, K, BLOCK_SIZE_K):
            a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
            b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
            acc += dot(a, b)
        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
其中，上述双重嵌套循环中的每一次迭代都由一个独立的 Triton 程序实例（program instance）并行执行。

---

## 计算内核（Compute Kernel）

上述算法在 Triton 中实际上相当直观。主要难点在于**内层循环中计算矩阵 A 和 B 的数据块所需读取的内存地址**。为此，我们需要使用**多维指针算术（multi-dimensional pointer arithmetic）**。

---

### 指针算术（Pointer Arithmetic）

对于一个以行优先（row-major）方式存储的二维张量 X ，其元素 X[i, j] 的内存地址可表示为：
&X[i, j] = X + i * stride_xi + j * stride_xj


因此，矩阵块 A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K] 和 B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N] 所对应的指针块，在伪代码中可定义为：

```python
&A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K] =
    a_ptr + (m : m+BLOCK_SIZE_M)[:, None] * A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :] * A.stride(1)

&B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N] =
    b_ptr + (k : k+BLOCK_SIZE_K)[:, None] * B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :] * B.stride(1)
```

这意味着，在 Triton 中，我们可以在**初始时（即 k = 0）** 将 A 和 B 的块指针初始化如下。  
**注意**：我们还需要额外使用模运算（`%`）来处理  M  不是 BLOCK_SIZE_M 的整数倍，或 N 不是 BLOCK_SIZE_N 的整数倍的情况。
此时，我们可以用任意“无用”值填充越界部分——这些值不会影响最终计算结果（因为后续会通过掩码忽略）。
至于 \( K \) 维度的边界处理，我们稍后将通过 Triton 的**掩码加载（masking load）语义**来实现。

```python
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offs_k = tl.arange(0, BLOCK_SIZE_K)

a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
```

随后，在内层循环中，这些指针将按如下方式更新，以指向下一个 K 方向的数据块：

```python
a_ptrs += BLOCK_SIZE_K * stride_ak
b_ptrs += BLOCK_SIZE_K * stride_bk
```

---

## L2 缓存优化（L2 Cache Optimizations）

如前所述，每个 Triton 程序实例负责计算矩阵 \( C \) 中一个大小为 [BLOCK_SIZE_M, BLOCK_SIZE_N] 的输出块。
**这些块的计算顺序至关重要**，因为它直接影响程序的 **L2 缓存命中率**。遗憾的是，简单的行优先（row-major）调度顺序：
```python
pid = tl.program_id(axis=0)
grid_n = tl.cdiv(N, BLOCK_SIZE_N)
pid_m = pid // grid_n
pid_n = pid % grid_n
```
**无法满足高性能需求**。

一种有效的解决方案是：**以促进数据重用的方式调度程序块的执行顺序**。
具体做法是：在切换到下一列之前，先将 **`GROUP_SIZE_M` 行** 的程序块“超分组”（super-group）在一起进行调度。

实现代码如下：
```python
# 当前程序的全局 ID
pid = tl.program_id(axis=0)

# M 方向上的程序块总数
num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
# N 方向上的程序块总数
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

# 每个组包含的程序块数量
num_pid_in_group = GROUP_SIZE_M * num_pid_n

# 当前程序所属的组 ID
group_id = pid // num_pid_in_group

# 该组中第一个程序的行 ID
first_pid_m = group_id * GROUP_SIZE_M

# 如果 num_pid_m 不能被 GROUP_SIZE_M 整除，则最后一个组会更小
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

# 在组内，程序按列优先（column-major）顺序排列
# 当前程序在启动网格（launch grid）中的行 ID
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
# 当前程序在启动网格中的列 ID
pid_n = (pid % num_pid_in_group) // group_size_m
```

### 优化效果示例

例如，在一个 \(9 \times 9\) 块的矩阵乘法中（即 A 和 B 均由 9×9 个块组成）：

- 若采用**行优先顺序**计算前 9 个输出块（即第一行的 9 个块），则需要从全局内存加载 **90 个块**（9 个 A 块 × 9 列 + 9 个 B 块 × 9 行，存在大量重复加载）到片上存储（SRAM）。
- 而若采用上述**分组调度顺序**（每组包含多行，组内列优先），则只需加载 **54 个块** 即可完成相同计算。

这种调度策略显著减少了对全局内存的访问次数，从而大幅提升 L2 缓存利用率和整体性能。
在实际应用中，这种调度策略可以在某些硬件架构上将矩阵乘法 kernel 的性能提升 **超过 10%**。例如，在 NVIDIA A100 GPU 上，计算吞吐量可从 **220 TFLOPS 提升至 245 TFLOPS**。

'''

import torch
import triton
import triton.language as tl

# 获取当前活跃的 PyTorch 设备（如 'cuda:0'）
assert torch.cuda.is_available(), "This tutorial requires a CUDA GPU"
DEVICE = torch.device("cuda")


def is_cuda():
    """判断当前后端是否为 CUDA（而非 HIP/ROCm）"""
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    """
    为 CUDA 平台（如 NVIDIA A100/H100）定义一组候选配置。
    每个配置包含分块大小（BLOCK_SIZE_*）、分组大小（GROUP_SIZE_M）、
    流水线级数（num_stages）和 warp 数量（num_warps）。
    """
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # ...（其他配置，略）
        # 针对 fp8 输入的优化配置
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6}, num_stages=4, num_warps=4),
    ]


def get_hip_autotune_config():
    """
    为 HIP/ROCm 平台（如 AMD MI200/MI300）定义候选配置。
    注意：AMD CDNA 架构需指定 `matrix_instr_nonkdim=16` 以启用 WMMA 指令。
    """
    sizes = [
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        # ...（其他尺寸）
    ]
    return [
        triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2)
        for s in sizes
    ]


def get_autotune_config():
    """根据当前设备后端返回对应的自动调优配置列表"""
    return get_cuda_autotune_config() if is_cuda() else get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(
    # 矩阵指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 步长（stride）：移动一个元素在各维度上指针的增量
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 编译时常量（由自动调优器填充）
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr  # 支持融合激活函数，如 "leaky_relu"
):
    """
    计算矩阵乘法：C = A @ B
    - A: (M, K)
    - B: (K, N)
    - C: (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# 我们现在可以创建一个便捷的封装函数（wrapper function），该函数仅接收两个输入张量，并完成以下任务：  
# (1) 检查形状约束；  
# (2) 分配输出张量；  
# (3) 启动上述 kernel。
def matmul(a, b, activation=""):
    # 检查约束
    assert a.shape[1] == b.shape[0], "矩阵维度不匹配"
    assert a.is_contiguous(), "矩阵 A 必须是连续的"
    M, K = a.shape
    K, N = b.shape
    # 分配输出
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D 启动 kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), 
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c

# # 我们可以将自定义的矩阵乘法操作与 PyTorch 原生实现（即底层调用 cuBLAS）进行对比测试
# torch.manual_seed(0)
# a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
# b = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
# triton_output = matmul(a, b)
# torch_output = torch.matmul(a, b)
# print(f"triton_output_with_fp16_inputs={triton_output}")
# print(f"torch_output_with_fp16_inputs={torch_output}")

# if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
# if TORCH_HAS_FP8 and is_cuda():
#     torch.manual_seed(0)
#     a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
#     b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
#     a = a.to(torch.float8_e5m2)
#     # pre-transpose b for efficiency.
#     b = b.T
#     b = b.to(torch.float8_e5m2)
#     triton_output = matmul(a, b)
#     torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
#     print(f"triton_output_with_fp8_inputs={triton_output}")
#     print(f"torch_output_with_fp8_inputs={torch_output}")
#     if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
#         print("✅ Triton and Torch match")
#     else:
#         print("❌ Triton and Torch differ")

# 基准测试  
# 方形矩阵性能  
# 我们现在可以将我们编写的内核性能与 cuBLAS 或 rocBLAS 进行比较。此处我们聚焦于方形矩阵，但您可以根据需要自由调整此脚本，以对其他任意矩阵形状进行基准测试。
# 根据当前设备类型选择参考库：CUDA 上用 cuBLAS，ROCm 上用 rocBLAS
ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
# 分别测试普通 float16 输入和 fp8 输入的情况
for fp8_inputs in [False, True]:
    # 如果启用了 fp8 但当前环境不支持（如非 CUDA 或 PyTorch 无 fp8 支持），则跳过
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # 用于绘图横轴的参数名（矩阵维度）
            # 测试从 256 到 4096（步长为128）的方阵尺寸
            x_vals=[128 * i for i in range(2, 33)],  
            line_arg="provider",  # 决定绘图中不同曲线的参数
            # 如果是 fp8 模式，PyTorch 的 matmul 不支持 fp8，因此不与 cuBLAS/rocBLAS 比较
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # 曲线标签
            styles=[("green", "-"), ("blue", "-")],  # 曲线样式（颜色和线型）
            ylabel="TFLOPS",  # 纵轴标签：每秒万亿次浮点运算
            # 图表名称，也用作保存文件的名称
            plot_name="matmul-performance-" + ("fp16" if not fp8_inputs else "fp8"),
            args={"fp8_inputs": fp8_inputs},  # 传递给 benchmark 函数的额外参数
        )
    )


# 使用 Triton 的性能报告装饰器，将 configs 中的配置应用到 benchmark 函数
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    # 创建随机 float16 输入矩阵
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    # 如果启用了 fp8 且环境支持，转换为 fp8 格式（注意 b 要转置后再转 fp8）
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)

    # 定义性能采样的分位数（中位数、下20%、上80%）
    quantiles = [0.5, 0.2, 0.8]

    # 根据 provider 选择运行参考库（cuBLAS/rocBLAS）还是 Triton 内核
    if provider == ref_lib.lower():
        # 使用 PyTorch 的 matmul（底层调用 cuBLAS 或 rocBLAS）
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        # 调用自定义的 Triton matmul 内核
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)

    # 将运行时间（毫秒）转换为 TFLOPS
    # GEMM 的 FLOP 数为 2 * M * N * K
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    # 返回中位性能、最差（max_ms 对应最低性能）、最好（min_ms 对应最高性能）
    return perf(ms), perf(max_ms), perf(min_ms)


# 执行基准测试，并显示图表和数据
benchmark.run(show_plots=True, print_data=True)