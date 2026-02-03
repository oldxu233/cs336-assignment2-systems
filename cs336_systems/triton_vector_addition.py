# https://triton-lang.org/main/getting-started/tutorials/index.html
import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector. 总元素数量
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # 有多个'程序'在处理不同的数据。我们在这里确定当前是哪个程序：
    pid = tl.program_id(axis=0) # 使用一维启动网格，所以轴为0
    # 该程序将处理从初始数据偏移的输入。
    # 例如，如果有一个长度为256的向量，block_size为64，那么各个程序
    # 将分别访问元素[0:64, 64:128, 128:192, 192:256]。
    # 注意offsets是一个指针列表：
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    #创建一个掩码，防止内存操作越界访问
    mask = offsets < n_elements
    # 从DRAM加载x和y，如果输入不是block_size的倍数，则屏蔽多余元素
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # 将 x+y 写回DRAM
    tl.store(output_ptr + offsets, output, mask=mask)

# 我们还需要声明一个辅助函数来：
# (1) 分配输出 z tensor
# (2) 使用适当的 grid/block sizes 将上述内核加入队列
def add(x: torch.Tensor, y: torch.Tensor):
    # 我们需要预分配输出张量
    output = torch.empty_like(x)
    # 确保所有张量都在指定设备上 （如GPU）
    assert x.device == y.device == output.device
    n_elements = output.numel()
    
    # SPMD启动网格表示并行运行的内核实例数量
    # 它类似于CUDA启动网格，可以是Tuple[int]或Callable(metaparameters) -> Tuple[int]
    # 这里我们使用一维网格，其大小是所需的块数量：
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # 注意：
    #  - 每个 torch.tensor 对象都会隐式转换为指向其首元素的指针
    #  - 使用启动网格索引`triton.jit`装饰的函数，可以获得可调用的GPU内核
    #  - 别忘了将 meta-parameters 作为关键字参数传递

    # 启动内核，每个线程块处理1024个元素
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    # 我们返回output的句柄，但由于尚未调用`torch.cuda.synchronize()`，
    # 此时内核仍在异步运行
    return output

# 现在我们可以使用上述函数来计算两个torch.tensor对象的逐元素和，并验证其正确性。
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')