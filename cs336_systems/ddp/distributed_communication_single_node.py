import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import numpy as np

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if backend == "nccl":
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, tensor_size, backend="gloo"):
    setup(rank, world_size, backend)
    
    if torch.cuda.is_available() and backend == "nccl":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        device_type = "gpu"
    else:
        device = torch.device("cpu")
        device_type = "cpu"

    num_elements = int(tensor_size * 1024 * 1024 / 4) #float32占用4字节
    data = torch.randn(num_elements, dtype=torch.float32, device=device)

    # warmup 5次
    for _ in range(5):
        dist.all_reduce(data, async_op=False)
        if device_type == "gpu":
            torch.cuda.synchronize()
    
    times = []
    for _ in range(10):
        if device_type == "gpu":
            torch.cuda.synchronize()
        start_time = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        if device_type == "gpu":
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    
    # 收集所有rank的时间
    local_times = torch.tensor([np.mean(times)], device=device)
    gathered_times = [torch.zeros_like(local_times) for _ in range(world_size)]
    dist.all_gather(gathered_times, local_times)
    
    # 清理
    dist.destroy_process_group()
    
    if rank == 0:
        all_times = [t.item() for t in gathered_times]
        mean_time_ms = np.mean(all_times)
        bandwidth = calculate_bandwidth(tensor_size, mean_time_ms)
        print(
            
            f"结果: 平均时间={mean_time_ms:.2f}ms, "
            f"进程数={world_size},"
            f"内存大小={tensor_size},"
            f"带宽={bandwidth:.2f}Gbps")

def calculate_bandwidth(size_mb, time_ms):
    """计算带宽 (Gbps)"""
    # 总数据量 = 大小(MB) * 1024 * 1024 * 8 bits * 2 (发送+接收)
    total_bits = size_mb * 1024 * 1024 * 8 * 2
    time_sec = time_ms / 1000
    return (total_bits / time_sec) / 1e9  # 转换为Gbps

if __name__ == "__main__":
    world_sizes = [2, 4, 6]
    SIZES_MB = [1, 10, 100, 1024]  # 1MB, 10MB, 100MB, 1GB
    # SIZES_MB = [1]
    for world_size in world_sizes:
        for size_mb in SIZES_MB:
            mp.spawn(fn=distributed_demo, args=(world_size, size_mb, "nccl" if torch.cuda.is_available() else "gloo"), nprocs=world_size, join=True)