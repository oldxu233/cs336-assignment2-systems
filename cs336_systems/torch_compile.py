import argparse
import timeit
import numpy as np
import torch
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.nanochat.common import autodetect_device_type, compute_init
from contextlib import nullcontext
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import adamw, get_lr_cosine_schedule


# 基准测试不同规模的注意力实现
def benchmark_attention(args, d_model, seq_len):
    device_type = ""   # 设备类型：cuda / cpu / mps（空值则自动检测）
    device_type = autodetect_device_type() if device_type == "" else device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    
    batch_size = 8
    lr = 1.2e-3                   # 初始学习率
    weight_decay = 0.01           # 权重衰减
    grad_clip = 1.0               # 梯度裁剪阈值（L2 范数上限）
    num_layers = 4
    num_heads = 1
    d_ff = 4*d_model

    max_learning_rate = 2e-3
    min_learning_rate = max_learning_rate / 20
    warmup_iters = 20           # 学习率预热步数
    cosine_cycle_iters = args.steps - warmup_iters
    vocab_size = 10000
    
    # 创建随机输入x, 随机输出y
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.int64)
    y = torch.randint(0, d_model, (batch_size, seq_len), device=device, dtype=torch.int64)
    
    model = TransformerLM(vocab_size=vocab_size, context_length=seq_len, d_model=d_model, 
                      num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=10000.0)
    model.to(device=device) # 将模型移动到指定设备
    model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
    adamw_optimizer = adamw(model.parameters(), lr=lr, weight_decay=weight_decay, 
                        betas=(0.9, 0.95), eps=1e-8)
    
    # 热身
    print(f"Warming up ({args.warmup_steps} steps)...")
    for _ in range(args.warmup_steps):
        with autocast_ctx:
            y_pred = model(x)
        loss = cross_entropy(y_pred, y)
        loss = loss / args.warmup_steps
        loss.backward()
        model.zero_grad(set_to_none=True)
    synchronize()
    
    # 计时正向传播
    print(f"Measuring forward and backward pass ({args.steps} steps)...")
    forward_times = []
    backward_times = []
    for step in range(args.steps):
        start_time = timeit.default_timer()
        with autocast_ctx:
            y_pred = model(x)
        synchronize()
        forward_times.append(timeit.default_timer() - start_time)
        
        loss = cross_entropy(y_pred, y)
        loss = loss / args.warmup_steps

        start_time = timeit.default_timer()
        loss.backward()
        # gradient clipping
        grad_clip_enabled = grad_clip > 0.0  # grad_clip_enabled = False
        if grad_clip_enabled:
            grad_norm_tensor = gradient_clipping(model.parameters(), grad_clip)
            grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note: cpu-gpu sync point)
        # step the optimizers
        lrm = get_lr_cosine_schedule(step, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        for param in adamw_optimizer.param_groups:
            param["lr"] = lrm
        adamw_optimizer.step()
        model.zero_grad(set_to_none=True)
        backward_times.append(timeit.default_timer() - start_time)
    synchronize()
        

    # 计算统计量
    def compute_stats(times):
        times = np.array(times)
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    
    forward_stats = compute_stats(forward_times)
    backward_stats = compute_stats(backward_times)
    
    # 报告结果
    print(f"\nResults:")
    print(f"Forward pass:")
    print(f"  Mean time: {forward_stats['mean']*1000:.2f} ms")
    print(f"  Std dev:   {forward_stats['std']*1000:.2f} ms")
    print(f"  Min time:  {forward_stats['min']*1000:.2f} ms")
    print(f"  Max time:  {forward_stats['max']*1000:.2f} ms")
    
    print(f"\nBackward pass:")
    print(f"  Mean time: {backward_stats['mean']*1000:.2f} ms")
    print(f"  Std dev:   {backward_stats['std']*1000:.2f} ms")
    print(f"  Min time:  {backward_stats['min']*1000:.2f} ms")
    print(f"  Max time:  {backward_stats['max']*1000:.2f} ms") 
    # 计算注意力内存使用量
    attention_memory_mb = (batch_size * seq_len * seq_len * 2) / (1024**2)  # 注意力分数矩阵（bfloat16）
    
    return {
        'd_model': d_model,
        'seq_len': seq_len,
        'forward_mean_ms': forward_stats['mean'] * 1000,
        'backward_mean_ms': backward_stats['mean'] * 1000,
        'attention_memory_mb': attention_memory_mb,
        'oom': False
    }

def main():
    parser = argparse.ArgumentParser(description='注意力机制基准测试')
    parser.add_argument('--warmup-steps', type=int, default=5,
                       help='热身步骤数')
    parser.add_argument('--steps', type=int, default=100,
                       help='计时步骤数')
    args = parser.parse_args()
    
    # 测试配置
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    
    results = []
    
    for d_model in d_models:
        for seq_len in seq_lens:
            try:               
                result = benchmark_attention(args, d_model, seq_len)
                results.append(result)
                
            except torch.cuda.OutOfMemoryError:
                print(f"!!! OUT OF MEMORY ERROR for d_model={d_model}, seq_len={seq_len} !!!")
                
    print(f"\n{'='*100}")
    print("SUMMARY OF ALL ATTENTION BENCHMARKS")
    print(f"{'='*100}")
    print(f"{'d_model':<10} {'seq_len':<10} {'Fwd(ms)':<12} {'Bwd(ms)':<12} {'Bwd/Fwd':<12} {'AttnMem(MB)':<12} {'Status':<15}")
    print(f"{'-'*100}")
    
    for result in results:
        d_model = result['d_model']
        seq_len = result['seq_len']
        forward_ms = result['forward_mean_ms']
        backward_ms = result['backward_mean_ms']
        attn_mem = result['attention_memory_mb']
        status = "OOM" if result['oom'] else "OK"
        
        if not result['oom'] and forward_ms > 0:
            ratio = backward_ms / forward_ms
            ratio_str = f"{ratio:.2f}"
        else:
            ratio_str = "N/A"
        
        if result['oom']:
            forward_str = "OOM"
            backward_str = "OOM"
        else:
            forward_str = f"{forward_ms:.2f}"
            backward_str = f"{backward_ms:.2f}"
        
        print(f"{d_model:<10} {seq_len:<10} {forward_str:<12} {backward_str:<12} {ratio_str:<12} {attn_mem:<12.1f} {status:<15}")
    
    print(f"{'-'*100}")
if __name__ == "__main__":
    main()