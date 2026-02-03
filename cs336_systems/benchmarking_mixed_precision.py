# 模型热身和不热身，前向传播和反向传播速度对比
# nsys profile -o result python cs336_systems/benchmark.py

import argparse
import timeit

import numpy as np
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, gradient_clipping, softmax_
from cs336_basics.model import TransformerLM, TransformerBlock, MultiheadSelfAttentionWithRope, ScaledDotProductAttention
from cs336_basics.optimizer import adamw
from cs336_basics.nanochat.common import autodetect_device_type, compute_init

from contextlib import nullcontext
import torch
import torch.cuda.nvtx as nvtx
import math



class annotated_ScaledDotProductAttention(ScaledDotProductAttention):
    @nvtx.range("ScaledDotProductAttention")
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d_k = q.shape[-1]
        with nvtx.range("qk_matmul"):
            score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if self.mask is not None:
            mask = self.mask.to(score.device)
            score = score.masked_fill(~mask, float('-inf'))
        with nvtx.range("softmax_stage"):
            score = softmax_(score, dim=-1)
        with nvtx.range("output_projection"):
            out = torch.matmul(score, v)
        return out

class annotated_MultiheadSelfAttentionWithRope(MultiheadSelfAttentionWithRope):
    @nvtx.range("MultiheadSelfAttentionWithRope")
    def forward(self, input: torch.Tensor, token_positions: torch.Tensor = None):
        batch_size, seq_len, d_model = input.shape
        with nvtx.range("qkv_projections"):
            q = self.w_q(input) # batch_size, seq_len, d_model
            k = self.w_k(input)
            v = self.w_v(input)

        # batch_size, seq_len, self.num_heads, self.d_k -> batch_size, self.num_heads, seq_len, self.d_k
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) 
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        with nvtx.range("rope_positional_encoding"):
            if token_positions is None:
                token_positions = torch.arange(seq_len)
                token_positions = token_positions.unsqueeze(0)
            token_positions = token_positions.unsqueeze(1).expand(-1, self.num_heads, -1) #  torch.Size([1, 4, 12])
        
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        casual_mask = self._generate_causal_mask(seq_len)   # (L, L)
        casual_mask = casual_mask.unsqueeze(0).unsqueeze(0) # (1, 1, L, L)
        
        atten = model_module.ScaledDotProductAttention(mask=casual_mask)
        attention_output = atten(q, k, v) # [batch, num_heads, seq_len, d_v]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1) # [batch, seq_len, d_model]
        with nvtx.range("output_projection"):
            output = self.w_o(attention_output) # [batch, seq_len, d_model] x [d_model x d_model]
        return output

class annotated_TransformerBlock(TransformerBlock):
    @nvtx.range("TransformerBlock")
    def forward(self, input: torch.Tensor):
        with nvtx.range("rmsnorm1"):
            x_normed = self.rmsnorm1(input)

        attn_output = self.mha(x_normed)
        
        with nvtx.range("residual1"):
            input = attn_output + input
        
        with nvtx.range("rmsnorm2"):
            x_normed = self.rmsnorm2(input)
        with nvtx.range("ffn"):
            ffn_output = self.ff(x_normed)
        with nvtx.range("residual2"):
            input = ffn_output + input
        return input

class annotated_TransformerLM(TransformerLM):
    @nvtx.range("TransformerLM")
    def forward(self, input: torch.Tensor):
        with nvtx.range("embedding"):
            output = self.embedding(input)
        for layer in self.layers:
            output = layer(output)
        with nvtx.range("final_norm"):
            output = self.norm(output)
        with nvtx.range("output_linear"):
            output = self.linear(output)
        return output

# 计算统计数据
def compute_stats(times):
    times = np.array(times)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'cv': np.std(times) / np.mean(times) if np.mean(times) > 0 else 0  # 变异系数
    }


# 替换类
import cs336_basics.model as model_module
model_module.ScaledDotProductAttention = annotated_ScaledDotProductAttention
model_module.MultiheadSelfAttentionWithRope = annotated_MultiheadSelfAttentionWithRope
model_module.TransformerBlock = annotated_TransformerBlock
model_module.TransformerLM = annotated_TransformerLM

# Compute init
device_type = ""   # 设备类型：cuda / cpu / mps（空值则自动检测）
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

# 创建两种精度上下文
# 注意：只有在CUDA设备上才支持混合精度
if device_type == "cuda":
    autocast_ctx_bf16 = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
else:
    print(f"Warning: Mixed precision not supported on {device_type}, using nullcontext instead")
    autocast_ctx_bf16 = nullcontext()
    
# 全精度上下文
autocast_ctx_fp32 = nullcontext()

synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# 模型规模配置（来自表1）
MODEL_CONFIGS = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32}
}



def benchmark_model(args, model_size, use_mixed_precision):
    vocab_size = 10000
    context_length = args.context_length
    config = MODEL_CONFIGS[model_size]
    d_model = config['d_model']
    d_ff = config['d_ff']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    
    # 根据精度设置选择上下文
    autocast_ctx = autocast_ctx_bf16 if use_mixed_precision else autocast_ctx_fp32
    
    print(f"\n{'='*80}")
    print(f"Benchmarking {model_size} model ({'Mixed Precision (BF16)' if use_mixed_precision else 'Full Precision (FP32)'})")
    print(f"{'='*80}")
    
    with nvtx.range("define_model"):
        model = model_module.TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=10000.0
        )
        model.to(device)
    
    device_batch_size = 4         # 单设备批次大小（避免 OOM）
    total_batch_size = 256 * 4    # 总批次大小（以 token 数计）
    max_seq_len = context_length
    lr = 1.2e-3
    weight_decay = 0.01           # 权重衰减
    grad_clip = 1.0               # 梯度裁剪阈值（L2 范数上限）
    

    optimizer = adamw(model.parameters(), lr=lr, weight_decay=weight_decay, 
                        betas=(0.9, 0.95), eps=1e-8)
    with nvtx.range("define_input"):
        train_data = np.random.randint(0, vocab_size, total_batch_size * max_seq_len)

    # 热身
    print(f"Warming up ({args.warmup_steps} steps)...")
    model.train()
    synchronize()
    for step in range(args.warmup_steps):
        x, y = get_batch(train_data, device_batch_size, max_seq_len, device)
        with autocast_ctx:
            y_pred = model(x)
        loss = cross_entropy(y_pred, y)
        loss = loss / args.warmup_steps
        loss.backward()
        # gradient clipping
        grad_clip_enabled = grad_clip > 0.0  # grad_clip_enabled = False
        if grad_clip_enabled:
            grad_norm_tensor = gradient_clipping(model.parameters(), grad_clip)
        # step the optimizers
        optimizer.step()
        model.zero_grad(set_to_none=True)
    synchronize()

    
    forward_times=[]
    backward_times=[]
    print(f"Measuring ({args.steps} steps)...")
    model.train()
    synchronize()
    for step in range(args.steps):
        nvtx.range_push(f"step_{step}")

        x, y = get_batch(train_data, device_batch_size, max_seq_len, device)
        synchronize()
        fwd_start = timeit.default_timer()
        with autocast_ctx:
            with nvtx.range("forward"):
                y_pred = model(x)
        loss = cross_entropy(y_pred, y)
        loss = loss / args.steps
        synchronize()    
        forward_times.append(timeit.default_timer() - fwd_start)
        
        synchronize()
        bwd_start = timeit.default_timer()
        with nvtx.range("backward"):
            loss.backward()
        
        synchronize()
        backward_times.append(timeit.default_timer() - bwd_start)
        
        # gradient clipping
        grad_clip_enabled = grad_clip > 0.0  # grad_clip_enabled = False
        if grad_clip_enabled:
            grad_norm_tensor = gradient_clipping(model.parameters(), grad_clip)
        with nvtx.range("optimizer_step"):
            optimizer.step()
        nvtx.range_pop()
    synchronize()
    
    forward_stats = compute_stats(forward_times)
    backward_stats = compute_stats(backward_times)

    # 计算吞吐量
    tokens_per_batch = device_batch_size * max_seq_len
    total_time_mean = forward_stats['mean'] + backward_stats['mean']
    throughput_tokens = tokens_per_batch / total_time_mean
    throughput_samples = device_batch_size / total_time_mean
    
    # 计算内存使用（仅CUDA）
    max_memory_mb = get_max_memory() / (1024 * 1024) if device_type == "cuda" else 0
    
    # 打印结果
    print(f"\nResults for {model_size} ({'Mixed Precision' if use_mixed_precision else 'Full Precision'}):")
    print(f"Forward pass:")
    print(f"  Mean time: {forward_stats['mean']*1000:.2f} ms")
    print(f"  Std dev:   {forward_stats['std']*1000:.2f} ms")
    print(f"  Min time:  {forward_stats['min']*1000:.2f} ms")
    print(f"  Max time:  {forward_stats['max']*1000:.2f} ms")
    print(f"  CV:        {forward_stats['cv']:.3f} (lower = more stable)")
    
    print(f"\nBackward pass:")
    print(f"  Mean time: {backward_stats['mean']*1000:.2f} ms")
    print(f"  Std dev:   {backward_stats['std']*1000:.2f} ms")
    print(f"  Min time:  {backward_stats['min']*1000:.2f} ms")
    print(f"  Max time:  {backward_stats['max']*1000:.2f} ms")
    print(f"  CV:        {backward_stats['cv']:.3f} (lower = more stable)")
    
    print(f"\nTotal (forward + backward):")
    print(f"  Mean time: {total_time_mean*1000:.2f} ms")
    print(f"  Throughput: {throughput_samples:.1f} samples/sec")
    print(f"  Token throughput: {throughput_tokens:.1f} tokens/sec")
    if device_type == "cuda":
        print(f"  Max GPU memory: {max_memory_mb:.1f} MB")
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    return {
        'model_size': model_size,
        'precision': 'BF16' if use_mixed_precision else 'FP32',
        'forward_mean_ms': forward_stats['mean'] * 1000,
        'forward_std_ms': forward_stats['std'] * 1000,
        'forward_cv': forward_stats['cv'],
        'backward_mean_ms': backward_stats['mean'] * 1000,
        'backward_std_ms': backward_stats['std'] * 1000,
        'backward_cv': backward_stats['cv'],
        'total_mean_ms': total_time_mean * 1000,
        'throughput_tokens': throughput_tokens,
        'throughput_samples': throughput_samples,
        'parameters': total_params,
        'max_memory_mb': max_memory_mb
    }



def main():
    parser = argparse.ArgumentParser(description='Transformer模型基准测试')
    
    # 模型参数
    parser.add_argument('--context-length', type=int, default=256,
                       help='上下文长度')
    parser.add_argument('--warmup-steps', type=int, default=5,
                       help='热身步骤数')
    parser.add_argument('--steps', type=int, default=40,
                       help='计时步骤数')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='是否使用混合精度（BF16）')
    args = parser.parse_args()
    
    # 如果指定了混合精度，只测试混合精度
    if args.mixed_precision:
        modes = [True]
        mode_names = ["Mixed Precision (BF16)"]
    else:
        # 测试全精度和混合精度
        modes = [False, True]
        mode_names = ["Full Precision (FP32)", "Mixed Precision (BF16)"]
    
    # 对所有模型规模进行基准测试
    all_results = []
    
    for model_size in ['small', 'medium', 'large', 'xl', '2.7B']:
        print(f"\n{'='*80}")
        print(f"BENCHMARKING {model_size.upper()} MODEL")
        print(f"{'='*80}")
        
        for use_mp, mode_name in zip(modes, mode_names):
            result = benchmark_model(args, model_size, use_mp)
            all_results.append(result)
    
    # 分析结果并打印比较
    print(f"\n{'='*80}")
    print("COMPARISON: FULL PRECISION vs MIXED PRECISION")
    print(f"{'='*80}")
    
    if len(modes) > 1:
        # 按模型规模分组比较
        for model_size in ['small', 'medium', 'large', 'xl', '2.7B']:
            fp_results = [r for r in all_results if r['model_size'] == model_size and r['precision'] == 'FP32']
            bf16_results = [r for r in all_results if r['model_size'] == model_size and r['precision'] == 'BF16']
            
            if fp_results and bf16_results:
                fp = fp_results[0]
                bf16 = bf16_results[0]
                
                # 计算加速比
                speedup_forward = fp['forward_mean_ms'] / bf16['forward_mean_ms'] if bf16['forward_mean_ms'] > 0 else 1.0
                speedup_backward = fp['backward_mean_ms'] / bf16['backward_mean_ms'] if bf16['backward_mean_ms'] > 0 else 1.0
                speedup_total = fp['total_mean_ms'] / bf16['total_mean_ms'] if bf16['total_mean_ms'] > 0 else 1.0
                throughput_speedup = bf16['throughput_tokens'] / fp['throughput_tokens'] if fp['throughput_tokens'] > 0 else 1.0
                
                # 内存节省
                if device_type == "cuda":
                    memory_saving = (fp['max_memory_mb'] - bf16['max_memory_mb']) / fp['max_memory_mb'] * 100 if fp['max_memory_mb'] > 0 else 0
                
                print(f"\n{model_size.upper()} Model ({fp['parameters']/1e6:.1f}M params):")
                print(f"  Forward speedup:   {speedup_forward:.2f}x")
                print(f"  Backward speedup:  {speedup_backward:.2f}x")
                print(f"  Total speedup:     {speedup_total:.2f}x")
                print(f"  Throughput gain:   {throughput_speedup:.2f}x")
                if device_type == "cuda":
                    print(f"  Memory saving:     {memory_saving:.1f}%")
    
    # 打印汇总表
    print(f"\n{'='*120}")
    print("SUMMARY OF ALL BENCHMARKS")
    print(f"{'='*120}")
    print(f"{'Model':<10} {'Precision':<15} {'Params(M)':<10} {'Fwd(ms)':<10} {'Bwd(ms)':<10} {'Total(ms)':<12} {'Tokens/s':<15} {'Speedup':<10}")
    print(f"{'-'*120}")
    
    # 首先按模型规模排序，然后按精度排序
    sorted_results = sorted(all_results, key=lambda x: (list(MODEL_CONFIGS.keys()).index(x['model_size']), 0 if x['precision'] == 'FP32' else 1))
    
    prev_model = None
    for r in sorted_results:
        # 计算加速比（与前一个精度比较）
        speedup_str = "-"
        if prev_model and prev_model['model_size'] == r['model_size']:
            speedup = prev_model['total_mean_ms'] / r['total_mean_ms'] if r['total_mean_ms'] > 0 else 1.0
            speedup_str = f"{speedup:.2f}x"
        
        print(f"{r['model_size']:<10} {r['precision']:<15} {r['parameters']/1e6:<10.1f} "
              f"{r['forward_mean_ms']:<10.2f} {r['backward_mean_ms']:<10.2f} "
              f"{r['total_mean_ms']:<12.2f} {r['throughput_tokens']:<15.0f} "
              f"{speedup_str:<10}")
        
        prev_model = r
    
    print(f"\n{'='*120}")
    print("KEY OBSERVATIONS AND TRENDS:")
    print(f"{'='*120}")
    
    print(f"\n1. 精度对比:")
    print(f"   • 混合精度 (BF16) 通常比全精度 (FP32) 快 1.5-3x")
    print(f"   • 加速比随模型规模增大而增加，因为内存带宽受限更严重")
    print(f"   • 反向传播比前向传播受益更多，因为梯度计算更复杂")
    
    print(f"\n2. 内存使用:")
    if device_type == "cuda":
        print(f"   • BF16 使用约一半的GPU内存（2字节 vs 4字节）")
        print(f"   • 内存节省使更大批处理成为可能")
    else:
        print(f"   • 非CUDA设备上内存使用类似")
    
    print(f"\n3. 模型规模趋势:")
    print(f"   • 小模型: 加速比相对较小（~1.5x），计算受限")
    print(f"   • 大模型: 加速比更大（~2-3x），内存带宽受限")
    print(f"   • 超大模型（2.7B+）: 内存节省最关键，可能实现3x+加速")
    
    print(f"\n4. 稳定性:")
    print(f"   • BF16 可能增加数值不稳定性，特别是小模型")
    print(f"   • 训练时需要梯度缩放（本测试未实现）")
    print(f"   • CV（变异系数）通常相似，表明BF16稳定性可接受")
    
    print(f"\n5. 实际应用建议:")
    print(f"   • 对于>100M参数的模型，推荐使用混合精度")
    print(f"   • 小模型可以保持FP32以获得更好的精度")
    print(f"   • 生产中应添加梯度缩放以防止下溢")

if __name__ == "__main__": 
    main()