# 分析表1中large模型在上下文长度为128、256和512时的正向传播、反向传播和优化器步骤

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

from datetime import datetime
from torch.autograd.profiler import record_function


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
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# 模型规模配置（来自表1）
MODEL_CONFIGS = {
    # 'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    # 'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    # 'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    # '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32}
}

def benchmark_model(args, model_size):
    vocab_size = 10000
    context_length = args.context_length
    config = MODEL_CONFIGS[model_size]
    d_model = config['d_model']
    d_ff = config['d_ff']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    
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

    train_data = np.random.randint(0, vocab_size, total_batch_size * max_seq_len)

    # 热身
    print(f"\nWarming up ({args.warmup_steps} steps)...")
    model.train()
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

    # 开始记录内存历史
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    forward_times=[]
    backward_times=[]
    print(f"\nMeasuring ({args.steps} steps)...")
    model.train()
    for step in range(args.steps):
        nvtx.range_push(f"step_{step}")

        x, y = get_batch(train_data, device_batch_size, max_seq_len, device)
        fwd_start = timeit.default_timer()
        with autocast_ctx:
            with nvtx.range("forward"):
                y_pred = model(x)
        loss = cross_entropy(y_pred, y)
        loss = loss / args.steps
        forward_times.append(timeit.default_timer() - fwd_start)

        # # # 关键修改点1: 在正向传播完成后立即保存快照
        # if step == args.steps - 1:  # 只在最后一步保存，避免生成太多文件
        #     torch.cuda.memory._dump_snapshot(f"forward_only_{model_size}_{context_length}.pickle")
        #     print(f"Saved forward-only memory snapshot for step {step}")
        
        bwd_start = timeit.default_timer()
        with nvtx.range("backward"):
            loss.backward()
        
        backward_times.append(timeit.default_timer() - bwd_start)
        
        # gradient clipping
        grad_clip_enabled = grad_clip > 0.0  # grad_clip_enabled = False
        if grad_clip_enabled:
            grad_norm_tensor = gradient_clipping(model.parameters(), grad_clip)
        with nvtx.range("optimizer_step"):
            optimizer.step()
        model.zero_grad(set_to_none=True)
        nvtx.range_pop()
    
    # 保存Pickle文件，供PyTorch在线工具加载
    torch.cuda.memory._dump_snapshot(f"full_{model_size}_{context_length}.pickle")
    # 停止记录历史
    torch.cuda.memory._record_memory_history(enabled=None)

    forward_stats = compute_stats(forward_times)
    backward_stats = compute_stats(backward_times)

    # 计算吞吐量
    tokens_per_batch = device_batch_size * max_seq_len
    total_time_mean = forward_stats['mean'] + backward_stats['mean']
    throughput_tokens = tokens_per_batch / total_time_mean
    throughput_samples = device_batch_size / total_time_mean
    
    # 打印结果
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)

    print(f"\nResults for {model_size}:")
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
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    return {
        'model_size': model_size,
        'forward_mean_ms': forward_stats['mean'] * 1000,
        'forward_std_ms': forward_stats['std'] * 1000,
        'forward_cv': forward_stats['cv'],
        'backward_mean_ms': backward_stats['mean'] * 1000,
        'backward_std_ms': backward_stats['std'] * 1000,
        'backward_cv': backward_stats['cv'],
        'total_mean_ms': total_time_mean * 1000,
        'throughput_tokens': throughput_tokens,
        'parameters': total_params
    }

def trace_handler(prof: torch.profiler.profile):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"visual_mem_{timestamp}"

    prof.export_chrome_trace(f"{file_name}.json")
    prof.export_memory_timeline(f"{file_name}.html", device="cuda:0")

def train_html(args, model_size):
    vocab_size = 10000
    context_length = args.context_length
    config = MODEL_CONFIGS[model_size]
    d_model = config['d_model']
    d_ff = config['d_ff']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    

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

    train_data = np.random.randint(0, vocab_size, total_batch_size * max_seq_len)

    # 热身
    print(f"\nWarming up ({args.warmup_steps} steps)...")
    model.train()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=trace_handler,
    ) as prof:
        for step in range(args.warmup_steps):
            prof.step()
            x, y = get_batch(train_data, device_batch_size, max_seq_len, device)
            with record_function("## forward ##"):
                # with autocast_ctx:
                #     y_pred = model(x)
                y_pred = model(x)
            with record_function("## backward ##"):
                loss = cross_entropy(y_pred, y)
                loss = loss / args.warmup_steps
                loss.backward()
            # gradient clipping
            grad_clip_enabled = grad_clip > 0.0  # grad_clip_enabled = False
            if grad_clip_enabled:
                grad_norm_tensor = gradient_clipping(model.parameters(), grad_clip)

            with record_function("## optimizer ##"):
                optimizer.step()
                model.zero_grad(set_to_none=True)


    forward_times=[]
    backward_times=[]
    print(f"\nMeasuring ({args.steps} steps)...")
    model.train()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=trace_handler,
    ) as prof:
        for step in range(args.steps):
            prof.step()
            x, y = get_batch(train_data, device_batch_size, max_seq_len, device)
            fwd_start = timeit.default_timer()
            with record_function("## forward ##"):
                # with autocast_ctx:
                #     y_pred = model(x)
                y_pred = model(x)
            with record_function("## backward ##"):
                loss = cross_entropy(y_pred, y)
                loss = loss / args.steps
                forward_times.append(timeit.default_timer() - fwd_start)
                
                bwd_start = timeit.default_timer()
                loss.backward()
            
            backward_times.append(timeit.default_timer() - bwd_start)
            
            # gradient clipping
            grad_clip_enabled = grad_clip > 0.0  # grad_clip_enabled = False
            if grad_clip_enabled:
                grad_norm_tensor = gradient_clipping(model.parameters(), grad_clip)
            with record_function("## optimizer ##"):
                optimizer.step()
                model.zero_grad(set_to_none=True)

    forward_stats = compute_stats(forward_times)
    backward_stats = compute_stats(backward_times)

    # 计算吞吐量
    tokens_per_batch = device_batch_size * max_seq_len
    total_time_mean = forward_stats['mean'] + backward_stats['mean']
    throughput_tokens = tokens_per_batch / total_time_mean
    throughput_samples = device_batch_size / total_time_mean
    
    # 打印结果
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)

    print(f"\nResults for {model_size}:")
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
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    return {
        'model_size': model_size,
        'forward_mean_ms': forward_stats['mean'] * 1000,
        'forward_std_ms': forward_stats['std'] * 1000,
        'forward_cv': forward_stats['cv'],
        'backward_mean_ms': backward_stats['mean'] * 1000,
        'backward_std_ms': backward_stats['std'] * 1000,
        'backward_cv': backward_stats['cv'],
        'total_mean_ms': total_time_mean * 1000,
        'throughput_tokens': throughput_tokens,
        'parameters': total_params
    }


def main():
    parser = argparse.ArgumentParser(description='Transformer模型基准测试')
    
    # 模型参数
    parser.add_argument('--context-length', type=int, default=256,
                       help='上下文长度')
    parser.add_argument('--warmup-steps', type=int, default=5,
                       help='热身步骤数')
    parser.add_argument('--steps', type=int, default=5,
                       help='计时步骤数')
    args = parser.parse_args()
    
    # 对large规模模型进行基准测试
    results = []
    for model_size in ['large']:
        # result = benchmark_model(args, model_size)
        result = train_html(args, model_size)
        results.append(result)
    
    # 打印汇总表
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL MODEL SIZES")
    print(f"{'='*80}")
    print(f"{'Model':<10} {'Params(M)':<10} {'Fwd(ms)':<12} {'Fwd CV':<8} {'Bwd(ms)':<12} {'Bwd CV':<8} {'Total(ms)':<12} {'Tokens/s':<12}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['model_size']:<10} {r['parameters']/1e6:<10.1f} "
              f"{r['forward_mean_ms']:<12.2f} {r['forward_cv']:<8.3f} "
              f"{r['backward_mean_ms']:<12.2f} {r['backward_cv']:<8.3f} "
              f"{r['total_mean_ms']:<12.2f} {r['throughput_tokens']:<12.0f}")
    
    print(f"\nKey Observations:")
    print(f"1. Forward pass times increase with model size")
    print(f"2. Backward pass times are typically 2-3x longer than forward pass")
    print(f"3. Coefficient of Variation (CV) indicates stability:")
    print(f"   - CV < 0.05: Excellent stability")
    print(f"   - CV 0.05-0.10: Good stability")
    print(f"   - CV 0.10-0.20: Moderate variability")
    print(f"   - CV > 0.20: High variability")

if __name__ == "__main__":
    main()