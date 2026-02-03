# # 习题ddp_overlap_bucketed的实现
# # 参考自https://github.com/chaser682/cs336-assignment2-systems/blob/main/tests/adapters.py
# # pytest -k test_ddp.py

import torch
import torch.distributed as dist
import torch.nn as nn

class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.buckets = []
        self.param_to_bucket_info = {}
        
        self._broadcast_parameters_and_buffers()
        self._init_buckets()
        self._register_hooks()

    def _broadcast_parameters_and_buffers(self):
        # 确保所有 rank 的初始权重一致，否则训练必定发散
        src_rank = 0
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=src_rank)

    def _init_buckets(self):
        params = [p for p in self.module.parameters() if p.requires_grad]
        # 按照 backward 执行顺序（逆序）排列，有助于重叠通信与计算
        params.reverse()
        
        current_bucket_params = []
        current_bucket_size = 0
        target_bucket_bytes = self.bucket_size_mb * 1024 * 1024

        for param in params:
            param_bytes = param.numel() * param.element_size()
            # 简单的分桶逻辑
            if current_bucket_size + param_bytes > target_bucket_bytes and current_bucket_params:
                self._create_bucket(current_bucket_params)
                current_bucket_params = []
                current_bucket_size = 0
            current_bucket_params.append(param)
            current_bucket_size += param_bytes

        if current_bucket_params:
            self._create_bucket(current_bucket_params)

    def _create_bucket(self, params):
        total_numel = sum(p.numel() for p in params)
        dtype = params[0].dtype
        device = params[0].device
        
        # 创建扁平 Bucket
        bucket_tensor = torch.zeros(total_numel, dtype=dtype, device=device)
        
        bucket_idx = len(self.buckets)
        current_offset = 0
        
        for param in params:
            numel = param.numel()
            self.param_to_bucket_info[param] = (bucket_idx, current_offset, numel)
            current_offset += numel
            
        self.buckets.append({
            "tensor": bucket_tensor,     # 扁平的张量，存储所有梯度
            "params": params,            # 桶中的参数列表
            "handle": None,              # 异步通信句柄
            "ready_params": 0,           # 已就绪的参数数量
            "total_params": len(params)  # 总参数数量
        })

    def _prepare_for_backward(self):
        # 每个 Batch 开始前调用，重置计数器和 Tensor
        for bucket in self.buckets:
            bucket["ready_params"] = 0
            bucket["handle"] = None
            bucket["tensor"].zero_() 

    def _register_hooks(self):
        world_size = dist.get_world_size()
        
        def get_hook(param):
            # 使用闭包捕获 param
            def hook(grad):
                # 1. 查找位置
                if param not in self.param_to_bucket_info:
                    # 防御性编程：如果 param 虽然 requires_grad 但未被归入 bucket（极少见）
                    return grad
                    
                bucket_idx, offset, numel = self.param_to_bucket_info[param]
                bucket = self.buckets[bucket_idx]
                
                # 2. Copy-in: 将梯度拷入 Bucket
                # grad 可能是非连续的，view(-1) 可能会失败，使用 reshape 或 flatten 更安全，
                # 但通常 autograd 传出的 grad 是连续的。
                bucket["tensor"][offset : offset + numel].copy_(grad.detach().view(-1))
                
                bucket["ready_params"] += 1
                
                # 3. 触发通信
                if bucket["ready_params"] == bucket["total_params"]:
                    # 除以 world_size 做平均
                    bucket["tensor"].div_(world_size)
                    # 异步 AllReduce
                    bucket["handle"] = dist.all_reduce(
                        bucket["tensor"], op=dist.ReduceOp.SUM, async_op=True
                    )
                
                # 返回 grad 以保持 PyTorch 默认行为（虽然会被我们在 step 前覆盖）
                return grad
            return hook

        for param in self.module.parameters():
            if param.requires_grad:
                # 注册 hook
                param.register_hook(get_hook(param))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # 同步所有通信，并 Copy-out
        for bucket in self.buckets:
            if bucket["handle"]:
                bucket["handle"].wait()
            
            current_offset = 0
            for param in bucket["params"]:
                numel = param.numel()
                grad_data = bucket["tensor"][current_offset : current_offset + numel]
                
                # 此时 bucket 中的数据已经是 全局平均梯度
                # 将其写回 param.grad，供 optimizer 使用
                if param.grad is None:
                    # 显式 detach 避免建立计算图
                    param.grad = grad_data.view(param.shape).clone().detach()
                else:
                    param.grad.copy_(grad_data.view(param.shape))
                
                current_offset += numel