# pytest tests/test_sharded_optimizer.py

import torch
import torch.distributed as dist
from typing import Any, Type

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        self.params = list(params)

        # 记录原始的参数配置，用于后续add_param_group
        self.optimizer = optimizer_cls

        # 2. 计算本rank负责的参数
        # 规则：第i个参数分配给 rank = i % world_size
        my_params = [p for i, p in enumerate(self.params) if dist.get_rank() == i % dist.get_world_size()]
        
        # 3. 创建本地优化器，只优化本rank负责的参数
        self.optimizer = optimizer_cls([{'params':my_params}], **kwargs)
        
        self.handles = []
        super().__init__(self.params, {})

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)
        self.synchronize_params()
        self.wait_for_all_params()


    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)

    def synchronize_params(self):
        for i, p in enumerate(self.params):
            rank = i % dist.get_world_size() # 计算该参数的所有者rank
            # 由参数所有者广播给所有其他rank
            self.handles.append(dist.broadcast(p.data, src=rank, async_op=True)) 

    def wait_for_all_params(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()