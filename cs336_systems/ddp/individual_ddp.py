# 习题ddp_overlap_individual_parameters的实现
# copy自https://github.com/heng380/cs336_assignment2
# pytest -k test_ddp_individual_parameters.py
import torch
import torch.distributed as dist
from torch.autograd.profiler import record_function


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        """
        分布式数据并行(DDP)的初始化函数
        参数:
            module (torch.nn.Module): 需要进行分布式训练的神经网络模块
        """
        super(DDP, self).__init__()  # 调用父类的初始化方法
        self.module = module  # 存储要分布式的模块
        self.handles = []    # 用于存储异步allreduce操作的句柄

        # initialize all parameters to be the same
        for param in self.module.parameters():  # 遍历模型的所有参数
            dist.broadcast(param.data, src=0)  # 将所有参数广播源为0的节点，确保所有进程参数一致
            if param.requires_grad:  # 如果参数需要梯度计算
                param.register_post_accumulate_grad_hook(self.transform_grad)  # 注册hook，这里注册的钩子会在梯度计算完成后自动调用transform_grad函数

    def transform_grad(self, param):
        # 使用torch.no_grad()上下文管理器，确保在此期间不计算梯度
        with torch.no_grad():
            # 将参数的梯度数据 除以 总进程数
            # 这样做的目的是在梯度聚合前先进行缩放，防止梯度值过大
            param.grad.data /= dist.get_world_size()

        # 使用record_function记录"allreduce_async"操作，用于性能分析
        with record_function("allreduce_async"):
            # 异步执行梯度全归约操作，将所有进程的梯度求和
            # 将操作句柄添加到self.handles列表中，以便后续可以检查操作是否完成
            self.handles.append(dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True))

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
    
    
    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)