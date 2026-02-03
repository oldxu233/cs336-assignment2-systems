# 习题naive_ddp_benchmarking的实现
# cd ~/codes/cs336/assignment2-systems-main
# python -m tests.LM_ddp

# 迭代5次，atol=1e-7 可以通过测试。但是atol=1e-8不行, why?
# 迭代10次，atol=1e-7 也不可以通过测试了
import logging
from copy import deepcopy
from typing import Type

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import adamw
from cs336_basics.nn_utils import cross_entropy
import torch.optim as optim
from cs336_systems.ddp.individual_ddp import DDP
from cs336_systems.ddp.buketed_ddp import DDPBucketed
# from cs336_systems.ddp.sharded_optimizer import ShardedOptimizer

from .common import (
    _cleanup_process_group,
    _setup_process_group,
    validate_ddp_net_equivalence,
)
import numpy as np
logger = logging.getLogger(__name__)



def _test_DistributedDataParallelIndividualParameters(rank: int, world_size: int, model_class: Type[torch.nn.Module]):
    atol = 1e-7
    epoch = 5
    # Use gloo backend for CPU
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    # Execute barrier prior to running test to ensure that every process
    # has finished initialization and that the following test
    # immediately exiting due to a skip doesn't cause flakiness.
    dist.barrier()

    # Seed to ensure that ranks are initialized with different initial models.
    torch.manual_seed(rank)
    non_parallel_model = model_class().to(device)

    ddp_base = deepcopy(non_parallel_model)
    # ddp_model = DDP(ddp_base)
    ddp_model = DDPBucketed(ddp_base, bucket_size_mb=100)

    # Make sure all the ranks have the same model state
    validate_ddp_net_equivalence(ddp_model)

    # Load the dataset from disk, so we can ensure that every rank has the same
    # overall pool of data.
    # Shape: (32, 256)
    all_x = torch.load("tests/ddp_test_data.pt")
    # Shape: (32, 256)
    all_y = torch.load("tests/ddp_test_labels.pt")
    print(f"all_x: {all_x.size()}, all_y: {all_y.size()}")

    assert all_x.size(0) % world_size == 0, "batch_size必须能被world_size整除"
    local_bs = int(all_y.size(0) / world_size)

    loss_fn = cross_entropy
    
    # Optimizer for the DDP model
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    # Optimizer for the non-parallel model
    non_parallel_optimizer = optim.SGD(non_parallel_model.parameters(), lr=0.1)

    for i in range(epoch):
        ddp_model._prepare_for_backward() # for buketed_ddp
        ddp_optimizer.zero_grad()
        non_parallel_optimizer.zero_grad()

        # Run the non-parallel model on all the data and take a gradient step
        non_parallel_data = all_x.to(device)
        non_parallel_labels = all_y.to(device)
        non_parallel_outputs = non_parallel_model(non_parallel_data)
        non_parallel_loss = loss_fn(non_parallel_outputs, non_parallel_labels)
        non_parallel_loss.backward()
        non_parallel_optimizer.step()

        # At this point, the parameters of non-parallel model should differ
        # from the parameters of the DDP model (since we've applied the
        # gradient step to the non-parallel model, but not to the DDP model).
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                if non_parallel_model_parameter.requires_grad and ddp_model_parameter.requires_grad:
                    # The only parameters that change are those that require_grad
                    assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
                else:
                    # parameters that don't require_grad shouldn't change
                    assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

        # While the non-parallel model does a forward pass on all the data (32 examples),
        # each DDP rank only sees 16 (disjoint) examples.
        # However, the end result should be the same as doing a forward pass on all 16 examples.
        offset = rank * local_bs
        ddp_data = all_x[offset : offset + local_bs, :].to(device)
        ddp_labels = all_y[offset : offset + local_bs, :].to(device)
        ddp_outputs = ddp_model(ddp_data)
        ddp_loss = loss_fn(ddp_outputs, ddp_labels)
        ddp_loss.backward()

        # Run student-written code that needs to execute after the backward pass,
        # but before the optimizer step (e.g., to wait for all DDP ranks to sync gradients)
        ddp_model.finish_gradient_synchronization()

        ddp_optimizer.step()

        # At this point, the non-parallel model should exactly match the parameters of the DDP model
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter, atol=atol)
        
        # Shuffle the data so that during the next iteration, each DDP rank sees a different set of inputs.
        # We make sure to use the same seed when shuffling (else the per-rank examples might not be disjoint).
        torch.manual_seed(42 + i)
        shuffle_idxs = torch.randperm(all_x.size(0))
        all_x = all_x[shuffle_idxs]
        all_y = all_y[shuffle_idxs]

    # After training is done, we should have the same weights on both the non-parallel baseline
    # and the model trained with DDP.
    if rank == 0:
        for non_parallel_model_parameter, ddp_model_parameter in zip(
            non_parallel_model.parameters(), ddp_model.parameters()
        ):
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter, atol=atol)
    _cleanup_process_group()
    

def GetDataset(dataset, batch_size, context_length, device):
    n = len(dataset) - context_length
    ix = np.random.randint(0, n, size=batch_size)

    # 分配一次大缓冲（最快、最干净）
    x = np.empty((batch_size, context_length), dtype=np.uint16)
    y = np.empty((batch_size, context_length), dtype=np.uint16)

    for k, i in enumerate(ix):
        x[k] = dataset[i:i+context_length]
        y[k] = dataset[i+1:i+context_length+1]

    # 直接转 tensor 并放到 device
    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)
    return x, y

def save_pt():
    mmap_file_valid = "/home/xqzzz1/codes/cs336/assignment1-basics-main/cs336_basics/data/TinyStoriesV2-GPT4-valid.memmap"
    train_data = np.memmap(mmap_file_valid, dtype=np.uint16, mode="r")
    batch_size = 32  # 必须能被world_size整除
    max_seq_len = 256  # 最大上下文长度（序列长度）
    all_x, all_y = GetDataset(train_data, batch_size, max_seq_len, "cpu") # 每个计算节点将只能看到16个样本（数据集总数为32个）
    torch.save(all_x, "ddp_test_data.pt")
    torch.save(all_y, "ddp_test_labels.pt")


if __name__ == "__main__":
    # save_pt()

    depth = 4          # Transformer 模型深度
    max_seq_len = 256  # 最大上下文长度（序列长度）
    num_layers = depth
    d_model = 512 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
    num_heads = 16
    d_ff = 1344
    vocab_size = 10000

    from functools import partial
    model_constructor = partial(
        TransformerLM,
        vocab_size=vocab_size,
        context_length=max_seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000.0
    )

    world_size = 2
    mp.spawn(
        _test_DistributedDataParallelIndividualParameters,
        args=(world_size, model_constructor),
        nprocs=world_size,
        join=True,
    )