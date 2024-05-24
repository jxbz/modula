#     examples of calling this script:
#
#  >  torchrun --nproc_per_node=8 examples/multi-gpu.py
#  >  python -m torch.distributed.run --nproc_per_node=8 examples/multi-gpu.py
#  >  OMP_NUM_THREADS=10 python -m torch.distributed.run --nproc_per_node=8 examples/multi-gpu.py

from torch import randn, no_grad, manual_seed, get_num_threads
import torch.distributed as dist

from modula.atom import Linear
from modula.bond import ReLU

# distributed setup
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
print(f"rank {rank} \t world size {dist.get_world_size()} \t threads {get_num_threads()}")
dist.barrier()

# sample worker-specific data
manual_seed(rank)
data, target = randn(1000, device=rank), randn(10, device=rank)

# create model
mlp = Linear(10,10000) @ ReLU() @ Linear(10000, 1000)
weights = mlp.initialize(device=rank)
weights.broadcast() # synchronize weights across workers

# train model
for step in range(steps:=20):
    output = mlp(data, weights)

    loss = (target - output).square().mean()
    loss.backward()

    weights.grad().all_reduce() # average the raw gradients across workers

    with no_grad():
        manual_seed(0) # ensure spectral normalization uses the same seed across workers
        mlp.normalize(grad := weights.grad())
        weights -= 0.1 * grad
        weights.zero_grad()

        mlp.regularize(weights, strength = 0.01)

    if rank == 0:
        if step == 0: print("step", "\t", "train loss")
        print(step, '\t', loss.item())

# print final loss for each worker
dist.barrier()
print(f"rank {rank} \t loss {loss.item()}")
