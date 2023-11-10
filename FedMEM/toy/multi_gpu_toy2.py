import os
import torch
import torch.distributed as dist

from torch.multiprocessing import Process


def train(rank, size):
    """Distributed function to be implemented later."""
    pass

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '5000'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, train))
        p.start()
        processes.append(p)
    print(processes)

    for p in processes:
        p.join()