import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP


def run(rank, local_rank):
    this_device = f"cuda:{4+local_rank}"
    master_device = "cuda:4"

    print(f"{rank} {local_rank} {os.getpid()} {this_device} {master_device}")

    model = nn.Linear(1, 1, bias=False, device=this_device)
    model = DDP(model, device_ids=[this_device])

    if os.path.exists("model.save"):
        model.load_state_dict(torch.load("model.save", map_location={master_device: this_device}))
        print('model loaded')

    dist.barrier()

    input_ = torch.ones([8, 1], dtype=torch.float32, device=this_device)
    label_ = torch.ones([8], dtype=torch.float32, device=this_device) * float(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for i in range(1000):
        optimizer.zero_grad()
        output = model(input_).squeeze(-1)
        if i == 0:
            print(output[0])
        loss = nn.MSELoss()(output, label_)
        loss.backward()
        optimizer.step()
    print(output[0])

    dist.barrier()

    if rank == 0:
        torch.save(model.state_dict(), "model.save")

@record
def main():
    parser = argparse.ArgumentParser()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group("nccl")
    run(rank, local_rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

