
set -x

NODE_RANK=$1
NUM_ON_THIS=$2
GLOO_SOCKET_IFNAME=$3
if [ x$GLOO_SOCKET_IFNAME = x ]
then GLOO_SOCKET_IFNAME=enp129s0f0
fi

OMP_NUM_THREADS=4 NCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME NCCL_DEBUG=INFO GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME torchrun --start_method=fork --master_addr 162.105.88.71 --master_port 22426 --node_rank $NODE_RANK --nproc_per_node $NUM_ON_THIS --nnodes 2 ddptest.py
