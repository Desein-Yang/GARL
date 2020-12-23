from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank % 2 == 0:
    a = 2
    is_test_rank = 1
    print(rank,"I am not odd")
else:
    is_test_rank = 0
    a = 1
    print(rank,"I am odd")
Train_test_comm  = comm.Split(1 if is_test_rank else 0,0)

def aver(value,comm):
    size = comm.size
    x = np.array(value)
    buf = np.zeros_like(x)
    comm.Allreduce(x, buf, op=MPI.SUM)
    buf = buf / size
    return buf

c = aver(a,Train_test_comm)
print(type(c))
