from mpi4py import MPI
from coinrun.main_utils import mpi_print

rank = MPI.COMM_WORLD.Get_rank()
print(rank)
a = 138
b = 'dasdfaf'
c = ['dajkd','dahdua','dsab']
d = ('afhjla',384,2378)
e = {
        'afhjla':384,
        'sdasd':'dahsud'
    }
mpi_print(a)
mpi_print(b)
mpi_print(c)
mpi_print(d)
mpi_print(e)

