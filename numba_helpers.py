from numba import njit
from numpy.typing import ArrayLike


@njit    
def get_path_from_predecessors(predecessors: ArrayLike, end: int) -> list:
    path = [end]
    k = end
    while predecessors[k] != -9999:
        path.append(predecessors[k])
        k = predecessors[k]
    return path[-2::-1]
    