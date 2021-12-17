import numpy as np
from scipy.spatial import KDTree


def nearest_neighbor(streamlines, tractogram, verbose=True, workers=-1):
    """Compute the nearest neighbor of the flattened streamline and the
    flipped flattened streamlines with respect to the given
    tractogram. The code uses a KDTree, so the tractogram can be
    passed also as a KDTree (useful for repeated use of this
    function).
    """
    if verbose: print("Flattening streamlines into vectors")
    x = streamlines.reshape(len(streamlines), -1)
    if verbose: print("Flattening flipped streamlines into vectors")
    x_flip = streamlines[:, ::-1, :].reshape(len(streamlines), -1)
    if verbose: print("Creating a KDTree with the flattened tractogram")
    if type(tractogram) is not KDTree:
        kdt = KDTree(tractogram.reshape(len(tractogram), -1))  # Memory intensive!
    else:
        if verbose: print("Tractogram is already a KDTree")

    distances = np.zeros((len(streamlines), 2), dtype=np.float32)
    indices = np.zeros((len(streamlines), 2), dtype=int)
    if verbose: print("Querying streamlines (as vectors)")
    distances[:, 0], indices[:, 0] = kdt.query(x, k=1, workers=-1)
    if verbose: print("Querying flipped streamlines (as vectors)")
    distances[:, 1], indices[:, 1] = kdt.query(x_flip, k=1,
                                               workers=workers)
    if verbose: print("Returning closest streamlines")
    tmp1 = range(len(streamlines))
    tmp2 = distances.argmin(1)
    return distances[tmp1, tmp2], indices[tmp1, tmp2]
