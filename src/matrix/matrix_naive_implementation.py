import numpy as np

N = int(1e6)         # Nb of elements in matrix
NCOL = int(1e3)      # Nb matrix columns
NLIN = int(1e3)      # Nb matrix lines


def compatible(a, b):
    """
    Check if matrix multiplication is doable.

    Return tuple with (row, column, common) if doable else None
    """
    if b.shape[0] != a.shape[1]:
        return None
    return (a.shape[0], b.shape[1], b.shape[0])


def matrix_mult(a, b):
    dimensions = compatible(a, b)
    assert(dimensions is not None)
    c = np.zeros([dimensions[0], dimensions[1]], dtype=np.float32)
    for i in range(0, dimensions[0]):
        for j in range(0, dimensions[1]):
            for k in range(0, dimensions[2]):
                c[i, j] = c[i, j] + (a[i, k] * b[k, j])
    return c


def main():
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    a.shape = (NCOL, NLIN)
    b.shape = (NCOL, NLIN)
    c = matrix_mult(a, b)


if __name__ == "__main__":
    main()
