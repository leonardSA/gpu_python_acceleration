import pyopencl as ocl
import numpy as np

A_NCOL = 1e3        # Nb matrix columns
A_NLIN = 1e3        # Nb matrix lines
B_NCOL = 1e3
B_NLIN = 1e3

LOW = int(-1e6)     # Random bounds
HIGH = int(1e6)


def matrix_mult_program(context):
    program_source = """
        __kernel void matrix_mult(__global int * a,
                                  __global int * b,
                                  __global int * c,
                                  const unsigned int a_ncol,
                                  const unsigned int b_ncol) {
            int rows = get_global_id(0);    /* iterate over rows */
            int columns = get_global_id(1); /* then iterate over columns */

            /* compute value */
            int value = 0;
            for (unsigned int i = 0 ; i < a_ncol ; i++) {
                value += a[rows * a_ncol + i] * b[i * b_ncol + columns];
            }

            c[rows * b_ncol + columns] = value;
        }
    """
    return ocl.Program(context, program_source).build()


def matrix_mult(a, b, c, a_dimensions, b_dimensions,
                platform, devices, context, program, queue):
    # define buffers
    a_buffer = ocl.Buffer(context, flags=ocl.mem_flags.READ_ONLY,
                          size=a.nbytes)
    b_buffer = ocl.Buffer(context, flags=ocl.mem_flags.READ_ONLY,
                          size=b.nbytes)
    c_buffer = ocl.Buffer(context, flags=ocl.mem_flags.WRITE_ONLY,
                          size=c.nbytes)

    # copying data onto GPU
    ocl.enqueue_copy(queue, src=a, dest=a_buffer)
    ocl.enqueue_copy(queue, src=b, dest=b_buffer)

    # running program
    kernel_arguments = (a_buffer, b_buffer, c_buffer,
                        a_dimensions[1], b_dimensions[1])

    program.matrix_mult(queue,
                        np.array([a_dimensions[0], b_dimensions[1]]),
                        None,
                        *kernel_arguments)

    # copying data off GPU
    copy_off_event = ocl.enqueue_copy(queue, src=c_buffer, dest=c)
    copy_off_event.wait()


def main():
    # init matrixes as 1D arrays => opencl does not deal with 2D arrays
    a_dimensions = (np.uint32(A_NLIN), np.uint32(A_NCOL))
    b_dimensions = (np.uint32(B_NLIN), np.uint32(B_NCOL))
    assert(a_dimensions[1] == b_dimensions[0])
    # a = np.random.rand(a_dimensions[0] * a_dimensions[1]).astype(np.int32)
    # b = np.random.rand(b_dimensions[0] * b_dimensions[1]).astype(np.int32)

    a = np.random.randint(LOW, high=HIGH,
                          size=a_dimensions[0] * a_dimensions[1],
                          dtype=np.int32)
    b = np.random.randint(LOW, high=HIGH,
                          size=b_dimensions[0] * b_dimensions[1],
                          dtype=np.int32)
    c = np.zeros(a_dimensions[0] * b_dimensions[1], dtype=np.int32)

    # init opencl
    platform = ocl.get_platforms()[0]
    devices = platform.get_devices()
    context = ocl.Context(devices=devices)
    program = matrix_mult_program(context)
    queue = ocl.CommandQueue(context)

    # run program
    matrix_mult(a, b, c, a_dimensions, b_dimensions,
                platform, devices, context, program, queue)
    c.shape = (a_dimensions[0], b_dimensions[1])

    # verify output
    a.shape = a_dimensions
    b.shape = b_dimensions
    c_expected = np.matmul(a, b)

    for i in range(0, len(c)):
        for j in range(0, len(c[i])):
            assert(c[i][j] == c_expected[i][j])


if __name__ == "__main__":
    main()
