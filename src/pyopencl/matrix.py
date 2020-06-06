import pyopencl as ocl
import numpy as np
import sys


def argsCheck():
    assert(len(sys.argv) == 5)
    for i in range(1, len(sys.argv)):
        for c in sys.argv[i]:
            assert(c.isdigit())


def interval(a, b):
    assert(a.shape == b.shape)
    low = 0
    high = 0
    for i in range(0, len(a)):
        for j in range(0, len(a[i])):
            diff = a[i][j] - b[i][j]
            if diff > high:
                high = diff
            if diff < low:
                low = diff
    return (low, high)


def matrix_mult_program(context):
    program_source = """
        __kernel void matrix_mult(__global float *a,
                                  __global float* b,
                                  __global float* c,
                                  const unsigned int a_ncol,
                                  const unsigned int b_ncol) {
            int rows = get_global_id(0);    /* iterate over rows */
            int columns = get_global_id(1); /* then iterate over columns */

            /* compute value */
            float value = 0;
            for (unsigned int i = 0 ; i < a_ncol ; i++) {
                value += a[rows * a_ncol + i] * b[i * b_ncol + columns];
            }

            c[rows * b_ncol + columns] = value;
        }
    """
    return ocl.Program(context, program_source).build()


def matrix_mult(a, b, c, a_dimensions, b_dimensions,
                platform, devices, context, program, queue):
    # TODO time transfer
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

    # TODO time computation
    program.matrix_mult(queue,
                        np.array([a_dimensions[0], b_dimensions[1]]),
                        None,
                        *kernel_arguments)

    # copying data off GPU
    copy_off_event = ocl.enqueue_copy(queue, src=c_buffer, dest=c)
    copy_off_event.wait()


def main():
    argsCheck()
    # init matrixes as 1D arrays => opencl does not deal with 2D arrays
    a_dimensions = (np.uint32(sys.argv[1]), np.uint32(sys.argv[2]))
    b_dimensions = (np.uint32(sys.argv[3]), np.uint32(sys.argv[4]))
    assert(a_dimensions[1] == b_dimensions[0])
    a = np.random.rand(a_dimensions[0] * a_dimensions[1]).astype(np.float32)
    b = np.random.rand(b_dimensions[0] * b_dimensions[1]).astype(np.float32)
    c = np.zeros(a_dimensions[0] * b_dimensions[1], dtype=np.float32)

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
    precision = interval(c, c_expected)
    if precision[0] != 0 or precision[1] != 0:
        print("Floating points differences: ", precision)


if __name__ == "__main__":
    main()
