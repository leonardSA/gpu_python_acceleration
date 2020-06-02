import pyopencl as ocl
import numpy as np

# N = int(1e6)         # Nb of elements in matrix
# NCOL = int(1e3)      # Nb matrix columns
# NLIN = int(1e3)      # Nb matrix lines


def matrix_mult_program(context):
    program_source = """
        __kernel void matrix_mult(__global float *a,
                                  __global float* b,
                                  __global float* c,
                                  const unsigned int a_ncol,
                                  const unsigned int b_ncol) {
            int gid = get_global_id(0);
            c[gid] = 1;
        }
    """
    return ocl.Program(context, program_source).build()


def matrix_mult(a, b, c, a_col_size, b_col_size,
                platform, devices, context, program, queue):
    # define buffers
    a_buffer = ocl.Buffer(context, flags=ocl.mem_flags.READ_ONLY,
                          size=a.nbytes)
    b_buffer = ocl.Buffer(context, flags=ocl.mem_flags.READ_ONLY,
                          size=b.nbytes)
    c_buffer = ocl.Buffer(context, flags=ocl.mem_flags.WRITE_ONLY,
                          size=c.nbytes)
    a_col_size_buffer = ocl.Buffer(context, flags=ocl.mem_flags.READ_ONLY,
                                   size=a_col_size.nbytes)
    b_col_size_buffer = ocl.Buffer(context, flags=ocl.mem_flags.READ_ONLY,
                                   size=b_col_size.nbytes)

    # copying data onto GPU
    ocl.enqueue_copy(queue, src=a, dest=a_buffer)
    ocl.enqueue_copy(queue, src=b, dest=b_buffer)
    ocl.enqueue_copy(queue, src=a_col_size, dest=a_col_size_buffer)
    ocl.enqueue_copy(queue, src=b_col_size, dest=b_col_size_buffer)

    # running program
    kernel_arguments = (a_buffer, b_buffer, c_buffer,
                        a_col_size_buffer, b_col_size_buffer)
    program.matrix_mult(queue,
                        c.shape,  # global size
                        None,     # local size
                        *kernel_arguments)

    # copying data off GPU
    copy_off_event = ocl.enqueue_copy(queue, src=c_buffer, dest=c)
    copy_off_event.wait()


def main():
    # init matrixes as 1D arrays => opencl does not deal with 2D arrays
    a_dimensions = (np.int32(2), np.int32(2))
    b_dimensions = (np.int32(2), np.int32(2))
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
    matrix_mult(a, b, c, a_dimensions[1], b_dimensions[1],
                platform, devices, context, program, queue)
    c.shape = (a_dimensions[0], b_dimensions[1])
    for i in c:
        print(i)


if __name__ == "__main__":
    main()
