import pyopencl as ocl
import numpy as np
import time
import argparse


NAIVE_IMPLEMENTATION_SOURCE = "naive_implementation.cl"


def positive_int(string):
    value = int(string)
    if value < 0:
        raise argparse.ArgumentTypeError("Invalid {}".format(value))
    return value


def matmul_possible(a_columns, b_rows):
    if a_columns != b_rows:
        msg = "Cannot multiply matrices: {} != {}".format(a_columns, b_rows)
        raise ValueError(msg)


def args_parse():
    parser = argparse.ArgumentParser(
        description="Multiplication of two matrices.")
    parser.add_argument("A_ROWS",
                        type=positive_int,
                        help="Number of rows of the first matrix")
    parser.add_argument("A_COLUMNS",
                        type=positive_int,
                        help="Number of columns of the first matrix")
    parser.add_argument("B_ROWS",
                        type=positive_int,
                        help="Number of rows of the second matrix")
    parser.add_argument("B_COLUMNS",
                        type=positive_int,
                        help="Number of columns of the second matrix")
    parser.add_argument('-t', "--time", action="store_true",
                        help="""
                        Prints out time measurements in the following order:
                        copying buffers onto GPU, GPU computing time,
                        copying buffers off GPU
                        """)
    parser.add_argument('-p', "--precision", action="store_true",
                        help="""
                        Measure floating point computed differences between
                        numpy.matmul result and GPU operation printing:
                        (LOW,HIGH)
                        """)
    parser.add_argument('-n', "--time-numpy-matmul", action="store_true",
                        help="""
                        Prints out time measurements for numpy.matmul
                        """)
    args = parser.parse_args()
    matmul_possible(args.A_COLUMNS, args.B_ROWS)
    return args


def interval(a, b):
    low = 0
    high = 0
    for i in range(0, len(a)):
        diff = a[i] - b[i]
        diff_high = np.amax(diff)
        diff_low = np.amin(diff)
        if diff_high > high:
            high = diff_high
        if diff_low < low:
            low = diff_low
        del diff
    return (low, high)


def read_program(context, filename):
    with open(filename, "r") as f:
        program = ocl.Program(context, f.read()).build()
    return program


def matrix_mult(a, b, c, a_dimensions, b_dimensions,
                platform, devices, context, program, queue, args):
    # define buffers
    a_buffer = ocl.Buffer(context, flags=ocl.mem_flags.READ_ONLY,
                          size=a.nbytes)
    b_buffer = ocl.Buffer(context, flags=ocl.mem_flags.READ_ONLY,
                          size=b.nbytes)
    c_buffer = ocl.Buffer(context, flags=ocl.mem_flags.WRITE_ONLY,
                          size=c.nbytes)

    start = time.time()
    # copying data onto GPU
    copy_a_event = ocl.enqueue_copy(queue, src=a, dest=a_buffer)
    copy_b_event = ocl.enqueue_copy(queue, src=b, dest=b_buffer)
    copy_a_event.wait()
    copy_b_event.wait()
    end = time.time()
    if args.time:
        print(end - start)

    # running program
    kernel_arguments = (a_buffer, b_buffer, c_buffer,
                        a_dimensions[1], b_dimensions[1])

    start = time.time()
    program.matrix_mult(queue,
                        np.array([a_dimensions[0], b_dimensions[1]]),
                        None,
                        *kernel_arguments).wait()
    end = time.time()
    if args.time:
        print(end - start)

    # copying data off GPU
    start = time.time()
    ocl.enqueue_copy(queue, src=c_buffer, dest=c).wait()
    end = time.time()
    if args.time:
        print(end - start)


def main():
    args = args_parse()
    # init matrices as 1D arrays => opencl does not deal with 2D arrays
    a_dimensions = (np.uint32(args.A_ROWS), np.uint32(args.A_COLUMNS))
    b_dimensions = (np.uint32(args.B_ROWS), np.uint32(args.B_COLUMNS))
    a = np.random.rand(a_dimensions[0] * a_dimensions[1]).astype(np.float32)
    b = np.random.rand(b_dimensions[0] * b_dimensions[1]).astype(np.float32)
    c = np.zeros(a_dimensions[0] * b_dimensions[1], dtype=np.float32)

    # init opencl
    platform = ocl.get_platforms()[0]
    devices = platform.get_devices()
    context = ocl.Context(devices=devices)
    program = read_program(context, NAIVE_IMPLEMENTATION_SOURCE)
    queue = ocl.CommandQueue(context)

    # run program
    matrix_mult(a, b, c, a_dimensions, b_dimensions,
                platform, devices, context, program, queue, args)
    c.shape = (a_dimensions[0], b_dimensions[1])

    # verify output
    if args.precision or args.time_numpy_matmul:
        a.shape = a_dimensions
        b.shape = b_dimensions
        start = time.time()
        c_expected = np.matmul(a, b)
        end = time.time()
        if args.time_numpy_matmul:
            print(end - start)
        if args.precision:
            precision = interval(c, c_expected)
            print(precision)


if __name__ == "__main__":
    main()
