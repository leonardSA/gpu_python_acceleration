import time
import argparse
import numpy as np
import pyopencl as cl
import naivemm as nmm
import mm


def positive_int(string):
    """
    Positive integer type checker.

    Raises
    ---
    ArgumentTypeError
    """
    value = int(string)
    if value < 0:
        raise argparse.ArgumentTypeError("Invalid {}".format(value))
    return value


def matmul_possible(a_columns: int, b_rows: int):
    """
    Verifies if matrix multiplication AxB=C can be done.

    Parameters
    ---
    a_columns :
        Number of columns in matrix A
    b_rows    :
        Number of rows in matrix B

    Raises
    ---
    ValueError
    """

    if a_columns != b_rows:
        msg = "Cannot multiply matrices: {} != {}".format(a_columns, b_rows)
        raise ValueError(msg)


def args_parse():
    """
    Parse the command line arguments and return them.
    """

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
    parser.add_argument('-p', "--accuracy", action="store_true",
                        help="""
                        Measure floating point computed differences between
                        numpy.matmul result and GPU operation printing:
                        (LOW,HIGH)
                        """)
    parser.add_argument('-n', "--time-numpy-matmul", action="store_true",
                        help="""
                        Prints out time measurements for numpy.matmul
                        """)
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="""
                        Adds text with the measures.
                        """)
    parser.add_argument("--naive", action="store_true",
                        help="""
                        Use the naive implementation.
                        """)
    args = parser.parse_args()
    matmul_possible(args.A_COLUMNS, args.B_ROWS)
    return args


def main():
    args = args_parse()

    cl_items = {}
    cl_items['platform'] = cl.get_platforms()[0]
    cl_items['devices'] = cl_items['platform'].get_devices()
    cl_items['context'] = cl.Context(devices=cl_items['devices'])
    cl_items['queue'] = cl.CommandQueue(cl_items['context'])

    if args.naive:
                                     # dict with opencl configuration
        m = nmm.MatrixMultiplication(cl_items,
                                     # dimensions of A
                                     (args.A_ROWS, args.A_COLUMNS),
                                     # dimensions of B
                                     (args.B_ROWS, args.B_COLUMNS),
                                     # data type used
                                     np.float32)
    else:
        m = mm.MatrixMultiplication(cl_items,
                                    (args.A_ROWS, args.A_COLUMNS),
                                    (args.B_ROWS, args.B_COLUMNS),
                                    np.float32)
    m.compute()

    if args.time:
        if args.verbose:
            print("Buffer copy time onto GPU:", end=" ")
        print(m.time_buffer_to_GPU)
        if args.verbose:
            print("Matrix multiplication execution time:", end=" ")
        print(m.time_execution)
        if args.verbose:
            print("Buffer copy time off GPU:", end=" ")
        print(m.time_buffer_from_GPU)

    if args.time_numpy_matmul:
        start = time.time()
        np.matmul(m.a, m.b)
        end = time.time()
        if args.verbose:
            print("Numpy matrix multiplication execution time:", end=" ")
        print(end - start)

    if args.accuracy:
        accuracy = m.accuracy()
        if args.verbose:
            print("Accuracy lower bound:", end=" ")
        print(accuracy[0])
        if args.verbose:
            print("Accuracy higher bound:", end=" ")
        print(accuracy[1])


if __name__ == "__main__":
    main()
