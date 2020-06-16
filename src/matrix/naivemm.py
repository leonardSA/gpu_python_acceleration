import sys
import time
import numpy as np
import pyopencl as cl

LOW = int(-1e6)     # Random integer bounds
HIGH = int(1e6)

CL_INT32_IMPLEMENTATION = "naive_integer_implementation.cl"
CL_FLOAT32_IMPLEMENTATION = "naive_float_implementation.cl"


class NaiveMatrixMultiplication:

    a = None
    b = None
    c = None
    time_buffer_to_GPU = -1.0
    time_buffer_from_GPU = -1.0
    time_execution = -1.0
    __program = None

    def __init__(self,
                 cl_items: dict, a_dim: tuple, b_dim: tuple, dtype: np.dtype):
        """
        Instantiate MatrixMultiplication object.

        Parameters
        ---
        cl_items:
            Dictionnary containing keys:
                - platform
                - devices
                - context
                - queue
        a_dim :
            Dimensions of matrix a
        b_dim :
            Dimensions of matrix b
        dtype :
            Type of data (e.g. np.float32)
            Supported types: np.float32, np.int32
        """

        self.cl_items = cl_items
        self.a_dim = (np.uint32(a_dim[0]), np.uint32(a_dim[1]))
        self.b_dim = (np.uint32(b_dim[0]), np.uint32(b_dim[1]))
        self.dtype = dtype
        self.__init_matrices()
        self.__init_program()
        self.__to2Darray()

    def compute(self):
        """
        Runs the OpenCL program.
        """
        self.__to1Darray()
        buffers = self.__create_buffers()

        # copying data onto GPU
        start = time.time()
        cl.enqueue_copy(self.cl_items['queue'], src=self.a, dest=buffers[0])
        cl.enqueue_copy(self.cl_items['queue'], src=self.b, dest=buffers[1])
        end = time.time()
        self.time_buffer_to_GPU = end - start

        pass
        # running program
        kernel_arguments = (buffers[0], buffers[1], buffers[2],
                            self.a_dim[1], self.b_dim[1])
        start = time.time()
        self.__program.matrix_mult(self.cl_items['queue'],
                                   [self.a_dim[0], self.b_dim[1]],
                                   None,
                                   *kernel_arguments).wait()
        end = time.time()
        self.time_execution = end - start

        # copying data off GPU
        start = time.time()
        cl.enqueue_copy(self.cl_items['queue'], src=buffers[2], dest=self.c)
        end = time.time()
        self.time_buffer_from_GPU = end - start
        self.__to2Darray()

    def accuracy(self):
        """
        """

        if self.time_execution < 0:
            self.compute()

        c_np = np.matmul(self.a, self.b)

        low = 0
        high = 0
        for i in range(0, len(c_np)):
            diff = c_np[i] - self.c[i]
            diff_high = np.amax(diff)
            diff_low = np.amin(diff)
            if diff_high > high:
                high = diff_high
            if diff_low < low:
                low = diff_low
            del diff
        return (low, high)

    def __init_matrices(self):
        """
        Initializes matrices a and b with random numbers, and c with zeroes.

        Raises
        ---
        TypeError :
            Type is not supported.
        """

        if np.issubdtype(self.dtype, np.int32):
            self.a = np.random.randint(LOW, high=HIGH,
                                       size=self.a_dim[0] * self.a_dim[1],
                                       dtype=self.dtype)
            self.b = np.random.randint(LOW, high=HIGH,
                                       size=self.b_dim[0] * self.b_dim[1],
                                       dtype=self.dtype)
        elif np.issubdtype(self.dtype, np.float32):
            self.a = np.random.rand(
                self.a_dim[0] * self.a_dim[1]).astype(self.dtype)
            self.b = np.random.rand(
                self.b_dim[0] * self.b_dim[1]).astype(self.dtype)
        else:
            msg = "Unsupported type: {}".format(self.dtype)
            raise TypeError(msg)
        self.c = np.zeros(self.a_dim[1] * self.b_dim[0], dtype=self.dtype)

    def __init_program(self):
        """
        Initializes the program accordingly to self.dtype.

        Raises
        ---
        TypeError :
            Type is not supported.
        """

        if np.issubdtype(self.dtype, np.int32):
            self.__read_program(CL_INT32_IMPLEMENTATION)
        elif np.issubdtype(self.dtype, np.float32):
            self.__read_program(CL_FLOAT32_IMPLEMENTATION)
        else:
            msg = "Unsupported type: {}".format(self.dtype)
            raise TypeError(msg)

    def __read_program(self, filename: str):
        """
        Reads the kernel defined program source, builds it and stores it
        in self.program.

        Parameters
        ---
        filename :
            Path to file containing the Kernel.

        Raises
        ---
        OSError :
            Error when opening file.
        cl.Error :
            Error with defined Kernel in filename.
        """

        try:
            with open(filename, "r") as f:
                self.__program = cl.Program(self.cl_items['context'],
                                            f.read()).build()
        except (OSError, cl.Error) as e:
            print(e)
            sys.exit(1)

    def __create_buffers(self):
        """
        Creates the buffers for the matrices.

        Returns
        ---
        tuple
            Tuple containing buffer for matrix a, buffer for matrix b
            and buffer for matrix c
        """
        a_buffer = cl.Buffer(self.cl_items['context'],
                             flags=cl.mem_flags.READ_ONLY,
                             size=self.a.nbytes)
        b_buffer = cl.Buffer(self.cl_items['context'],
                             flags=cl.mem_flags.READ_ONLY,
                             size=self.b.nbytes)
        c_buffer = cl.Buffer(self.cl_items['context'],
                             flags=cl.mem_flags.WRITE_ONLY,
                             size=self.c.nbytes)
        return (a_buffer, b_buffer, c_buffer)

    def __to1Darray(self):
        self.a = self.a.ravel()
        self.b = self.b.ravel()
        self.c = self.c.ravel()

    def __to2Darray(self):
        self.a.shape = (self.a_dim[0], self.a_dim[1])
        self.b.shape = (self.b_dim[0], self.b_dim[1])
        self.c.shape = (self.a_dim[0], self.b_dim[1])
