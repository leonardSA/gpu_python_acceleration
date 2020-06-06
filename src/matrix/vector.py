import pyopencl
import numpy

print(pyopencl.get_platforms())
intel_platform = pyopencl.get_platforms()[0]
intel_devices = intel_platform.get_devices()
intel_context = pyopencl.Context(devices=intel_devices)

program_source = """
    kernel void sum(global float *a, global float* b, global float* c) {
        int gid = get_global_id(0);
        c[gid] = a[gid] + b[gid];
    }
"""
intel_program_source = pyopencl.Program(intel_context, program_source)
intel_program = intel_program_source.build()
print(intel_program.get_info(pyopencl.program_info.KERNEL_NAMES))

intel_queue = pyopencl.CommandQueue(intel_context)

N = int(1e7)
a = numpy.random.rand(N).astype(numpy.float32)
b = numpy.random.rand(N).astype(numpy.float32)
c = numpy.empty_like(a)
a_intel_buffer = pyopencl.Buffer(intel_context,
                                 flags=pyopencl.mem_flags.READ_ONLY,
                                 size=a.nbytes)
b_intel_buffer = pyopencl.Buffer(intel_context,
                                 flags=pyopencl.mem_flags.READ_ONLY,
                                 size=b.nbytes)
c_intel_buffer = pyopencl.Buffer(intel_context,
                                 flags=pyopencl.mem_flags.WRITE_ONLY,
                                 size=c.nbytes)


def run_gpu_program():
    # copying data onto GPU
    pyopencl.enqueue_copy(intel_queue,
                          src=a,
                          dest=a_intel_buffer)
    pyopencl.enqueue_copy(intel_queue,
                          src=b,
                          dest=b_intel_buffer)

    # running program
    kernel_arguments = (a_intel_buffer, b_intel_buffer, c_intel_buffer)
    intel_program.sum(intel_queue,
                      a.shape,  # global size
                      None,     # local size
                      *kernel_arguments)

    # copying data off GPU
    copy_off_event = pyopencl.enqueue_copy(intel_queue,
                                           src=c_intel_buffer,
                                           dest=c)
    copy_off_event.wait()


def check_results(a, b, c):
    print(a, b, c)
    print(a[0] + b[0], c[0])
    s = a + b
    if ((c - (a + b)).all() > 0.0):
        print("result does not match")
    else:
        print("result matches")


run_gpu_program()
check_results(a, b, c)
