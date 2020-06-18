# gpu_python_acceleration

## Requirements

Minimum requirements:
- Linux  
- OpenCL == 2.1  
- python >= 3.8.2  

For `make environment-setup`:   
- python3 venv  
- pip   

For `make graphs`: gnuplot   

For `make doc`: 
- latex  
- bibtex

## Make instructions

`make environment-setup`: sets up the python virtual environment.  
`make graphs`: execute scripts and graphs up data.  
`make doc`: compiles the slides and report into pdfs.  
`make clean`: remove output directories.    

## Adapt the code

A few things can be adjusted in the code to use it on your GPU.   
First the choice of the platform in **./src/matrix/main.py**:
```python
# Modify this line in order to choose other platforms 
# Here we are choosing the first platform (make sure to list your platforms first)
cl_items['platform'] = cl.get_platforms()[0]
```
Second the choice of the type in **./src/matrix/main.py**:
```python
# Only the naive matrix supports two types: np.int32 and np.float32
m = nmm.MatrixMultiplication(cl_items,
                             (args.A_ROWS, args.A_COLUMNS),
                             (args.B_ROWS, args.B_COLUMNS),
                             np.float32)    # you can use np.int32 instead
```
Third, the number of *work groups* in **./src/matrix/mm.py**:
```python
# Modify accordingly to the number of compute units on your GPU
# Needs to be a power of 2
N_WORK_GROUPS = 32  
# Side note: for 32 work groups you must have at least of the dimensions 
# in one of your matrix > 16
# for 64 > 32
# for 128 > 64 and so on
```

## Running the main *a la mano*


Printing help:  
```
$ python3 main.py -h
usage: main.py [-h] [-t] [-p] [-n] [-v] [--naive] A_ROWS A_COLUMNS B_ROWS B_COLUMNS

Multiplication of two matrices.

positional arguments:
  A_ROWS                Number of rows of the first matrix
  A_COLUMNS             Number of columns of the first matrix
  B_ROWS                Number of rows of the second matrix
  B_COLUMNS             Number of columns of the second matrix

optional arguments:
  -h, --help            show this help message and exit
  -t, --time            Prints out time measurements in the following order: copying buffers onto GPU, GPU computing time, copying buffers off GPU
  -p, --accuracy        Measure floating point computed differences between numpy.matmul result and GPU operation printing: (LOW,HIGH)
  -n, --time-numpy-matmul
                        Prints out time measurements for numpy.matmul
  -v, --verbose         Adds text with the measures.
  --naive               Use the naive implementation.
```

Executing for matrices A(200, 200) * B(200, 200):
```
$ python3 main.py 200 200 200 200 -t -n -p -v
Buffer copy time onto GPU: 0.00013113021850585938
Matrix multiplication execution time: 0.048624277114868164
Buffer copy time off GPU: 0.00010395050048828125
Numpy matrix multiplication execution time: 0.0004436969757080078
Accuracy lower bound: 0
Accuracy higher bound: 0
```

Executing for matrices A(20, 150) * B(150, 40) using the naive version and without verbose:
```
$ python3 main.py 20 150 150 40 -t -n -p --naive
7.2479248046875e-05
0.003324270248413086
3.743171691894531e-05
9.965896606445312e-05
0
0
```
