# Log

## 20/05:00:00: research Python version and venv
- **numba**: [requires](http://numba.pydata.org/numba-doc/latest/user/installing.html) 
Python 3.6 or newer and Numby version 1.15 or newer (TODO: works with intel cpu?)
    - installation: `pip install numba`
- **PyOpenCL**: [requires](https://wiki.tiker.net/PyOpenCL/Installation/Linux) 
Python 2.4 or newer and and OpenCL implementation
    - installation: `pip install pyopencl`

## 20/05:00:16: automate environment configuration (venv, pip, Makefile)
## 20/05:01:10: research OpenCL and attempt to install - did not work and made OS unusable
## 20/05:10:00: install Ubuntu and configure
## 21/05:11:56: OpenCL installation - did not work
- research [beignet](https://github.com/intel/beignet) (said to be buggy)
- install openCL linkers: `sudo apt install ocl-icd-*`
- install libdrm: `sudo apt install libdrm libdrm-dev libdrm-intel1`
- attempt at installing from source [LLVM](https://llvm.org/) version 3.7 
([installation guide](https://releases.llvm.org/3.7.0/docs/GettingStarted.html))

## 22/05:11:00: OpenCl installation - more research, no results
## 24/05:18:00: OpenCl installation - worked
- step by step explanation:
    - install the headers provided by *Khronos*: `sudo apt install ocl-icd-*`
    - download *Intel*'s OpenCL [runtimes](https://software.intel.com/content/www/us/en/develop/articles/opencl-drivers.html):
    two are required, one of the *Graphics Technology* section, the other of the *Processor* section
    - installation of **Graphics runtime**: 
    ```
    mkdir neo
    cd neo
    wget https://github.com/intel/compute-runtime/releases/download/19.14.12751/intel-gmmlib_19.1.1_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/19.14.12751/intel-igc-core_19.11.1622_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/19.14.12751/intel-igc-opencl_19.11.1622_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/19.14.12751/intel-opencl_19.14.12751_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/19.14.12751/intel-ocloc_19.14.12751_amd64.deb
    sudo apt install ./*deb
    ```
    - installation of **CPU runtime**:
    ```sh
    tar -xvf l_opencl_p_18.1.0.015.tgz
    cd l_opencl_p_18.1.0.015/rpm
    # requires alien and libnuma1 - converts everything to deb packages
    alien *.rpm
    sudo apt install ./*deb
    # makes directories for vendors
    sudo mkdir -p /usr/lib/OpenCL/vendors/
    sudo mv /opt/intel /usr/lib/OpenCL/vendors/
    # paths may vary
    sudo cp /usr/lib/x86_64-linux-gnu/libOpenCL.so /usr/lib/OpenCL/vendors/intel/libOpenCL.so
    # configure dynamic linking
    sudo echo "/usr/lib/OpenCL/vendors/intel" > /etc/ld.so.conf.d/opencl-vendor-intel.conf
    sudo ldconfig
    ```
    - testing installation:
    ```
    sudo apt install clinfo
    # if no devices are detected it probably did not work
    clinfo
    ```
- guides used:
    - [streamhpc.com](https://streamhpc.com/blog/2011-06-24/install-opencl-on-debianubuntu-orderly/)
    - [askubuntu.com](https://askubuntu.com/questions/796770/how-to-install-libopencl-so-on-ubuntu)
    - [intel.com](https://software.intel.com/content/www/us/en/develop/articles/opencl-drivers.html)

## 24/05:21:00: do the log and readme
