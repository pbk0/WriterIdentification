# Environment and installing instructions

## Environment
+ Currently tested for 
    + Windows 10 and Ubuntu 14.04 
    + Python 3.5.1

## Dependencies

### Python Libraries (mandatory)
```bat
source activate py3
conda install python=3.5
conda install scikit-learn
conda install scikit-image
conda install cython
conda install pandas
conda install seaborn
```

### Boost and more dependencies
```bat
git submodule add https://github.com/boostorg/boost.git third_party/boost
git submodule update --init --recursive
rem  git submodule foreach git merge origin master
cd third_party
cd boost
bootstrap.bat
b2
bjam
cd ..
cd ..
move third_party/boost/stage/lib third_party/boost_lib
```

### PyCUDA
+ Make sure that vcvarsall.bat and amd64/cl.exe is in PATH
+ Make sure that CUDA bin, lib and include directories are in PATH
+ [Reference blog](https://kerpanic.wordpress.com/2015/09/28/pycuda-windows-installation-offline/)

```bat
git submodule add http://git.tiker.net/trees/pycuda.git third_party/pycuda
git submodule update --init --recursive
cd third_party
cd pycuda
python configure.py
```

+ siteconf.py file will be generated please update it with

```py
BOOST_INC_DIR = []
BOOST_LIB_DIR = []
BOOST_COMPILER = 'msvc'
USE_SHIPPED_BOOST = True
BOOST_PYTHON_LIBNAME = ['libboost_python3-vc140-mt-1_61']
BOOST_THREAD_LIBNAME = ['libboost_thread-vc140-mt-1_61']
CUDA_TRACE = False
CUDA_ROOT = 'D:\\InstalledPrograms\\Nvidia\\CUDA\\v75'
CUDA_ENABLE_GL = False
CUDA_ENABLE_CURAND = True
CUDADRV_LIB_DIR = ['${CUDA_ROOT}/lib', '${CUDA_ROOT}/lib/x64']
CUDADRV_LIBNAME = ['cuda']
CUDART_LIB_DIR = ['${CUDA_ROOT}/lib', '${CUDA_ROOT}/lib/x64']
CUDART_LIBNAME = ['cudart']
CURAND_LIB_DIR = ['${CUDA_ROOT}/lib', '${CUDA_ROOT}/lib/x64']
CURAND_LIBNAME = ['curand']
CXXFLAGS = ['/EHsc']
LDFLAGS = ['/FORCE']
```

+ Next

```bat
python setup.py build
python setup.py install
```

+ Next update `nvcc.profile` in cuda installation bin directory to:

```
INCLUDES        +=  "-I$(TOP)/include" "-I$(TOP)/include/cudart" "-ID:\InstalledPrograms\Nvidia\CUDA\v75\include" $(_SPACE_)
```

### Intel math kernel libraries (optional)
Use pro licence of anaconda for using `intel mkl libraries`. If not available then it can use default math libraries provided by visual studio `cl` compiler.
```bat
conda install accelerate
```