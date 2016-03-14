
# Writer Identification
+ Image Processing project at Uni-Hamburg
+ `Author: Praveen Baburao Kulkarni`

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
```

### Boost for windows
```bat
git submodule add https://github.com/boostorg/boost.git third_party/boost
cd third_party\boost_1_60_0
bootstrap.bat
b2
bjam
```

### PyCUDA
+ Make sure that vcvarsall.bat and amd64/cl.exe is in PATH
+ Make sure that CUDA bin, lib and include directories are in PATH
```bat
pip install pycuda
```

### Intel math kernel libraries (optional)
Used pro licence of anaconda for using `intel mkl libraries`. If not available then it can use default math libraries provided by visual studio `cl` compiler.
```bat
conda install accelerate
```

## Data-set 
Data-set is used from Kaggle competition ([ICFHR 2012 - Arabic Writer Identification](https://www.kaggle.com/c/awic2012/data))
The Data-set is downloaded and placed in data folder with some modifications.
Folder structure of data-set is as below:
+ `./data` The data folder
+ `./data/images` The images of handwritten manuscript
+ `./data/features` The features of handwritten manuscript images (stored in `.csv` format)
+ Matlab code is used to extract features (code can be found in `./source/feature_extraction`)

## Source code

## Documentation
