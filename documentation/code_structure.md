# Code and file Structure

# Folder Structure
+ `./data` 
    + Holds the data required by code
+ `./documentation`
    + Holds the markdown style documentation files
    + Holds the Doxyfile (configuration file for doxygen) which can be used to create documentation with doxygen.
+ `./scripts`
    + This folder contains automation `bat` scripts
    + `generate_documents.bat` generates doxygen documents
    + `compile_cy.bat` compiles python code to low level `C++` code
    + `run_unittest.bat` runs unit tests
    + `run_benchmark.bat` runs benchmarks and reports results
+ `./source`
    + Contains the python and cython source code
    + `data_handling.py` file for handling data
    + `unit_tests.py` file for unit tests

