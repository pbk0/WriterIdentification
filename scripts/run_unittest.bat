echo on
cd ..
set PYTHONPATH=%cd%
echo %PYTHON_PATH%
python source/unit_tests.py
cd scripts