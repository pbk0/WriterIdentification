"""
This file hods Unit tests for different functionality of project.
"""
import unittest


class TestDataHandling(unittest.TestCase):
    """
    Class for unit testing functions in data_handling.py
    """

    # noinspection PyBroadException
    def test_load_features_and_labels(self):
        """
        test_load_features_and_labels
        :return:
        :rtype:
        """
        try:
            from .data_handling import load_features_and_labels
            _train, _train_label, _test, _test_label, _feature_name = load_features_and_labels()
        except Exception:
            self.fail('Failed to test_load_features_and_labels')


class TestPyCUDA(unittest.TestCase):
    """
    Class to check PyCUDA functionlity
    """

    def test_pycuda(self):
        """
        Test pycuda installation with small example.
        :return:
        :rtype:
        """
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            from pycuda.compiler import SourceModule
            import numpy as np
            a = np.random.randn(4, 4)
            print(a)
            a= a.astype(np.float32)
            a_gpu = cuda.mem_alloc(a.nbytes)
            cuda.memcpy_htod(a_gpu, a)
            mod = SourceModule(
                """
                __global__ void doublify(float *a)
                {
                int idx = threadIdx.x + threadIdx.y*4;
                a[idx] *= 2;
                }
                """
            )
            func = mod.get_function("doublify")
            func(a_gpu, block=(4,4,1))
            a_doubled = np.empty_like(a)
            cuda.memcpy_dtoh(a_doubled, a_gpu)
            #print(a_doubled)
            #print(a)
        except Exception:
            self.fail('Still not working')
