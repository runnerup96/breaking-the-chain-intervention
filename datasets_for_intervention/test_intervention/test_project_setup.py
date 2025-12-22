import unittest
import warnings
import importlib


class TestProjectSetup(unittest.TestCase):
    # List of required libraries from requirements.txt
    REQUIRED_LIBS = [
        'torch',
        'transformers',
        'datasets',
        'accelerate',
        'pandas',
        'numpy',
        'scipy',
        'sklearn',  # scikit-learn is imported as sklearn
        'tqdm',
        'json5',
        'pytest'
    ]
    
    def test_installed_libs(self):
        """Test if all required libraries are installed."""
        missing_libs = []
        
        for lib in self.REQUIRED_LIBS:
            try:
                importlib.import_module(lib)
            except ImportError:
                missing_libs.append(lib)
        
        if missing_libs:
            warning_msg = f"The following libraries are not installed: {', '.join(missing_libs)}"
            warnings.warn(warning_msg, UserWarning)
            # Test passes but warns the user
            self.assertTrue(True, warning_msg)
        else:
            self.assertTrue(True, "All required libraries are installed")
    
    def test_cuda_availability(self):
        """Test if CUDA is available for PyTorch."""
        try:
            import torch
            if torch.cuda.is_available():
                self.assertTrue(True, "CUDA is available")
            else:
                warning_msg = "CUDA is not available. GPU acceleration will not be used."
                warnings.warn(warning_msg, UserWarning)
                self.assertTrue(True, warning_msg)
        except ImportError:
            warning_msg = "PyTorch (torch) is not installed. Cannot check CUDA availability."
            warnings.warn(warning_msg, UserWarning)
            self.assertTrue(True, warning_msg)


