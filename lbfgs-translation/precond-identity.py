from typing import Optional
import numpy as np
import unittest

class PrecondIdentity:
    """
    Identity preconditioner that returns input unchanged.
    Base class for preconditioners used in optimization.
    """
    
    def apply(self, x: np.ndarray, z: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply the identity preconditioner to input vector.
        
        Args:
            x: Input vector to precondition
            z: Additional vector (unused in identity preconditioner)
            
        Returns:
            The input vector x unchanged
        """
        return x


class TestPrecondIdentity(unittest.TestCase):
    def setUp(self):
        """Initialize preconditioner for each test"""
        self.precond = PrecondIdentity()
        
    def test_vector_unchanged(self):
        """Test that vectors are returned unchanged"""
        # Test 1D array
        x = np.array([1.0, 2.0, 3.0])
        result = self.precond.apply(x)
        np.testing.assert_array_equal(result, x)
        
        # Test 2D array
        x_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        result_2d = self.precond.apply(x_2d)
        np.testing.assert_array_equal(result_2d, x_2d)
        
    def test_unused_z_parameter(self):
        """Test that z parameter is properly ignored"""
        x = np.array([1.0, 2.0, 3.0])
        z = np.array([4.0, 5.0, 6.0])
        result = self.precond.apply(x, z)
        np.testing.assert_array_equal(result, x)
        
    def test_different_array_types(self):
        """Test with different types of numpy arrays"""
        # Integer array
        x_int = np.array([1, 2, 3], dtype=np.int32)
        result_int = self.precond.apply(x_int)
        np.testing.assert_array_equal(result_int, x_int)
        
        # Float array
        x_float = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        result_float = self.precond.apply(x_float)
        np.testing.assert_array_equal(result_float, x_float)

if __name__ == '__main__':
    unittest.main()

