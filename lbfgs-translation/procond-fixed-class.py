import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from time import time

from abc import ABC, abstractmethod
from time import time

class Precond(ABC):
    """Abstract base class for preconditioners."""
    
    def __init__(self):
        self.elapsed_time = None
    
    @abstractmethod
    def apply(self, x, z):
        """
        Return y = P_z(x), where P is a preconditioner acting on x, which depends on z
        
        Parameters:
            x: The vector to precondition
            z: The vector that the preconditioner depends on
            
        Returns:
            y: The preconditioned vector
        """
        pass

# Test section
def test_precond():
    # Create a concrete implementation for testing
    class ConcretePrecond(Precond):
        def apply(self, x, z):
            t_start = time()
            y = x * z  # Simple example implementation
            self.elapsed_time = time() - t_start
            return y
    
    # Test 1: Check that we can't instantiate abstract class
    try:
        p = Precond()
        assert False, "Test 1 failed: Should not be able to instantiate abstract class"
    except TypeError:
        print("Test 1 passed: Cannot instantiate abstract class")
    
    # Test 2: Check that we can instantiate concrete implementation
    try:
        p = ConcretePrecond()
        print("Test 2 passed: Can instantiate concrete implementation")
    except TypeError:
        assert False, "Test 2 failed: Should be able to instantiate concrete implementation"
    
    # Test 3: Check that apply method works and elapsed_time is set
    p = ConcretePrecond()
    result = p.apply(2, 3)
    assert result == 6, "Test 3 failed: Incorrect result"
    assert p.elapsed_time is not None, "Test 3 failed: elapsed_time not set"
    print("Test 3 passed: apply method works and elapsed_time is set")
    
    print("\nAll tests passed!")


# Minimal SparseLU class to match MATLAB's functionality
class SparseLU:
    def __init__(self, A):
        self.lu = splu(A)
    
    def solve(self, x):
        return self.lu.solve(x)

class Precond_Fixed(Precond):
    """Fixed preconditioner class that inherits from Precond."""
    
    def __init__(self, A):
        """Initialize the fixed preconditioner with matrix A."""
        super().__init__()
        t_start = time()
        print('Init preconditioner... ', end='')
        self.P = SparseLU(A)
        self.elapsed_time = time() - t_start
        print(f'Done ({self.elapsed_time:.3g} sec)')
    
    def apply(self, x, z):
        """Apply the preconditioner to vector x (z is unused)."""
        return self.P.solve(x)


# Test section
def test_precond_fixed():
    # Test 1: Initialize with simple matrix
    print("\nTest 1: Initialization")
    A = csc_matrix([[1, 0], [0, 1]])  # Use identity matrix for simpler testing
    precond = Precond_Fixed(A)
    assert hasattr(precond, 'P'), "Test 1 failed: P not initialized"
    assert hasattr(precond, 'elapsed_time'), "Test 1 failed: elapsed_time not initialized"
    assert precond.elapsed_time is not None, "Test 1 failed: elapsed_time is None"
    print("Test 1 passed: Preconditioner initialized successfully")
    
    # Test 2: Apply preconditioner to identity matrix
    print("\nTest 2: Apply preconditioner to identity matrix")
    x = np.array([1., 2.])
    z = None  # z is unused in this implementation
    y = precond.apply(x, z)
    # For identity matrix, input should equal output
    assert np.allclose(y, x), "Test 2 failed: Incorrect preconditioner result"
    print("Test 2 passed: Preconditioner applied correctly")
    
    # Test 3: Test with diagonal matrix
    print("\nTest 3: Test with diagonal matrix")
    A = csc_matrix([[2, 0], [0, 2]])  # diagonal matrix
    precond = Precond_Fixed(A)
    x = np.array([2., 4.])
    y = precond.apply(x, z)
    expected = np.array([1., 2.])  # Solution should be x/2
    assert np.allclose(y, expected), "Test 3 failed: Incorrect preconditioner result"
    print("Test 3 passed: Preconditioner works with diagonal matrix")
    
    # Test 4: Check inheritance
    print("\nTest 4: Check inheritance")
    assert isinstance(precond, Precond), "Test 4 failed: Not instance of Precond"
    print("Test 4 passed: Proper inheritance from Precond")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_precond_fixed()
