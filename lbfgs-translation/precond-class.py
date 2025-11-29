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

if __name__ == "__main__":
    test_precond()
