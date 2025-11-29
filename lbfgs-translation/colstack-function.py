import numpy as np

def colStack(x):
    """
    Convert input array x into a column vector by stacking all elements.
    Equivalent to MATLAB's x(:) operation.
    """
    return np.asarray(x).flatten('F')[:, np.newaxis]

# Test section
def test_colStack():
    # Test 1: 2D matrix
    x1 = np.array([[1, 2], 
                   [3, 4]])
    y1 = colStack(x1)
    assert np.array_equal(y1, np.array([[1], [3], [2], [4]])), "Test 1 failed"
    
    # Test 2: 1D array
    x2 = np.array([1, 2, 3])
    y2 = colStack(x2)
    assert np.array_equal(y2, np.array([[1], [2], [3]])), "Test 2 failed"
    
    # Test 3: 3D array
    x3 = np.array([[[1, 2], [3, 4]], 
                   [[5, 6], [7, 8]]])
    y3 = colStack(x3)
    assert np.array_equal(y3, np.array([[1], [5], [3], [7], [2], [6], [4], [8]])), "Test 3 failed"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_colStack()
