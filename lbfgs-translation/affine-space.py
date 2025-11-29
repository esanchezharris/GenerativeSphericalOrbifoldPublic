import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from time import time
import warnings

class AffineSpace:
    def __init__(self, A, b):
        """Initialize AffineSpace with matrix A and vector b."""
        self.A = A
        self.b = b
        self.tol = 1e-8
        
        t_start = time()
        print('Computing affine space... ', end='')
        
        if A is not None and A.size > 0:
            # Compute QR decomposition of A transpose
            Q, R = np.linalg.qr(A.T, mode='complete')  # Changed to complete mode
            # Get nullspace from Q
            self.N = Q[:, A.shape[0]:]
            self.dim_N = self.N.shape[1]
            # Compute particular solution
            self.x_p = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            # If A is empty, use identity matrix for nullspace
            self.N = sparse.eye(A.shape[1])
            self.dim_N = A.shape[1]
            self.x_p = 0
            
        self.elapsed_time = time() - t_start
        print(f'Done ({self.elapsed_time:.3g} sec)')
    
    def projectOnto(self, x):
        """Project vector x onto the affine space."""
        return self.x_p + self.N @ (self.N.T @ (x - self.x_p))


def test_affine_space():
    # Test 1: Simple 2D case
    print("\nTest 1: Simple 2D case")
    A = np.array([[1., 0.],
                  [0., 1.]])
    b = np.array([1., 1.])
    affine = AffineSpace(A, b)
    x = np.array([2., 2.])
    projected = affine.projectOnto(x)
    assert np.allclose(projected, [1., 1.]), "Test 1 failed: Simple projection incorrect"
    print("Test 1 passed!")
    
    # Test 2: Nullspace computation
    print("\nTest 2: Nullspace computation")
    A = np.array([[1., 1., 0.],
                  [0., 1., 1.]])
    b = np.array([1., 1.])
    affine = AffineSpace(A, b)
    
    # Verify nullspace properties
    print(f"Matrix A shape: {A.shape}")
    print(f"Computed nullspace dimension: {affine.dim_N}")
    print(f"Nullspace matrix shape: {affine.N.shape}")
    
    # Verify N is actually in nullspace
    AN = A @ affine.N
    print(f"||AN|| (should be close to 0): {np.linalg.norm(AN)}")
    
    assert affine.dim_N == 1, "Test 2 failed: Incorrect nullspace dimension"
    print("Test 2 passed!")
    
    # Test 3: Empty matrix case
    print("\nTest 3: Empty matrix case")
    A = np.zeros((0, 3))
    b = np.array([])
    affine = AffineSpace(A, b)
    assert affine.dim_N == 3, "Test 3 failed: Incorrect dimension for empty matrix case"
    print("Test 3 passed!")
    
    # Test 4: Projection preservation
    print("\nTest 4: Projection preservation")
    A = np.array([[1., 0.],
                  [0., 1.]])
    b = np.array([1., 1.])
    affine = AffineSpace(A, b)
    x = np.array([2., 2.])
    projected1 = affine.projectOnto(x)
    projected2 = affine.projectOnto(projected1)
    assert np.allclose(projected1, projected2), "Test 4 failed: Projection not idempotent"
    print("Test 4 passed!")
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_affine_space()
