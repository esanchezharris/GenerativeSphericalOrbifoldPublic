import numpy as np
from scipy.linalg import qr
from scipy.sparse import eye
import subprocess
from scipy.io import loadmat

# Python implementation of AffineSpace
class AffineSpace:
    def __init__(self, A, b, tol=1e-8):
        """
        Initialize an AffineSpace object.
        
        Parameters:
        A : numpy.ndarray
            The coefficient matrix.
        b : numpy.ndarray
            The right-hand side vector.
        tol : float, optional
            Tolerance for numerical operations, default is 1e-8.
        """
        import time
        self.tol = tol
        start_time = time.time()
        print('Computing affine space... ', end="")
        
        self.A = A
        self.b = b
        
        if A is not None and A.size > 0:
            # Calculate the null space of A using QR decomposition
            Q, _ = qr(A.T, mode='economic')
            self.N = Q[:, A.shape[0]:]
            
            # Dimension of the null space
            self.dim_N = self.N.shape[1]
            
            # Particular solution (least-squares solution for non-square A)
            self.x_p = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            # For empty or None A
            self.N = eye(A.shape[1], format='csr').toarray()
            self.dim_N = A.shape[1]
            self.x_p = np.zeros(A.shape[1])
        
        self.elapsed_time = time.time() - start_time
        print(f'Done ({self.elapsed_time:.3g} sec)')

    def project_onto(self, x):
        """
        Project vector x onto the affine space.
        
        Parameters:
        x : numpy.ndarray
            The vector to be projected.
        
        Returns:
        numpy.ndarray
            The projected vector.
        """
        return self.x_p + self.N @ (self.N.T @ (x - self.x_p))


# Test Inputs
A = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 1])
x = np.array([1, 0, -1])

# MATLAB Script
matlab_script = f"""
A = [1, 2, 3; 4, 5, 6];
b = [1; 1];
x = [1; 0; -1];

space = AffineSpace(A, b);
y = space.projectOnto(x);

save('matlab_result.mat', 'y');
"""

# Write MATLAB script to a file
with open("test_affine_space.m", "w") as file:
    file.write(matlab_script)

# Run MATLAB and save result
subprocess.run(["matlab", "-batch", "test_affine_space"], check=True)

# Load MATLAB result
matlab_result = loadmat("matlab_result.mat")["y"].flatten()

# Run Python implementation
space = AffineSpace(A, b)
python_result = space.project_onto(x)

# Compare Results
if np.allclose(matlab_result, python_result, atol=1e-8):
    print("Success! MATLAB and Python implementations produce the same results.")
else:
    print("Failure. MATLAB and Python results differ.")
    print(f"MATLAB Result: {matlab_result}")
    print(f"Python Result: {python_result}")

