import numpy as np
import subprocess
from scipy.io import loadmat

# Python implementation of colStack
def col_stack(x):
    """
    Stack the elements of the input array x into a single column vector.
    
    Parameters:
    x : numpy.ndarray
        Input matrix or vector.
    
    Returns:
    numpy.ndarray
        Column vector with the elements of x.
    """
    return x.flatten('F')  # Flatten in column-major (Fortran-like) order

# Test Input
x = np.array([[1, 2, 3], [4, 5, 6]])

# MATLAB Script
matlab_script = f"""
x = [1, 2, 3; 4, 5, 6];
y = colStack(x);
save('matlab_result.mat', 'y');
"""

# Write MATLAB script to a file
with open("test_col_stack.m", "w") as file:
    file.write(matlab_script)

# MATLAB colStack function
matlab_function = """
function y = colStack(x)
y = x(:);
end
"""

# Save MATLAB function
with open("colStack.m", "w") as file:
    file.write(matlab_function)

# Run MATLAB and save result
subprocess.run(["matlab", "-batch", "test_col_stack"], check=True)

# Load MATLAB result
matlab_result = loadmat("matlab_result.mat")["y"].flatten()

# Run Python implementation
python_result = col_stack(x)

# Compare Results
if np.allclose(matlab_result, python_result, atol=1e-8):
    print("Success! MATLAB and Python implementations produce the same results.")
else:
    print("Failure. MATLAB and Python results differ.")
    print(f"MATLAB Result: {matlab_result}")
    print(f"Python Result: {python_result}")

