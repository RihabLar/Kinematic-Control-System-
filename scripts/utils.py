import numpy as np

def DLS(A, damping=0.01):
    # Create an identity matrix with dimensions matching A @ A.T
    I = np.eye((A @ A.T).shape[0])
    # Compute the DLS
    A_DLS = A.T @ np.linalg.inv(A @ A.T + damping**2 * I)
    return A_DLS
