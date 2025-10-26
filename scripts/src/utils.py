import numpy as np

def DLS(A, damping=0.1):
    # Create an identity matrix with dimensions matching A @ A.T
    I = np.eye((A @ A.T).shape[0])
    # Compute the DLS
    A_DLS = A.T @ np.linalg.inv(A @ A.T + damping**2 * I)
    return A_DLS

def W_DLS(A, damping, weight):
    w   = np.diag(weight)
    w_prime = np.linalg.inv(w)
    A_prime     = A @ w_prime @ A.T + (damping**2) * np.eye(A.shape[0])
    A_prime_inv = np.linalg.inv(A_prime)

    # Weighted DLS solution
    A_DLS = w_prime @ A.T @ A_prime_inv
    return A_DLS  
