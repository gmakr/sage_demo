import numpy as np
import torch

def benchmark_fun(x):
    """
    Hartmann 4D function.

    Args:
    x : np.ndarray
        4-D input domain. Each element must be in [0, 1].

    Returns:
    float : function value
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10.0, 3.0, 17.0, 3.5],
        [0.05, 10.0, 17.0, 0.1],
        [3.0, 3.5, 1.7, 10.0],
        [17.0, 8.0, 0.05, 10.0]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124],
        [2329, 4135, 8307, 3736],
        [2348, 1451, 3522, 2883],
        [4047, 8828, 8732, 5743]
    ])

    output = 0
    for i in range(4):
        inner_sum = np.sum(A[i, :] * (x - P[i, :])**2)
        output += alpha[i] * np.exp(-inner_sum)

    return output
