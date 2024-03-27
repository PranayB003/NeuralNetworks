from scipy.stats import ortho_group
import numpy as np

n = 4
U = ortho_group.rvs(n) # Random orthogonal matrix
print("U:\n", U)
print("Eigenvalues of U:\n", np.linalg.eigvals(U))
print("U @ U.T:\n", U @ U.T) # Confirm that U is an orthogonal matrix

# Since all diagonal elements are positive, its a +ve definite matrix
D = np.diag(list(range(5, n+5)))
M = U @ D @ U.T
print("Eigenvalues of M:\n", np.linalg.eigvals(M)) # Check that M is also a positive definite matrix
