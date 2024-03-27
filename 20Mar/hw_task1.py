import numpy as np
from scipy.stats import ortho_group
import matplotlib.pyplot as plt

# Conjugate Gradient Descent
class Conjugate:
    def __init__(self, A, d, c, grad_lim = 0.01):
        self.A = A
        self.d = d
        self.c = c
        # Stops the algorithm if the gradient magnitude becomes too small
        self.grad_lim = grad_lim
        # Tracks function value across iterations
        self.val = []

    def func(self, x):
        return (0.5*(x.T @ self.A @ x) + (self.d.T @ x) + self.c)[0, 0]

    def grad(self, x):
        return (self.A @ x) + self.d

    def alpha(self, x, p):
        g = self.grad(x)
        return (-(g.T @ p)/(p.T @ self.A @ p))[0, 0]

    def beta(self, x, x_prev):
        g = self.grad(x)
        g_prev = self.grad(x_prev)
        return ((g.T @ g)/(g_prev.T @ g_prev))[0, 0]

    def main(self):
        N = self.A.shape[0]
        x = np.random.uniform(-0.1, 0.1, [N, 1])
        g = self.grad(x)
        p = -g

        self.val = []
        self.val.append(self.func(x))

        for i in range(1, N+1):
            a = self.alpha(x, p)
            x_new = x + (a * p)
            # Calculate search direction (p) for the next iteration
            g_new = -self.grad(x_new)
            if (g_new.T @ g_new <= self.grad_lim ** 2):
                x = x_new
                self.val.append(self.func(x))
                break # Stop if the gradient becomes too small
            p = -g_new + (self.beta(x_new, x) * p)
            x = x_new
            self.val.append(self.func(x))

        return x


if __name__ == "__main__":
    # Generate random A, d, and c
    n = np.random.randint(5, 11) # Dimension of A is in [5, 10]
    U = ortho_group.rvs(n) # Random orthogonal matrix
    D = np.diag(list(range(1, n+1)))
    A = U @ D @ U.T # "A" is a positive definite matrix
    
    d = np.random.randint(0, 101, (n, 1))
    c = np.random.randint(0, 101)

    # Display values of A, d, and c
    print("n:", n)
    print("A:\n", A)
    print("d:\n", d)
    print("c:", c)

    # Run the conjugate gradient descent algorithm
    model = Conjugate(A, d, c)
    minima_x = model.main()

    # Print the optimal "x" and plot function values across iterations
    print(minima_x)
    plt.figure()
    plt.plot(range(len(model.val)), model.val, "bo-")
    plt.show()
