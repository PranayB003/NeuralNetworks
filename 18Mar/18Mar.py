import numpy as np

# Conjugate Gradient Descent
class Conjugate:
    def __init__(self, A, d, c):
        self.A = A
        self.d = d
        self.c = c

    def func(self, x):
        return 0.5*(x.T @ self.A @ x) + (self.d.T @ x) + self.c

    def grad(self, x):
        return (self.A @ x) + self.d

    def alpha(self, x, p):
        g = self.grad(x)
        return -(g.T @ p)/(p.T @ self.A @ p)

    def beta(self, x, x_prev):
        g = self.grad(x)
        g_prev = self.grad(x_prev)
        return (g.T @ g)/(g_prev.T @ g_prev)

    def main(self):
        N = self.A.shape[0]
        x_prev = np.random.uniform(-0.1, 0.1, [N, 1])

        g_prev = self.grad(x_prev)
        p_prev = -g_prev
        a_prev = self.alpha(x_prev, p_prev)
        x = x_prev + (a_prev * p_prev)

        for i in range(N-1):
            g = self.grad(x)
            p = -g + (self.beta(x, x_prev) * p_prev)
            a = self.alpha(x, p)
            x = x + (a * p)

        return x


if __name__ == "__main__":
    n = 7
#    A = np.diag(list(range(1, n+1))) # A is +ve definite matrix
#    d = np.array([np.random.uniform(-5, 5, [1, n])]).T
    A = np.diag(np.ones(n)); print(A)
    d = np.array([list(range(1, n+1))]).T; print(d)
    c = 69

    model = Conjugate(A, d, c)
    minima_x = model.main()
    minima_value = model.func(minima_x)
    print(minima_x)
    print(minima_value)
