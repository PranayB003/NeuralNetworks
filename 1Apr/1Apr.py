import numpy as np

# Conjugate Gradient Descent
class BackpropagationConjugate:
    def __init__(self, *dims, epsilon = 0.1, tol):
        # Contains dimensions of the input and layers
        # dims[0] = R, dims[1] = S1, dims[2] = S2 and so on...
        if len(dims) < 2: # ()
            raise Exception("Network must have at least 1 layer\n\
                            dims[0] = R, dims[1] = S1, dims[2] = S2\
                            and so on...")
        self.dims = dims
        self.num_layers = len(self.dims) - 1

        # Total number of network parameters
        self.n = 0
        for i in range(1, len(self.dims)):
            self.n += self.dims[i] * (self.dims[i-1] + 1)

        # Weight bias initialisation
        self.X = np.random.uniform(-0.1, 0.1, [n, 1])
        self.params = {}
        self.unflat()

    def unflat(self):
        '''
        Converts the flattened network parameter vector X into
        layer-wise wight and bias matrices
        '''
        start_ind = 0
        for i in range(1, num_layers + 1):
            mid_index = start_index + dims[i]*dims[i-1]
            end_index = mid_index + dims[i]
            self.params[f'W{i}'] = self.X[start_index:mid_index]
            self.params[f'B{i}'] = self.X[mid_index:end_index]
            start_index = end_index

    def logsig(self, input):
        return 1 / (1 + np.exp(-input))

    def logsig_derivative(self, input):
        x = input * (1 - input)
        return np.diag(x.T[0])

    def forward(self, input):
        outs = []
        self.unflat()
        prev_out = input
        for i in range(1, self.num_layers + 1):
            # Purelin transfer function for last layer, logsig otherwise
            transfer_func = (self.logsig) if (i < self.num_layers) else (lambda x: x)
            layer_out = transfer_func(self.params[f'W{i}'] @ prev_out + self.params[f'B{i}'])
            out.append(layer_out)
        return outs

    def sensitivity(self, input, target):
        outs = self.forward(input)

        sens = []
        sens_last_layer = -2 * (target - outs[-1]) # since we have purelin for last layer
        sens.append(sens_last_layer)
        for i in range(self.num_layers - 1, 0, -1):
            sens_cur = self.logsig_derivative(outs[i]) @ self.params[f'W{i}'].T @ sens[-1]

        sens1 = self.logsig_derivative(out1) @ self.w2.T @ sens2

        return out1, out2, sens1, sens2

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
