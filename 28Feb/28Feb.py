import numpy as np
import matplotlib.pyplot as plt

class Backpropagation:
    def __init__(self, R = 1, S1 = 3, S2 = 1, rate = 0.1):
        self.rate = rate
        # First layer weights and biases
        self.w1 = np.random.uniform(-0.1, 0.1, [S1, R])
        self.b1 = np.random.uniform(-0.1, 0.1, [S1, 1])

        # Second lauer weights and biases
        self.w2 = np.random.uniform(-0.1, 0.1, [S2, S1])
        self.b2 = np.random.uniform(-0.1, 0.1, [S2, 1])

    def forward(self, input):
        layer1_out = self.logsig(self.w1 @ input + self.b1)
        layer2_out = self.w2 @ layer1_out + self.b2
        return layer1_out, layer2_out

    def sensitivity(self, input, target):
        out1, out2 = self.forward(input)

        sens2 = -2 * (target - out2) # purelin transfer function for layer2
        sens1 = self.logsig_derivative(out1) @ self.w2.T @ sens2

        return out1, out2, sens1, sens2

    def backward(self, data, target):
        out1, out2, sens1, sens2 = self.sensitivity(data, target)

        # Layer2
        self.w2 = self.w2 - self.rate * (sens2 @ out1.T)
        self.b2 = self.b2 - self.rate * sens2

        # Layer1
        self.w1 = self.w1 - self.rate * (sens1 @ data.T)
        self.b1 = self.b1 - self.rate * sens1

    def train(self, inputs, targets):
        q = len(inputs)
        for i in range(0, q):
            self.backward(inputs[i], targets[i])

    def fit(self, inputs, targets, iter = 1000): # Will be running the training function multiple times
        for i in range(0, iter):
            self.train(inputs, targets)

    def predict(self, input):
        out1, out2 = self.forward(input)
        return out2

    def logsig(self, input):
        return 1 / (1 + np.exp(-input))

    def logsig_derivative(self, input):
        x = input * (1 - input)
        return np.diag(x.T[0])

x = np.linspace(-4, 4, 30)
y = np.sin(x)

inputs = [np.array([[x_i]]) for x_i in x]
targets = [np.array([[y_i]]) for y_i in y]

sin_model = Backpropagation(1, 3, 1, 0.1)
sin_model.fit(inputs, targets, 3000)

x_new = np.linspace(-4, 8, 60)
inputs_new = [np.array([[x_i]]) for x_i in x_new]
y_predicted = [sin_model.predict(input)[0, 0] for input in inputs_new]

plt.plot(x, y)
plt.plot(x_new, y_predicted, "-")
plt.show()
