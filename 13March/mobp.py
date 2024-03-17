import numpy as np
import matplotlib.pyplot as plt

class Backpropagation_MOBP:
    def __init__(self, R = 1, S1 = 3, S2 = 1, rate = 0.1, momentum = 0.2):
        self.rate = rate
        self.momentum = momentum
        # First layer weights and biases
        self.w1 = np.random.uniform(-0.1, 0.1, [S1, R])
        self.b1 = np.random.uniform(-0.1, 0.1, [S1, 1])

        # Second lauer weights and biases
        self.w2 = np.random.uniform(-0.1, 0.1, [S2, S1])
        self.b2 = np.random.uniform(-0.1, 0.1, [S2, 1])

        self.error = []

    def forward(self, input):
        layer1_out = self.logsig(self.w1 @ input + self.b1)
        layer2_out = self.w2 @ layer1_out + self.b2
        return layer1_out, layer2_out

    def sensitivity(self, input, target):
        out1, out2 = self.forward(input)

        sens2 = -2 * (target - out2) # purelin transfer function for layer2
        sens1 = self.logsig_derivative(out1) @ self.w2.T @ sens2

        return out1, out2, sens1, sens2

    def backward(self, input, target):
        out1, out2, sens1, sens2 = self.sensitivity(input, target)

        # Layer2
        self.w2 = self.w2 - self.rate * (sens2 @ out1.T)
        self.b2 = self.b2 - self.rate * sens2

        # Layer1
        self.w1 = self.w1 - self.rate * (sens1 @ input.T)
        self.b1 = self.b1 - self.rate * sens1

    # Operating in batch mode (updating using average sensitivity matrix)
    def train(self, inputs, targets):
        w2_update = 0
        b2_update = 0
        w1_update = 0
        b1_update = 0

        # Error tracking
        error = 0

        q = len(inputs)
        for i in range(0, q):
            a1, a2, s1, s2 = self.sensitivity(inputs[i], targets[i])
            w2_update = w2_update + (s2 @ a1.T)
            b2_update = b2_update + (s2)
            w1_update = w1_update + (s1 @ inputs[i].T)
            b1_update = b1_update + (s1)
            error = error + (targets[i] - a2).T @ (targets[i] - a2)

        self.error.append(error[0, 0]) # error[0,0] gives the numberical error (not in vector form)
        return w1_update/q, b1_update/q, w2_update/q, b2_update/q
        
    def fit(self, inputs, targets, iter = 1000): # Will be running the training function multiple times
        w2_upd_prev = 0
        b2_upd_prev = 0
        w1_upd_prev = 0
        b1_upd_prev = 0

        for i in range(0, iter):
            w1_upd, b1_upd, w2_upd, b2_upd = self.train(inputs, targets)
            # Calculate current updates
            w2_upd_cur = (self.momentum * w2_upd_prev + (1-self.momentum) * w2_upd)
            b2_upd_cur = (self.momentum * b2_upd_prev + (1-self.momentum) * b2_upd)
            w1_upd_cur = (self.momentum * w1_upd_prev + (1-self.momentum) * w1_upd)
            b1_upd_cur = (self.momentum * b1_upd_prev + (1-self.momentum) * b1_upd)
            # Do the actual update
            self.w2 = self.w2 - self.rate * w2_upd_cur 
            self.b2 = self.b2 - self.rate * b2_upd_cur 
            self.w1 = self.w1 - self.rate * w1_upd_cur  
            self.b1 = self.b1 - self.rate * b1_upd_cur  
            # Update the update_variables
            w2_upd_prev = w2_upd_cur 
            b2_upd_prev = b2_upd_cur 
            w1_upd_prev = w1_upd_cur 
            b1_upd_prev = b1_upd_cur 

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

sin_model = Backpropagation_MOBP(1, 3, 1, 0.1, 0.2)
sin_model.fit(inputs, targets, 3000)

x_new = np.linspace(-4, 8, 60)
inputs_new = [np.array([[x_i]]) for x_i in x_new]
y_predicted = [sin_model.predict(input)[0, 0] for input in inputs_new]

fig1 = plt.figure()
# Plot the original and predicted sine function
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.plot(x_new, y_predicted, "-")
# Plot error
plt.subplot(1, 2, 2)
plt.plot(sin_model.error)
plt.show()

