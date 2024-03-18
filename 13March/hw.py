import numpy as np
import matplotlib.pyplot as plt

class Backpropagation_Momentum:
    def __init__(self, R = 1, S1 = 3, S2 = 1, rate = 0.1, momentum = 0.2, batch = False):
        self.rate = rate
        self.momentum = momentum
        self.batch = batch

        # First layer weights and biases
        self.w1 = np.random.uniform(-0.1, 0.1, [S1, R])
        self.b1 = np.random.uniform(-0.1, 0.1, [S1, 1])

        # Second lauer weights and biases
        self.w2 = np.random.uniform(-0.1, 0.1, [S2, S1])
        self.b2 = np.random.uniform(-0.1, 0.1, [S2, 1])

        # Storing updates for momentum
        self.deltaw1 = np.random.uniform(-0.1, 0.1, [S1, R])
        self.deltab1 = np.random.uniform(-0.1, 0.1, [S1, 1])
        self.deltaw2 = np.random.uniform(-0.1, 0.1, [S2, S1])
        self.deltab2 = np.random.uniform(-0.1, 0.1, [S2, 1])

        # Error Tracking
        self.error = []

    def forward(self, input):
        layer1_out = self.tansig(self.w1 @ input + self.b1)
        layer2_out = self.w2 @ layer1_out + self.b2
        return layer1_out, layer2_out

    def sensitivity(self, input, target):
        out1, out2 = self.forward(input)

        sens2 = -2 * (target - out2) # purelin transfer function for layer2
        sens1 = self.tansig_derivative(out1) @ self.w2.T @ sens2

        return out1, out2, sens1, sens2

    # Operating in batch mode (updating using average sensitivity matrix)
    def train(self, inputs, targets):
        w2_update = 0
        b2_update = 0
        w1_update = 0
        b1_update = 0

        # Error tracking
        error = np.array([[0]])

        q = len(inputs)
        for i in range(0, q):
            a1, a2, s1, s2 = self.sensitivity(inputs[i], targets[i])
            # Overall error for the current iteration
            error = error + (targets[i] - a2).T @ (targets[i] - a2)

            if (self.batch):
                w2_update = w2_update + (s2 @ a1.T)
                b2_update = b2_update + (s2)
                w1_update = w1_update + (s1 @ inputs[i].T)
                b1_update = b1_update + (s1)
            else:
                self.w2 = self.w2 - self.rate * (s2 @ a1.T)
                self.b2 = self.b2 - self.rate * s2
                self.w1 = self.w1 - self.rate * (s1 @ inputs[i].T)
                self.b1 = self.b1 - self.rate * s1

        self.error.append(error[0, 0]) # error[0,0] gives the numerical error (not in vector form)
        return w1_update/q, b1_update/q, w2_update/q, b2_update/q
        
    def fit(self, inputs, targets, iter = 1000): # Will be running the training function multiple times
        for i in range(0, iter):
            w1_upd, b1_upd, w2_upd, b2_upd = self.train(inputs, targets)
            if (self.batch):
                # Calculate current updates
                self.deltaw1 = (self.momentum * self.deltaw1 - (1-self.momentum) * self.rate * w1_upd)
                self.deltab1 = (self.momentum * self.deltab1 - (1-self.momentum) * self.rate * b1_upd)
                self.deltaw2 = (self.momentum * self.deltaw2 - (1-self.momentum) * self.rate * w2_upd)
                self.deltab2 = (self.momentum * self.deltab2 - (1-self.momentum) * self.rate * b2_upd)
                # Do the actual update
                self.w2 = self.w2 + self.deltaw2 
                self.b2 = self.b2 + self.deltab2 
                self.w1 = self.w1 + self.deltaw1  
                self.b1 = self.b1 + self.deltab1  

    def predict(self, input):
        out1, out2 = self.forward(input)
        return out2

    def tansig(self, input):
        return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))

    def tansig_derivative(self, input):
        x = 1 - input**2
        return np.diag(x.T[0])

logsig = lambda x: 1 / (1 + np.exp(-x))
x = np.linspace(-10, 10, 60)
y = logsig(x)

inputs = [np.array([[x_i]]) for x_i in x]
targets = [np.array([[y_i]]) for y_i in y]

model          = Backpropagation_Momentum(1, 3, 1, 0.1, 0.2)
model_batching = Backpropagation_Momentum(1, 3, 1, 0.1, 0.2, True)
model.fit(inputs, targets, 50)
model_batching.fit(inputs, targets, 50)

y_predicted = [model.predict(input)[0, 0] for input in inputs]
y_predicted2 = [model_batching.predict(input)[0, 0] for input in inputs]

# No Batching
# Plot the original function
fig1 = plt.figure()
plt.subplot(2, 2, 1)
plt.plot(x, y)
plt.title("Original Function")
# Plot the predicted function
plt.subplot(2, 2, 2)
plt.plot(x, y_predicted, "r")
plt.title("Predicted Function")
# Plot error tracking curve
plt.subplot(2, 2, (3, 4))
plt.plot(model.error, "m")
plt.title("Error Tracking Curve")
plt.subplots_adjust(hspace=0.5)
plt.suptitle("No Batching")
fig1.show()

# With Batching
# Plot the original function
fig2 = plt.figure()
plt.subplot(2, 2, 1)
plt.plot(x, y)
plt.title("Original Function")
# Plot the predicted function
plt.subplot(2, 2, 2)
plt.plot(x, y_predicted2, "r")
plt.title("Predicted Function")
# Plot error tracking curve
plt.subplot(2, 2, (3, 4))
plt.plot(model_batching.error, "m")
plt.title("Error Tracking Curve")
plt.subplots_adjust(hspace=0.5)
plt.suptitle("With Batching")
fig2.show()
plt.show()
