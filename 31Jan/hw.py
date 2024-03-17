import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC # "Support vector classifier"

if __name__ == "__main__":
    num_points  = int(input("Enter the total number of points: "))
    # Generate random points for 2 classes
    inputs, targets = make_blobs(n_samples=num_points, centers=2, cluster_std=0.60)
    plt.scatter(inputs[:, 0], inputs[:, 1], c=targets, s=50, cmap="winter")

    # Get weights and biases for optimal hyperplane
    model = SVC(kernel='linear')
    model.fit(inputs, targets)

    # Get current axis limits
    ax = plt.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    
    # Get the weights, biases of the optimal separating hyperplane and the support vectors
    w = model.coef_
    b = model.intercept_
    vec = model.support_vectors_

    # Lambda function for equation of line
    get_y = lambda x, w, b: -((x * w[0, 0]) + b[0]) / w[0,1]
    # Create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = get_y(x, w, b)
    # Plot separating boundary
    plt.plot(x, y)

    # Plot boundaries passing through support vectors
    b_prime = -(vec @ w.T)
    y_prime = [get_y(x, w, b) for b in b_prime]
    for y_p in y_prime:
        plt.plot(x, y_p, "--")

    plt.show()
