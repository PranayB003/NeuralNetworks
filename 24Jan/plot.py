import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self) -> None:
        self.w = np.random.uniform(-0.1, 0.1, [1, 2])
        self.b = np.random.uniform(-0.1, 0.1, [1, 1])

    def update(self, p: list[int], target: int) -> None:
        n = (self.w @ p) + self.b
        a = self.hardlim(n)
        e = target - a

        print("e: ", e)
        print("w: ", self.w)
        self.w += e @ p.T
        print("w: ", self.w)
        self.b += e

    def hardlim(self, p):
        return (p > 0) * 1

if __name__ == "__main__":
    # Category 1
    p1 = np.array([[0.9, 0.5]]).T
    p2 = np.array([[0.7, 1.9]]).T
    cat1    = [p1, p2]
    cat1_x  = [p[0, 0] for p in cat1]
    cat1_y  = [p[1, 0] for p in cat1]

    # Category 2
    p3 = np.array([[-0.5, -0.5]]).T
    p4 = np.array([[-0.7, 0.8]]).T
    p5 = np.array([[-1.5, -1.1]]).T
    cat2    = [p3, p4, p5]
    cat2_x  = [p[0, 0] for p in cat2]
    cat2_y  = [p[1, 0] for p in cat2]

    # Create Model
    model = Perceptron()
    model.update(p1, 1)
    #model.update(p2, 1)
    #model.update(p3, 0)
    #model.update(p4, 0)
    #model.update(p5, 0)

    # Plot points and model
    plt.scatter(cat1_x, cat1_y, label="Category 1")
    plt.scatter(cat2_x, cat2_y, label="Category 2")

    x = np.linspace(-2, 2, 5)
    y = np.reshape(-((model.w[0, 0] * x) + model.b) / model.w[0, 1], len(x))
    plt.plot(x, y)

    plt.legend(loc="upper left")
    plt.show()
