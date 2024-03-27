import numpy as np
import plotly.graph_objects as go

class Golden:
    def __init__(self, f, g, epsilon = 0.01, tol = 0.01):
        self.f       = f
        self.g       = g
        self.epsilon = epsilon
        self.tol     = tol
        self.reset_tracked_points()
        
    def rough_interval(self, start):
        grad = self.g(start)
        a = start 
        b = start - (self.epsilon * grad)

        # Edge case: if we have already taken a step too far
        if self.f(a) < self.f(b):
            return a, b

        while self.f(b - (self.epsilon * grad)) <= self.f(b):
            a, b = b, b - (self.epsilon * grad)
        b = b - (self.epsilon * grad)

        return a, b
    
    def golden_search(self, start):
        eta = 0.382
        self.reset_tracked_points()

        a, b = self.rough_interval(start)
        c, d = a + (eta * (b - a)), b - (eta * (b - a))
        self.track_points(a, b, c, d)

        while np.linalg.norm(b - a) >= self.tol:
            if self.f(c) < self.f(d):
                b = d
                d = c
                c = a + (eta * (b - a))
            else:
                a = c
                c = d
                d = b - (eta * (b - a))
            self.track_points(a, b, c, d)

        return (a+b)/2

    def track_points(self, a, b, c, d):
        self.iter += 1
        # Append the current values of a, b, c, d to the tracking array
        self.a.append(a)
        self.b.append(b)
        self.c.append(c)
        self.d.append(d)
        # Append the current values of f() at a, b, c, d to the tracking array
        self.fa.append(self.f(a))
        self.fb.append(self.f(b))
        self.fc.append(self.f(c))
        self.fd.append(self.f(d))

    def reset_tracked_points(self):
        self.iter = 0
        # Reset the tracked values of a, b, c, d
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        # Reset the tracked values of f() at a, b, c, d
        self.fa = []
        self.fb = []
        self.fc = []
        self.fd = []

if __name__ == "__main__":
    f = lambda x: (x[0, 0]**2) + (x[1, 0]**2) - 1           # x^2 + y^2 - 1
    g = lambda x: np.array([[2 * x[0, 0], 2 * x[1, 0]]]).T

    start = np.array([[2, 2]]).T
    model = Golden(f, g, epsilon=0.5)
    model.golden_search(start)

    scatter_a = go.Scatter3d(x=[a[0,0] for a in model.a],
                             y=[a[1,0] for a in model.a],
                             z=model.fa,
                             name="a")
    scatter_b = go.Scatter3d(x=[b[0,0] for b in model.b],
                             y=[b[1,0] for b in model.b],
                             z=model.fb,
                             name="b")
    scatter_c = go.Scatter3d(x=[c[0,0] for c in model.c],
                             y=[c[1,0] for c in model.c],
                             z=model.fc,
                             name="c")
    scatter_d = go.Scatter3d(x=[d[0,0] for d in model.d],
                             y=[d[1,0] for d in model.d],
                             z=model.fd,
                             name="d")

    fig = go.Figure(data=[scatter_a, scatter_b])
    fig.show()
    fig = go.Figure(data=[scatter_c, scatter_d])
    fig.show()

