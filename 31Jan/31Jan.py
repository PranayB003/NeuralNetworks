import numpy as np

n = 10
# Points for category 1
points_x1 = np.random.uniform(2, 1, n)
points_y1 = np.random.uniform(2, 1, n)
points_z1 = np.random.uniform(2, 1, n)

# Points for category 2
points_x2 = np.random.uniform(-2, 1, n)
points_y2 = np.random.uniform(-2, 1, n)
points_z2 = np.random.uniform(-2, 1, n)

import plotly.graph_objects as go

# Do a scatter plot of category 1
Scatter1 = go.Scatter3d(
        x=points_x1,
        y=points_y1,
        z=points_z1,
        mode='markers',
        name='Category 1')

Scatter2 = go.Scatter3d(
        x=points_x2,
        y=points_y2,
        z=points_z2,
        mode='markers',
        name='Category 2')

# Support Vector Machine Implementation
from sklearn import svm

model = svm.SVC(kernel='linear') # This initialises the class SVC

# Create the inputs
cat1 = np.array(list(zip(points_x1, points_y1, points_z1)))
cat2 = np.array(list(zip(points_x2, points_y2, points_z2)))
inputs = np.vstack([cat1, cat2])
targets = [1]*n + [-1]*n

model.fit(inputs, targets)
w = model.coef_
b = model.intercept_
vec = model.support_vectors_

# Equation of separating plane as an inline function
get_z = lambda w, b, x, y: -w[0,0]*x/w[0,2] -w[0,1]*y/w[0,2] - b[0]/w[0,2]

# Plot the separating plane
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)
plane_z = get_z(w, b, X, Y)

Surface = go.Surface(
        x=X,
        y=Y,
        z=plane_z,
        opacity=0.5,
        showscale=False)

# Surfaces passing through the supoprt vectors
modified_b = -(vec @ w.T)
support_plane_z = [get_z(w, mod_b, X, Y) for mod_b in modified_b]
SupportSurfaces = [go.Surface(
                x=X,
                y=Y,
                z=sup_plane_z,
                opacity=0.5,
                showscale=False) for sup_plane_z in support_plane_z]

fig = go.Figure(data=[Scatter1, Scatter2, Surface, *SupportSurfaces])
fig.show()
