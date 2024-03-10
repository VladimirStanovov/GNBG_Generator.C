import os
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-100, 100, 101)
y = np.linspace(-100, 100, 101)
X, Y = np.meshgrid(x, y)
Z = np.loadtxt("vis.txt")
fig = plt.figure(figsize=(10,10))
plt.contour(X,Y,Z,50,cmap="jet")
fig.savefig("vis.png")
plt.show()
