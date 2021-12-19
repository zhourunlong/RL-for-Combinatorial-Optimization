import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(22, 25))
color_map = "viridis"

x = np.linspace(0.01, 1, 100)

X, Y = np.meshgrid(x, x)

for t in range(9):
    ax = fig.add_subplot(3, 3, t + 1, projection='3d')
    ax.set_title("i/n = 0.%d" % (t + 1))

    for i in range(6):
        z = i / 6
        tz = (X > i / 6 * Y).astype(float)
        ax.contourf(X, Y, z + 0.02 / 6 * tz, zdir='z', levels=100, cmap=color_map, norm=matplotlib.colors.Normalize(vmin=z, vmax=z + 0.02 / 6))

    ax.set_xlim3d(0, 1)
    ax.set_xlabel("x")
    ax.set_ylim3d(0, 1)
    ax.set_ylabel("y")
    ax.set_zlim3d(-0.01, 1.01 - 1 / 6)
    ax.set_zlabel("sum/B")
    ax.invert_zaxis()
    ax.view_init(-170, 60)

fig.subplots_adjust(wspace=0, hspace=0, right=0.9)
position = fig.add_axes([0.92, 0.4, 0.015, 0.2])
cb = fig.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=color_map), cax=position)

plt.savefig("test.jpg", bbox_inches="tight")
plt.close()