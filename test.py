import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(22, 25))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    lz = 6
    color_map = "viridis"

    for t in range(9):
        ax = fig.add_subplot(3, 3, t + 1, projection='3d')
        ax.set_title("i/n = 0.%d" % (t + 1))

        x = np.linspace(0.01, 1, 100)
        X, Y = np.meshgrid(x, x)

        levels = np.linspace(-1, 1, 40)

        for i in range(lz):
            z = i / lz
            Z = (X ** (i / lz)) * np.cos(Y)
            ax.contourf(X, Y, z + 0.02 / lz * Z, zdir='z', levels=100, cmap=color_map, norm=matplotlib.colors.Normalize(vmin=z, vmax=z + 0.02 / lz))

        ax.set_xlim3d(0, 1)
        ax.set_xlabel("v")
        ax.set_ylim3d(0, 1)
        ax.set_ylabel("s")
        ax.set_zlim3d(1 / lz - 0.01, 1.01)
        ax.set_zlabel("sum/B")
        ax.invert_zaxis()
        ax.view_init(-170, 60)
        

    fig.subplots_adjust(wspace=0, hspace=0, right=0.9)
    position = fig.add_axes([0.92, 0.4, 0.015, 0.2])
    cb = fig.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=color_map), cax=position)

    plt.savefig("test.jpg", bbox_inches="tight")
    plt.close()
