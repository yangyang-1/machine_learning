import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams['figure.autolayout'] = True


def quadratic(w):
    return np.dot(w.T, w)


if __name__ == '__main__':
    # (a)
    x = np.array([1])
    print("an N=1 example=> here x=np.array([1]), and quadratic(x)=", quadratic(x))
    x = np.array([1, 2, 3, 4])
    print("an N=4 example=> here x=np.array([1]), and quadratic(x)=", quadratic(x))
    N = [x for x in range(1, 101)]
    y = []
    for i in N:
        tmp_x = 2 * np.random.rand(i) - 1
        y.append(quadratic(tmp_x))
    plt.plot(N, y)
    plt.show()
    # (b)
