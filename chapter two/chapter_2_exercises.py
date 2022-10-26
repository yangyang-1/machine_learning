import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams['figure.autolayout'] = True


def function_f(w):
    return np.tanh(4 * w[0] + 4 * w[1]) + np.max(0.4 * w[0] ** 2, 1) + 1


def function_g(w):
    return np.sin(3 * w) + 0.3 * np.square(w)


def function_m(w):
    return 100 * (w[1] - w[0] ** 2) ** 2 + (w[0] - 1) ** 2


def function_n(w):
    return (1 - w[0] / 2 + w[0] ** 5 + w[1] ** 3) * np.exp(-w[0] ** 2 - w[1] ** 2)


def function_F(w):
    return np.square(w[0]) + np.square(w[1]) + 2


def function_G(w):
    return 0.26 * (np.square(w[0]) + np.square(w[1])) - 0.48 * w[0] * w[1]


def visualization_contour_map(g, x_start, x_end, y_start, y_end, weight_history, cost_history):
    point_x = np.linspace(x_start, x_end, 100).reshape(1, 100)
    point_y = np.linspace(y_start, y_end, 100).reshape(1, 100)
    x, y = np.meshgrid(point_x, point_y)
    w_data = np.concatenate([[x], [y]], axis=0)
    fig, ax = plt.subplots()
    contour = ax.contour(x, y, g(w_data), levels=[0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 4, 6, 8, 9], cmap='viridis',
                         linewidth=1)
    # ax.clabel(contour, inline=True, fontsize=10)
    ax.axhline(y=3, color='0.7', linestyle='--', alpha=0.3)
    ax.axvline(x=3, color='0.7', linestyle='--', alpha=0.3)
    ax.scatter(x=[x[0][0] for x in weight_history], y=[y[1][0] for y in weight_history],cmap='hsv', marker='o')
    size = len(weight_history)
    # for i in range(0, size - 1):
    #     ax.arrow(weight_history[i][0][0], weight_history[i][1][0],
    #              weight_history[i + 1][0][0] - weight_history[i][0][0],
    #              weight_history[i + 1][1][0] - weight_history[i][1][0],
    #              length_includes_head=True, head_width=0.1)

    ax.set_xlabel('w1')
    ax.set_ylabel('w2')

    # ax.set_xticks(np.arange(-4, 4, 0.5))
    # ax.set_yticks(np.arange(-7, 20, 1))
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-15, 15)
    # contour = plt.contour(x, y, g(w_data), colors='black', linewidth=0.5)
    # plt.axhline(y=0, color='0.7', linestyle='--')
    # plt.axvline(x=0, color='0.7', linestyle='--')
    # plt.clabel(contour, inline=True, fontsize=10)
    # plt.xlabel('w0')
    # plt.ylabel('w1')
    plt.show()


def visualization_analyse(weight_history, cost_history):
    x = np.arange(-5, 5, 0.1)
    y = function_g(x)
    color = np.arange(len(weight_history) + 1, 1, -1)
    plt.axhline(y=0, color='0.7', linestyle='--')
    plt.axvline(x=0, color='0.7', linestyle='--')
    plt.plot(x, y, color='black')
    plt.scatter(x=weight_history, y=cost_history, c=color, cmap='hsv', marker='X')
    plt.scatter(x=weight_history, y=np.zeros(len(weight_history)), c=color, cmap='hsv', marker='o')
    plt.xlabel("w")
    plt.ylabel("g(w)")
    plt.show()


def random_search(g, alpha, max_its, w, num_samples):
    weight_history = []
    cost_history = []
    for k in range(1, max_its + 1):
        weight_history.append(w)
        cost_history.append(g(w))
        # alpha = 1 / k
        while True:
            direction_candidates = (2 * np.random.rand(w.shape[0], num_samples) - 1)
            w_candidates = w + alpha * (direction_candidates / np.linalg.norm(direction_candidates, axis=0))
            best_direction = np.argmin(g(w_candidates))
            down_direction = w_candidates[:, best_direction][:, np.newaxis]
            if g(w) > g(down_direction):
                w = down_direction
                break
    weight_history.append(w)
    cost_history.append(g(w))
    return weight_history, cost_history


def coordinate_search(g, alpha, max_its, w):
    size = np.size(w)
    direction_plus = np.eye(size, size)
    direction_minus = -np.eye(size, size)
    directions = np.concatenate((direction_plus, direction_minus), axis=1)

    weight_history = []
    cost_history = []
    for k in range(1, max_its + 1):
        alpha = 1 / k
        weight_history.append(w)
        cost_history.append(g(w))
        candidate_points = w + alpha * directions
        evaluate_value = g(w) - g(candidate_points)
        best_direction_index = np.argmax(evaluate_value)
        if evaluate_value[best_direction_index] <= 0:
            return weight_history, cost_history
        w = candidate_points[:, best_direction_index][:, np.newaxis]
    weight_history.append(w)
    cost_history.append(g(w))
    return weight_history, cost_history


def coordinate_descend(g, alpha, max_its, w):
    size = np.size(w)
    direction_plus = np.eye(size, size)
    direction_minus = -np.eye(size, size)
    directions = np.concatenate((direction_plus, direction_minus), axis=1)

    weight_history = []
    cost_history = []

    for k in range(1, max_its + 1):
        alpha = 1 / k
        weight_history.append(w)
        cost_history.append(g(w))
        end_flag = True
        i = 0
        while i < size:
            candidate_direction = directions[:, [i, i + size]]
            w_candidate = w + alpha * candidate_direction
            evaluate_value = g(w_candidate) - g(w)
            w_index = np.argmin(evaluate_value)
            if evaluate_value[w_index] < 0:
                w = w_candidate[:, w_index][:,np.newaxis]
                end_flag = False
                break
            i = i + 1
        if end_flag:
            return weight_history, cost_history
    weight_history.append(w)
    cost_history.append(g(w))
    return weight_history, cost_history


if __name__ == '__main__':
    w0 = np.array([[3.2], [3.8]])
    # w_history, c_history = coordinate_search(function_F, 1, 7, w0)
    w_history, c_history = coordinate_search(function_G, 1, 1000, w0)
    # w_history, c_history = coordinate_descend(function_G, 1, 40, w0)
    visualization_contour_map(function_G, -1.5, 4.5, -1.5, 4.5, w_history, c_history)
