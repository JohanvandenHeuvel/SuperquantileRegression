"""
Reference:
Miranda, S. I. (2014).
Superquantile regression: theory, algorithms, and applications.
NAVAL POSTGRADUATE SCHOOL MONTEREY CA.
"""

import matplotlib.pyplot as plt
import numpy as np

from data import generate_synthetic_data


def basis_function(x):
    return np.array([x]).T


def linear_model(w, X):
    return X @ np.expand_dims(w, 1)


def superquantile(y, alpha):

    # check if array sorted
    assert np.all(np.diff(y) >= 0)

    alpha_index = -1
    prop_values = []
    for i in range(len(y)):
        prop_values.append(i / len(y))
        if alpha_index == -1 and prop_values[i] >= alpha:
            alpha_index = i

    if alpha == 0:
        return np.sum(prop_values * y)

    if alpha_index < len(prop_values):
        a = (prop_values[alpha_index] - alpha) * y[alpha_index]
        b = np.sum((1 / len(y) * y)[alpha_index:])
        return 1 / (1 - alpha) * (a + b)[0]

    if alpha > 1 - prop_values[-1]:
        return y[-1]


def objective_function(w, H, y, alpha):
    """
    Equation (III. 8) from paper

    :param w: weight parameters
    :param X: feature vector
    :param y: label vector
    :param alpha: quantile parameter
    :return:
    """
    n = len(H)

    Z = y - linear_model(w, H)
    # sort to make a cumulative distribution out of it
    sorted_idx = np.argsort(Z.flatten())
    H = H[sorted_idx]
    Z = Z[sorted_idx]

    Q = [evaluate_Q(Z, alpha, i) for i in range(len(Z))]
    Q = np.expand_dims(Q, 1)

    A = 1 / n * np.sum(np.multiply(Z, Q))  # E[Z_0(c) * \bar{Q}^{Z_0(c)}_\alpha]
    B = 1 / n * np.sum(Z)  # E[Z_0(c)]
    function_value = A - B

    A = -1 / n * np.sum(np.multiply(H, Q))
    B = 1 / n * np.sum(H)
    sub_gradient = A + B

    return function_value, sub_gradient


def evaluate_Q(y, alpha, i):
    """
    Equation (III. 9) from paper

    This can be seen as some measure of "weight" for the losses.
    I.e. if the loss < alpha we don't use it, and the weight increases proportional as loss >= alpha

    :param y: discrete cumulative distribution with n atoms
    :param alpha: quantile parameter
    :param i: index of atom
    :return: Q^y_alpha
    """

    Fy = i / len(y)  # cumulative probability of i: P(X <= i)
    lcFy = (
        (i - 1) / len(y) if i > 0 else 0
    )  # left-continuous point of the cumulative distribution: P(X < i)

    # Note that Fy - lcFy = P(X <= i) - P(X < i) = P(X == i)

    value = 0
    if alpha < lcFy == Fy < 1:  # lcFy == Fy iff there is a "plateau" in the CDF
        value = np.log((1 - alpha) / (1 - Fy))
    elif (
        alpha < lcFy < Fy == 1
    ):  # only happens for the last atom, first of the plateau if possible
        value = np.log((1 - alpha) / (1 - lcFy)) + 1
    elif alpha < lcFy < Fy:  # happens often if alpha is low
        value = (
            np.log((1 - alpha) / (1 - lcFy))
            + 1
            + (1 - Fy) / (Fy - lcFy) * np.log((1 - Fy) / (1 - lcFy))
        )
    elif lcFy < alpha < Fy == 1:  # alpha lies within probability of two atoms
        value = (Fy - alpha) / (Fy - lcFy)
    elif lcFy <= alpha <= Fy and lcFy < Fy:
        value = (Fy - alpha) / (Fy - lcFy) + (1 - Fy) / (Fy - lcFy) * np.log(
            (1 - Fy) / (1 - alpha)
        )

    return 1 / (1 - alpha) * value


def subgradient_method(X, y, alpha):

    # STEP 0.  Initialize
    H = basis_function(X)
    w = [np.random.random(H.shape[1])]
    w_0 = [0]
    k = 1

    f_best = [999.0]
    while True:

        # STEP 1.  Compute function value and sub gradient
        function_value, sub_gradient = objective_function(w[k - 1], H, y, alpha)

        # print(k, sub_gradient, w[k - 1])
        if sub_gradient == 0:
            # w_k is an optimal solution
            break

        # STEP 2.  Set values
        f_best.append(np.minimum(f_best[k - 1], function_value))

        # STEP 3.  Choose stepsize
        delta = 0.01

        # STEP 4.  Update and go to step 1
        w.append(w[k - 1] - delta * sub_gradient)

        pred = linear_model(w[k], H)
        Z = y - pred
        sorted_idx = np.argsort(Z.flatten())
        Z = Z[sorted_idx]

        w_0.append(superquantile(Z, alpha))

        k += 1

        if k > 1000:
            break

    # return the weights with the lowest function value
    return w[np.argmin(f_best)], w_0[np.argmin(f_best)]


def main():
    x, y = generate_synthetic_data("linear", w=[1], b=[1], noise_strength=1, num_points=1000)

    alphas = np.round(np.arange(0.1, 1, 0.1), decimals=2)

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(alphas)))
    fig_1, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig_2, axs = plt.subplots(len(alphas), 1, figsize=(10, 50))
    ax.plot(x, y, "o")
    for i, alpha in enumerate(alphas):

        w, w_0 = subgradient_method(x, np.expand_dims(y, 1), alpha)

        pred = linear_model(w, basis_function(x)) + w_0
        Z = np.expand_dims(y, 1) - pred

        ax.plot(
            x,
            pred,
            color=colors[i],
        )
        axs[i].hist(Z, bins=100)

    ax.legend(["data"] + list(alphas))
    ax.set(xlabel="x", ylabel="y")
    plt.show()


if __name__ == "__main__":
    main()
