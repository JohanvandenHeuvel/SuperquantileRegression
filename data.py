import numpy as np
import matplotlib.pyplot as plt


def linear_data(w, b, noise_strength=1, num_points=100):
    """
    Line with Gaussian noise for the target
    """
    w = np.array(w)
    b = np.array(b)

    def f(x):
        return x @ w.T + b

    def basis_function(x):
        return np.array([x]).T

    x = np.linspace(0, 10, num_points)
    h = basis_function(x)

    y = f(h)
    noise = np.random.normal(size=y.shape, scale=noise_strength)
    y += noise

    return x, y


def parabolic_data(w, b, noise_strength=1, num_points=100):
    """
    half parabola with Gaussian noise for the target
    """
    w = np.array(w)
    b = np.array(b)

    def f(x):
        return x @ w.T + b

    def basis_function(x):
        return np.array([x, x**2]).T

    x = np.linspace(0, 10, num_points)
    h = basis_function(x)

    y = f(h)
    noise = np.random.normal(size=y.shape, scale=noise_strength)
    y += noise

    return x, y


data_func_dict = {"linear": linear_data, "parabolic": parabolic_data}


def generate_synthetic_data(data_name, **kwargs):
    x, y = data_func_dict[data_name](**kwargs)

    return x, y


if __name__ == "__main__":
    x, y = linear_data()
    # x, y = parabolic_data()

    plt.plot(x, y, "o")
    plt.show()
