import numpy as np


def data_1():
    start = [0, 0]
    stop = [10, 10]

    signal = np.linspace(start, stop)
    noise = np.random.normal(size=len(signal))

    x = signal[:, 0]
    y = signal[:, 1] + noise

    return x, y


data_func_dict = {"1": data_1}


def generate_synthetic_data(data_name):
    x, y = data_func_dict[data_name]()

    return x, y
