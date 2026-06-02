import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def neuron(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    return sigmoid(weighted_sum)


if __name__ == "__main__":
    inputs = np.array([0.9, 0.7, .3])

    weights = np.array([0.6, .3, 0.1])

    bias = -0.2

    output = neuron(inputs, weights, bias)

    print(f"Inputs: cloud cover={inputs[0]}, humidity={inputs[1]}, wind speed={inputs[2]}")
    print(f"Output (probability): {output:.4f}")
    print(f"Decision: {'bring umbrella' if output > 0.5 else 'leave it'}")