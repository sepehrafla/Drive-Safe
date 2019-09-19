import numpy as np

def sigmoid(z):
    z =  z.astype(float)
    return 1 / (1 + np.exp(-z))

def sigmoidDerivative(z):
    return sigmoid(z) * (1 - sigmoid(z))
