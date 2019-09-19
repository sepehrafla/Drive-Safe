import numpy as np
from sigmoid import sigmoid
def predict(Theta1, Theta2, X):
    X = np.concatenate((np.ones((X.shape[0], 1), dtype=int), X), axis=1)
    h1 = sigmoid(X.dot(np.transpose(Theta1)))
    h1 = np.concatenate((np.ones((h1.shape[0], 1), dtype=int), h1), axis=1)
    h2 = sigmoid(h1.dot(np.transpose(Theta2)))
    mapToLabel = np.argmax(h2, axis=1)[np.newaxis]
    return np.transpose(mapToLabel)