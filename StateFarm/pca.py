from scipy import misc, io, linalg
import numpy as np
import matplotlib.pyplot as plot
import os
from PIL import Image

def rgbToGray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.597, 0.114])

def fsvd(X, k,i):
    if (X.shape[0] < X.shape[1]):
        X = np.transpose(X)
        isTransposed = True
    else:
        isTransposed = False

    n = X.shape[1]
    l = k + 2
    G = np.random.randn(n, l)

    prev = X.dot(G)
    H = np.array(prev)
    for j in range(1,i+1):
        i = X.dot(X.T.dot(prev))
        H = np.hstack([H, i])
        prev = i

    Q = np.linalg.qr(H, 'economic')

    T = np.transpose(X).dot(Q)

    [Vt, St, W] = np.linalg.svd(T, full_matrices=False)

    Ut = Q.dot(W)

    if isTransposed:
        V = Ut[:,0:k]
        U = Vt[:,0:k]
    else:
        U = Ut[:,0:k]
        V = Vt[:,0:k]
    S = np.diag(St)
    S = S[0:k,0:k]
    return (U, S, V)


def projectData(X, U, k):
    return X.dot(U[:,0:k])

def pca(X, fileNum):
    X = normalize(X)
    transposeX = X.T
    (m,n) = X.shape
    (mT, nT) = transposeX.shape
    covariance = transposeX.dot(X)
    covariance = covariance/m
    [U, S, V] = np.linalg.svd(covariance)
    Z = projectData(X, U, 1024)
    io.savemat('z' + str(fileNum) + '.mat', {'z': Z})


def normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X = (X - mu) / sigma
    return X

data = io.loadmat('./Data/c4.mat')
images = data['images']
images = np.array(images)
pca(images, 4)
