import numpy as np
import math
from sigmoid import sigmoid, sigmoidDerivative
from scipy import io

def forwardPropogation (a, theta, m, currentIter, allA, allZ):
    if (currentIter >= len(theta)):
        return (a, allA, allZ)
    a = np.concatenate((np.ones((m,1), dtype=int), a), axis=1)
    z = a.dot(np.transpose(theta[currentIter]))
    a2 = sigmoid(z)
    allA.append(a2)
    allZ.append(z)
    return forwardPropogation(a2, theta, m, currentIter + 1, allA, allZ)

def backPropogation(allA, allZ, theta, lambdaVal, h, y, layers, m):
    delta = [0] * layers
    delta[layers-1] = h - y
    i = layers - 2
    while (i > 0):
        delta[i] = deltaCalculation(theta[i], delta[i+1], allZ[i - 1])
        i -= 1
    thetaGrad = [0] * (layers - 1)
    for i in range(0, layers - 1):
        currThetaGrad = 0
        a = np.concatenate((np.ones((allA[i].shape[0], 1), dtype=int), allA[i]), axis=1)
        if (i == 0):
            currThetaGrad = thetaGrad[i] + np.transpose(delta[i + 1][:,1:]).dot(a)
        else:
            currThetaGrad = thetaGrad[i] + np.transpose(delta[i + 1]).dot(a)
        currThetaGrad = currThetaGrad / m
        currThetaGrad[:,1:] = currThetaGrad[:,1:] + ((float(lambdaVal) / float(m)) * theta[i][:,1:])
        thetaGrad[i] = currThetaGrad
    return thetaGrad

def deltaCalculation(theta, prevDelta, z):
    z = np.concatenate((np.ones((z.shape[0], 1), dtype=int), z), axis = 1)
    return prevDelta.dot(theta) * sigmoidDerivative(z)

def costFunction(parameters, layers, X, y, lambdaVal):
    thetaSizes = []
    for i in range(0, len(layers) - 1):
        thetaSizes.append(layers[i + 1] *(layers[i] + 1))

    theta = []
    for i in range(0, len(thetaSizes)):
        startLength = 0 if (i == 0) else thetaSizes[i-1]
        theta.append(np.reshape(parameters[startLength:startLength + thetaSizes[i]], (layers[i + 1], (layers[i] + 1)), order = "F"))

    m = X.shape[0]

    reshapeY = np.zeros((m, layers[len(layers) - 1]), dtype=int)
    for i in range(0,m):
        reshapeY[i, int(y[i,0]-1)] = 1

    [h, allA, allZ] = forwardPropogation(X, theta, m, 0, [X], [])

    sumOverNodes = np.sum((reshapeY * np.log(h)) + ((1 - reshapeY) * np.log(1 - h)),axis=1)
    cost = 1
    # print bool(np.log(1-h)[0][0]
    if np.sum(reshapeY * np.log(h)) != 0:
      cost = -1 * np.sum(sumOverNodes, axis=0) / m
    regularization = 0
    for currTheta in theta:
        regularization += (sum(sum(currTheta[:,1:] * currTheta[:,1:])))
    regularization = regularization * (float(lambdaVal) / (2*float(m)))
    cost = cost + regularization

    ##########################
    #####Back Propogation#####
    ##########################
    thetaGrad = []
    for currTheta in theta:
        thetaGrad.append(np.zeros(currTheta.shape, dtype=int))

    thetaGrad = backPropogation(allA, allZ, theta, lambdaVal, h, reshapeY, len(layers), m)
    resultTheta = np.concatenate((thetaGrad[0].T.ravel(), thetaGrad[1].T.ravel()), axis=0)
    return (cost, resultTheta)
def CostFunctionTest():
    # data = io.loadmat('TestData/ex4data1.mat')
    # X = data['X']
    # y = data['y']

    # data = io.loadmat('TestData/ex4weights.mat')
    # theta1 = data['Theta1']
    # theta2 = data['Theta2']
    # theta1 = (theta1.T).ravel()
    # theta2 = (theta2.T).ravel()
    data = io.loadmat('vars.mat',  struct_as_record=True)
    # params = np.concatenate((theta1, theta2), axis=0)
    params = data['nn_params']
    X = data['X']
    y = data['y']

    costFunction(params, [3,5,3] , X, y, 3)

# CostFunctionTest()