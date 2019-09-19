import numpy as np
from scipy import io
import os
import re
import math
from costFunction import costFunction
from predict import predict

def loadAllData(folderName):
    X = np.empty((0,1024))
    Y = np.empty((0,1))
    Xcv = np.empty((0, 1024))
    Ycv = np.empty((0, 1))

    for file in os.listdir(folderName):
        filePath = folderName + '/' + file
        data = io.loadmat(filePath)
        currentData = np.array(data['z'])
        numOfRows = currentData.shape[0]

        X = np.concatenate((X, currentData), axis=0)

        regex = re.compile(r'\d+')
        yVal = regex.findall(file)[0]
        currentY = np.full((numOfRows, 1), int(yVal), dtype=int)
        Y = np.concatenate((Y, currentY), axis=0)
    return(X, Y)


def randomInitialization(rows, cols, epsilon):
    randomArray = np.random.rand(rows, cols)
    return (randomArray * 2 * epsilon) - epsilon

def reshapeTheta(parameters):
    thetaSizes = []
    for i in range(0, len(layers) - 1):
        thetaSizes.append(layers[i + 1] *(layers[i] + 1))

    theta = []
    for i in range(0, len(thetaSizes)):
        startLength = 0 if (i == 0) else thetaSizes[i-1]
        theta.append(np.reshape(parameters[startLength:startLength + thetaSizes[i]], (layers[i + 1], (layers[i] + 1)), order = "F"))
    return theta

X, Y = loadAllData('z')
inputLayerSize = 1024
hiddenLayerSize = 2048
numLabels = 10
layers = [inputLayerSize, hiddenLayerSize, numLabels]
lambdaVal = 1

data = io.loadmat('./thetaPrediction.mat')
theta1 = data['theta1']
theta2 = data['theta2']

params = np.concatenate((theta1.T.ravel(), theta2.T.ravel()), axis=0)
cost = 1
alpha = 0.05
while True:
    [cost, thetaGrad] = costFunction(params, layers, X, Y, lambdaVal)
    resultTheta = reshapeTheta(thetaGrad)
    theta1 = theta1 - alpha * resultTheta[0]
    theta2 = theta2 - alpha * resultTheta[1]
    params = np.concatenate((theta1.T.ravel(), theta2.T.ravel()), axis=0)
    # if (cost < 1):
    #     io.savemat('thetaPrediction.mat', {'theta1': theta1, 'theta2': theta2})
    #     alpha = 0.02
    # if (cost < 1):
    #     prediction = predict(theta1, theta2, Xcv)
    #     print(np.mean(np.equal(prediction, Ycv).astype(int)))
    if (cost == 0.3):
        io.savemat('thetaPrediction03.mat', {'theta1': theta1, 'theta2': theta2})
    if (cost == 0.35):
        io.savemat('thetaPrediction035.mat', {'theta1': theta1, 'theta2': theta2})
    if (cost == 0.25):
        io.savemat('thetaPrediction025.mat', {'theta1': theta1, 'theta2': theta2})
    if (cost == 0.20):
        io.savemat('thetaPrediction20.mat', {'theta1': theta1, 'theta2': theta2})
    if (cost == 0.10):
        io.savemat('thetaPrediction10.mat', {'theta1': theta1, 'theta2': theta2})
    if (cost == 0.15):
        io.savemat('thetaPrediction15.mat', {'theta1': theta1, 'theta2': theta2})
    if (cost == 0.08):
        io.savemat('thetaPrediction08.mat', {'theta1': theta1, 'theta2': theta2})
    if (cost == 0.06):
        io.savemat('thetaPrediction06.mat', {'theta1': theta1, 'theta2': theta2})
    if (cost == 0.04):
        io.savemat('thetaPrediction04.mat', {'theta1': theta1, 'theta2': theta2})
    if (cost == 0.02):
        io.savemat('thetaPrediction02.mat', {'theta1': theta1, 'theta2': theta2})
    if (cost == 0.01):
        io.savemat('thetaPrediction01.mat', {'theta1': theta1, 'theta2': theta2})
        break
    print (cost)
    print("===========================================")
