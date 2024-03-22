import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from IU_SVM import IUSVMClassifier

def linKernel(x1, x2):
    return np.dot(x1, x2)

def othKernel(x1, x2, sigma):
    return np.exp(-np.dot(x1 - x2, x1 - x2) / (2 * sigma*sigma))

sigma = 4

kernel = {
    'linear': linKernel,
    'Gauss': lambda x, y:othKernel(x,y,sigma)
}

################################################################# НЕЛИНЕЙНОСТЬ #################################################################

def nonlinear_researh():
    size = 1000
    C = 1e1
    disp = 10
    mult = 1.4
    orig_data = multivariate_normal.rvs((0,0), ((disp,0),(0,disp)), size)
    rad = 4

    plt.figure()
    plt.scatter(orig_data[:,0], orig_data[:,1], s=10)
    # plt.show()

    C_list = [10, 100, 1000, 10000]

    for C in C_list:
        data = deepcopy(orig_data)
        sample1 = []
        sample2 = []
        sample = []
        color = []

        min_x = 1e12
        max_x = -1e12
        min_y = 1e12
        max_y = -1e12

        classifier = IUSVMClassifier(kernel['Gauss'], C)

        for i, point in enumerate(data):
            min_x = min(min_x, point[0])
            max_x = max(max_x, point[0])
            min_y = min(min_y, point[1])
            max_y = max(max_y, point[1])
            if np.sqrt(np.dot(point, point)) > rad:
                data[i] *= [1, 1]
                data[i] *= mult
                point = data[i]
                sample1.append(point)
                sample.append(point)
                color.append("red")
            else:
                data[i] *= [1, 1]
                sample2.append(point)
                sample.append(point)
                color.append("green")

        for i, point in enumerate(data[:100]):
            if color[i] == "red":
                classifier.addDataPair(point, 1)
            else:
                classifier.addDataPair(point, -1)

        plt.figure()

        delta_x = (max_x - min_x) / 100
        delta_y = (max_y - min_y) / 100

        x = np.arange(min_x, max_x, delta_x)
        y = np.arange(min_y, max_y, delta_y)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i, y_val in enumerate(tqdm(y)):
            for j, x_val in enumerate(x):
                Z[i,j] = classifier([x_val,y_val])

        plt.contour(X,Y,Z, levels=[-1,0,1], colors=['grey','k','grey'])

        sampleS = np.array([data[i] for i in classifier.S])
        colorS = [color[i] for i in classifier.S]
        sampleO = np.array([data[i] for i in classifier.O])
        colorO = [color[i] for i in classifier.O]
        sampleE = np.array([data[i] for i in classifier.E])
        colorE = [color[i] for i in classifier.E]

        testO = []
        testE = []
        colorTestO = []
        colorTestE = []
        for j, point in enumerate(data[100:]):
            i = 100 + j
            val = 0
            col = None
            if color[i] == 'red':
                col = 'm'
                val = 1
            if color[i] == 'green':
                col = 'c'
                val = -1
            if classifier(point) * val < 0:
                testE.append(point)
                colorTestE.append(col)
            else:
                testO.append(point)
                colorTestO.append(col)
            
        print('=========== ЛИНЕЙНО НЕРАЗДЕЛИМЫЕ ДАННЫЕ ===========')
        print('|S| = ', len(classifier.S))
        print('|E| = ', len(classifier.E))
        print('|R| = ', len(classifier.O))

        testO = np.array(testO)
        testE = np.array(testE)

        plt.title(f'Гауссово ядро, $C = {C}$')
        if len(sampleS) != 0:
            plt.scatter(sampleS[:,0], sampleS[:,1], c=colorS, marker="d", s=128)
        if len(sampleO) != 0:
            plt.scatter(sampleO[:,0], sampleO[:,1], c=colorO, marker=".", s=24)
        if len(sampleE) != 0:
            plt.scatter(sampleE[:,0], sampleE[:,1], c=colorE, marker="+", s=64)
        if len(testO) != 0:
            plt.scatter(testO[:,0], testO[:,1], c=colorTestO, marker=".", s=24)
        if len(testE) != 0:
            plt.scatter(testE[:,0], testE[:,1], c=colorTestE, marker="+", s=64)



        miss = 0
        for i, point in enumerate(data):
            val = 0
            if color[i] == 'red':
                val = 1
            if color[i] == 'green':
                val = -1
            if classifier(point) * val < 0:
                miss += 1
        print('miss count = ', miss)
        print(miss / size)
        plt.show()
