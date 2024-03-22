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

def linear_researh():
    size = 250
    space = 0.5
    C = 1e4
    plane = np.array([1.0, 1.0])
    plane /= np.sqrt(np.dot(plane,plane))
    dir = np.array([-1/plane[0], 1/plane[1]])
    dir /= np.sqrt(np.dot(dir,dir))
    b = 10
    disp = 3
    dist = 4

    sift = 0

    mean1 = plane * (b - dist/2) + dir * sift
    mean2 = plane * (b + dist/2) + dir * sift

    sample1 = multivariate_normal.rvs(mean1, ((disp,0),(0,disp)), size)
    sample2 = multivariate_normal.rvs(mean2, ((disp,0),(0,disp)), size)

    for point in sample1:
        dist = point[0] * plane[0] + point[1] * plane[1]
        if  dist - b + space > 0:
            point[0] -= 2 * plane[0] * (dist - b + space)
            point[1] -= 2 * plane[1] * (dist - b + space)

    for point in sample2:
        dist = point[0] * plane[0] + point[1] * plane[1]
        if  dist - b - space < 0:
            point[0] -= 2 * plane[0] * (dist - b - space)
            point[1] -= 2 * plane[1] * (dist - b - space)

    ############################################################ ЛИНЕЙНОСТЬ ЛИНЕЙНОЕ ЯДРО ############################################################

    C_list = [10, 100, 1000, 10000]
    for C in C_list:
        classifier = IUSVMClassifier(kernel['linear'], C)

        sample = []
        color = []
        min_x = 1e12
        max_x = -1e12
        min_y = 1e12
        max_y = -1e12
        for k in range(size):
            min_x = min(min_x, sample1[k][0])
            max_x = max(max_x, sample1[k][0])
            min_y = min(min_y, sample1[k][1])
            max_y = max(max_y, sample1[k][1])
            min_x = min(min_x, sample2[k][0])
            max_x = max(max_x, sample2[k][0])
            min_y = min(min_y, sample2[k][1])
            max_y = max(max_y, sample2[k][1])
            sample += [sample1[k]]
            color += ["red"]
            sample += [sample2[k]]
            color += ["green"]
            classifier.addDataPair(sample1[k], 1)
            classifier.addDataPair(sample2[k], -1)
        plt.figure()
        delta_x = (max_x - min_x) / 25
        delta_y = (max_y - min_y) / 25

        x = np.arange(min_x, max_x, delta_x)
        y = np.arange(min_y, max_y, delta_y)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i, y_val in enumerate(tqdm(y)):
            for j, x_val in enumerate(x):
                Z[i,j] = classifier([x_val,y_val])

        plt.contour(X,Y,Z, levels=[-1,0,1], colors=['grey','k','grey'])

        sampleS = np.array([sample[i] for i in classifier.S])
        colorS = [color[i] for i in classifier.S]
        sampleO = np.array([sample[i] for i in classifier.O])
        colorO = [color[i] for i in classifier.O]
        sampleE = np.array([sample[i] for i in classifier.E])
        colorE = [color[i] for i in classifier.E]

        print('=========== ЛИНЕЙНО РАЗДЕЛИМЫЕ ДАННЫЕ ===========')
        print('|S| = ', len(classifier.S))
        print('|E| = ', len(classifier.E))
        print('|R| = ', len(classifier.O))

        plt.title(f'Линейное ядро, $C = {C}$')
        if len(sampleS) != 0:
            plt.scatter(sampleS[:,0], sampleS[:,1], c=colorS, marker="d", s=128)
        if len(sampleO) != 0:
            plt.scatter(sampleO[:,0], sampleO[:,1], c=colorO, marker=".", s=24)
        if len(sampleE) != 0:
            plt.scatter(sampleE[:,0], sampleE[:,1], c=colorE, marker="+", s=64)



        miss = 0
        for k in range(size):
            if classifier(sample1[k]) * 1 < 0:
                miss += 1
            if classifier(sample2[k]) * -1 < 0:
                miss += 1

        print('miss count = ', miss)
        print(miss / size*2)
        plt.show()

    ################################################################# ЛИНЕЙНО ГАУСС #################################################################

    for C in C_list:
        classifier = IUSVMClassifier(kernel['Gauss'], C)

        sample = []
        color = []
        min_x = 1e12
        max_x = -1e12
        min_y = 1e12
        max_y = -1e12
        for k in range(size):
            min_x = min(min_x, sample1[k][0])
            max_x = max(max_x, sample1[k][0])
            min_y = min(min_y, sample1[k][1])
            max_y = max(max_y, sample1[k][1])
            min_x = min(min_x, sample2[k][0])
            max_x = max(max_x, sample2[k][0])
            min_y = min(min_y, sample2[k][1])
            max_y = max(max_y, sample2[k][1])
            sample += [sample1[k]]
            color += ["red"]
            sample += [sample2[k]]
            color += ["green"]
            classifier.addDataPair(sample1[k], 1)
            classifier.addDataPair(sample2[k], -1)
        plt.figure()
        delta_x = (max_x - min_x) / 25
        delta_y = (max_y - min_y) / 25

        x = np.arange(min_x, max_x, delta_x)
        y = np.arange(min_y, max_y, delta_y)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i, y_val in enumerate(tqdm(y)):
            for j, x_val in enumerate(x):
                Z[i,j] = classifier([x_val,y_val])

        plt.contour(X,Y,Z, levels=[-1,0,1], colors=['grey','k','grey'])

        sampleS = np.array([sample[i] for i in classifier.S])
        colorS = [color[i] for i in classifier.S]
        sampleO = np.array([sample[i] for i in classifier.O])
        colorO = [color[i] for i in classifier.O]
        sampleE = np.array([sample[i] for i in classifier.E])
        colorE = [color[i] for i in classifier.E]

        print('=========== ЛИНЕЙНО РАЗДЕЛИМЫЕ ДАННЫЕ ===========')
        print('|S| = ', len(classifier.S))
        print('|E| = ', len(classifier.E))
        print('|R| = ', len(classifier.O))

        plt.title(f'Гауссово ядро, $C = {C}$')
        if len(sampleS) != 0:
            plt.scatter(sampleS[:,0], sampleS[:,1], c=colorS, marker="d", s=128)
        if len(sampleO) != 0:
            plt.scatter(sampleO[:,0], sampleO[:,1], c=colorO, marker=".", s=24)
        if len(sampleE) != 0:
            plt.scatter(sampleE[:,0], sampleE[:,1], c=colorE, marker="+", s=64)



        miss = 0
        for k in range(size):
            if classifier(sample1[k]) * 1 < 0:
                miss += 1
            if classifier(sample2[k]) * -1 < 0:
                miss += 1

        print('miss count = ', miss)
        print(miss / size*2)
        plt.show()