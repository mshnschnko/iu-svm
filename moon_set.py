import os
import threading
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from sklearn.datasets import make_moons

from IU_SVM import IUSVMClassifier
from Kernel import GaussKernel, LinearKernel, PolyKernel


# def linKernel(x1, x2):
#     return np.dot(x1, x2)

# def othKernel(x1, x2, sigma):
#     return np.exp(-np.dot(x1 - x2, x1 - x2) / (2 * sigma*sigma))


def nonlinear_researh():
    size = 1000
    '''Генерация тучек'''
    # # Генерация данных для первого класса (кластера)
    # num_samples_class1 = 500
    # mean_class1 = [2, 2]  # Среднее значение для первого класса
    # covariance_class1 = [[1, 0], [0, 1]]  # Ковариационная матрица для первого класса
    # class1_data = np.random.multivariate_normal(mean_class1, covariance_class1, num_samples_class1)

    # # Генерация данных для второго класса (кластера)
    # num_samples_class2 = 500
    # mean_class2 = [-2, -2]  # Среднее значение для второго класса
    # covariance_class2 = [[1, 0], [0, 1]]  # Ковариационная матрица для второго класса
    # class2_data = np.random.multivariate_normal(mean_class2, covariance_class2, num_samples_class2)

    # # Объединение данных для образования общего набора данных
    # orig_data = np.vstack((class1_data, class2_data))
    # labels = np.hstack((np.zeros(num_samples_class1), np.ones(num_samples_class2)))  # Метки классов
    # dataset = np.hstack((orig_data, labels.reshape(-1, 1)))
    # np.random.shuffle(dataset)

    '''Генерация лун'''
    data, labels = make_moons(n_samples=size, noise=0.2, random_state=42)
    dataset = np.hstack((data, labels.reshape(-1, 1)))
    np.random.shuffle(dataset)
    # print(dataset)
    # return


    # print(len(dataset[:,2].where(dataset[:,2] == 0)))
    # print(len(np.where(dataset[:,2] == 1.0)[0]))
    # return
    # Визуализация данных
    plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2], cmap='viridis', s=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Два пересекающихся класса')
    plt.colorbar(label='Класс')
    # plt.show()

    # plt.figure()
    # plt.scatter(orig_data[:,0], orig_data[:,1], s=10)
    # plt.show()

    C_list = [10, 100, 1000, 10000, 1000000]
    kernels = [GaussKernel(sigma=1.0), GaussKernel(sigma=4.0), GaussKernel(sigma=8.0), PolyKernel(degree=2, c=1.0), PolyKernel(degree=3, c=1.0)]
    # print(kernel)
    for kernel in tqdm(kernels):
        for C in C_list:
            data = deepcopy(dataset)
            sample1 = []
            sample2 = []
            sample = []
            color = []

            min_x = 1e12
            max_x = -1e12
            min_y = 1e12
            max_y = -1e12

            classifier = IUSVMClassifier(kernel, C)

            for i, point in enumerate(data):
                min_x = min(min_x, point[0])
                max_x = max(max_x, point[0])
                min_y = min(min_y, point[1])
                max_y = max(max_y, point[1])
                if point[2] == 0.0:
                    sample1.append(point)
                    sample.append(point)
                    color.append("red")
                elif point[2] == 1.0:
                    sample2.append(point)
                    sample.append(point)
                    color.append("green")

            for i, point in enumerate(data[:100]):
                if color[i] == "red":
                    # print(point[:2])
                    classifier.addDataPair(point[:2], 1)
                else:
                    classifier.addDataPair(point[:2], -1)

            plt.figure()

            delta_x = (max_x - min_x) / 100
            delta_y = (max_y - min_y) / 100

            x = np.arange(min_x, max_x, delta_x)
            y = np.arange(min_y, max_y, delta_y)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            for i, y_val in enumerate(y):
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
            for j, point in enumerate(dataset[100:]):
                i = 100 + j
                val = 0
                col = None
                if color[i] == 'red':
                    col = 'm'
                    val = 1
                if color[i] == 'green':
                    col = 'c'
                    val = -1
                if classifier(point[:2]) * val < 0:
                    testE.append(point[:2])
                    colorTestE.append(col)
                else:
                    testO.append(point[:2])
                    colorTestO.append(col)

            testO = np.array(testO)
            testE = np.array(testE)

            plt.title(f'{kernel}, $C = {C}$')
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

            dirname = os.path.join('figures', kernel.filename)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            fig_filename = os.path.join(dirname, f'{kernel.filename}_C_{C}.png')
            txt_filename = os.path.join(dirname, f'{kernel.filename}_C_{C}.txt')
            plt.savefig(fig_filename)
            
            miss = 0
            for i, point in enumerate(data):
                val = 0
                if color[i] == 'red':
                    val = 1
                if color[i] == 'green':
                    val = -1
                if classifier(point[:2]) * val < 0:
                    miss += 1
                
            # print('=========== ЛИНЕЙНО НЕРАЗДЕЛИМЫЕ ДАННЫЕ ===========')
            # print('|S| = ', len(classifier.S))
            # print('|E| = ', len(classifier.E))
            # print('|R| = ', len(classifier.O))
            # print('miss count = ', miss)
            # print(miss / size)
            with open(txt_filename, 'w') as f:
                f.write(f'|S| = {len(classifier.S)}\n')
                f.write(f'|E| = {len(classifier.E)}\n')
                f.write(f'|R| = {len(classifier.O)}\n')
                f.write(f'miss count = {miss}\n')
                f.write(f'error (%) = {miss / size}\n')
        # plt.show()


if __name__ == '__main__':
    nonlinear_researh()