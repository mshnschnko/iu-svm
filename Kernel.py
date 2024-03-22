import numpy as np
from abc import ABC, abstractmethod

class Kernel(ABC):
    @abstractmethod
    def __init__(self, name: str, filename: str):
        self.name = name
        self.filename = filename

    @abstractmethod
    def __call__(self, x, y):
        pass

    @abstractmethod
    def __str__(self) -> str:
        return self.name

class LinearKernel(Kernel):
    def __init__(self):
        super().__init__(name='Линейное ядро', filename='linKernel')

    def __call__(self, x, y):
        return np.dot(x, y)
    
class GaussKernel(Kernel):
    def __init__(self, sigma: float = 1.0):
        super().__init__(name='Гауссово ядро', filename=f'GaussKernel_sigma_{sigma:.2f}')
        self.sigma = sigma

    def __call__(self, x, y):
        return np.exp(-np.dot(x - y, x - y) / (2 * self.sigma * self.sigma))
    
    def __str__(self) -> str:
        return super().__str__() + f' с σ = {self.sigma}'

class PolyKernel(Kernel):
    def __init__(self, degree: int = 2, c: float = 1.0):
        super().__init__(name='Полиномиальное ядро', filename=f'PolyKernel_degree_{degree}_const_{c}')
        self.degree = degree
        self.c = c

    def __call__(self, x, y):
        return (np.dot(x, y) + self.c) ** self.degree
    
    def __str__(self) -> str:
        return super().__str__() + f' степени {self.degree} и константой {self.c}'