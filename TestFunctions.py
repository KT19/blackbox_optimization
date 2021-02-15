#-*-coding:utf-8-*-
import numpy as np
import math

#All test function is from:
#https://en.wikipedia.org/wiki/Test_functions_for_optimization
class Rastrigin:
    def __init__(self, ndim=100, A=10):
        print("Rastrigin function")
        print("search space:  [-5.12 <= x_i <=5.12]")
        print("optimal: f(0,...,0) = 0")
        self.ndim = ndim
        self.A = A

    def calc(self, X):
        f = self.A*self.ndim
        for x in X:
            f += (x**2-self.A*np.cos(2*np.pi*x))

        return f

    def plot(self):
        """
        show for visualization
        """
        n = 2
        X1 = np.arange(-5.2, 5.2, 0.01)
        X2 = np.arange(-5.2, 5.2, 0.01)
        x, y = np.meshgrid(X1, X2)

        z = self.A*n + (x**2 - self.A*np.cos(2*np.pi*x)) + (y**2 - self.A*np.cos(2*np.pi*y))

        return x, y, z

class Ackley:
    def __init__(self, ndim = 2):
        print("Ackley function")
        print("search space: -5 <= x, y <= 5")
        print("optimal: f(0,0) = 0")
        assert ndim == 2, "dimension must be 2"
        self.ndim = ndim

    def calc(self, X):
        f = -20*np.exp(-0.2*np.sqrt(0.5*(X[0]**2+X[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*X[0])+np.cos(2*np.pi*X[1]))) + np.exp(1)+20

        return f

    def plot(self):
        X = np.arange(-5.1, 5.1, 0.1)
        Y = np.arange(-5.1, 5.1, 0.1)

        x, y = np.meshgrid(X, Y)
        z = -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + np.exp(1)+20

        return x, y, z

class Sphere:
    def __init__(self, ndim=100):
        print("Sphere function")
        print("Search space: -10 <= x_i <= 10")
        print("optimal: f(0,...,0) = 0")
        self.ndim = ndim

    def calc(self, X):
        f = np.sum(X**2)

        return f

    def plot(self):
        X = np.arange(-6, 6, 0.1)
        Y = np.arange(-6, 6, 0.1)

        x, y = np.meshgrid(X, Y)
        z = x**2*y**2

        return x, y, z

class Easom:
    def __init__(self, ndim=2):
        print("Easom function")
        print("search space: -100 <= x, y <= 100")
        print("optimal: f(pi, pi) = -1")
        assert ndim == 2, "dimension must be 2"
        self.ndim = ndim

    def calc(self, X):
        f = -np.cos(X[0])*np.cos(X[1])*np.exp(-((X[0] - np.pi)**2 + (X[1] - np.pi)**2))

        return f

    def plot(self):
        X = np.arange(-5, 5, 0.1)
        Y = np.arange(-5, 5, 0.1)

        x, y = np.meshgrid(X, Y)
        z = -np.cos(x)*np.cos(y)*np.exp(-((x - np.pi)**2 + (y - np.pi)**2))

        return x, y, z

class Eggholder:
    def __init__(self, ndim=2):
        print("Eggholder function")
        print("search space: -512 <= x, y <= 512")
        print("optimal: f(512, 404.2319) = -959.6407")
        assert ndim == 2, "dimension must be 2"
        self.ndim = ndim

    def calc(self, X):
        f = -(X[1] + 47)*np.sin(np.sqrt(np.abs(X[0]/2 + (X[1]+47)))) - X[0]*np.sin(np.sqrt(np.abs(X[0] - (X[1]+47))))

        return f

    def plot(self):
        X = np.arange(-513, 513, 1)
        Y = np.arange(-513, 513, 1)

        x, y = np.meshgrid(X, Y)
        z = -(y + 47)*np.sin(np.sqrt(np.abs(x/2 + (y+47)))) - x*np.sin(np.sqrt(np.abs(x - (y+47))))

        return x, y, z

class Stybliski_Tang:
    def __init__(self, ndim=100):
        print("Stybliski-Tang function")
        print("Search space: -5 <= x_i <= 5")
        print("optimal: -39. 16617n < f(-2.903534, ..., -2.903534) < -39.16616n")
        self.ndim = ndim

    def calc(self, X):
        f = 0
        for x in X:
            f += x**4 - 16*x**2 + 5*x

        f /= 2

        return f

    def plot(self):
        X = np.arange(-6, 6, 0.1)
        Y = np.arange(-6, 6, 0.1)
        x, y = np.meshgrid(X, Y)

        z = ( (x**4 - 16*x**2 + 5*x) + (y**4 - 16*y**2 + 5*y) )/ 2

        return x, y, z
