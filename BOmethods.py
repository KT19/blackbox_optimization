#-*-coding:utf-8-*-
import numpy as np
import math
from copy import copy

def uniform_random(low, high):
    return low + np.random.rand()*(high-low+0.1) #because rand() is defined [0, 1)

def modify(low, val, high):
    if val < low:
        val = low
    if val > high:
        val = high
    return val

class BO_Optimize:
    def __init__(self, target, ndim, ranges):
        """
        args:
        target: target object for optimization
        ndim: dimention of variables
        ranges: range of each dimension
        """
        self.target_obj = target
        self.ndim = ndim
        self.ranges = ranges

    def fit(self):
        raise "Not Implemented Error"

class RandomSearch(BO_Optimize):
    def fit(self, iteration):
        print("Random Search")
        optimal_results = [] #best result per iteration
        optimal_variables = [] #best variables per iteration
        cur_results= 1e9 #current best results
        cur_variables = [] #current best variables

        for it in range(iteration):
            X = [uniform_random(self.ranges[n][0], self.ranges[n][1]) for n in range(self.ndim)]
            output = self.target_obj.calc(X)

            if output < cur_results:
                cur_results = output
                cur_variables = X

            optimal_results.append(cur_results)
            optimal_variables.append(cur_variables)

        return optimal_results, optimal_variables

class Gaussian(BO_Optimize):
    def fit(self, iteration, n_sample):
        """
        args:
        iteration: number of iterations
        n_sample: number of samples per iteration
        """
        lam = math.ceil(0.2*n_sample)
        print("gaussian")
        optimal_results = [] #best result per iteration
        optimal_variables = [] #best variables per iteration
        cur_results= 1e9 #current best results
        cur_variables = [] #current best variables

        means = [uniform_random(self.ranges[n][0], self.ranges[n][1]) for n in range(self.ndim)]
        stds = [(self.ranges[n][1]-self.ranges[n][0])/4 for n in range(self.ndim)]

        for it in range(iteration):
            ranks = []
            for sample in range(n_sample): #for each samples
                X = [np.random.normal(loc=means[n], scale=stds[n], size=1) for n in range(self.ndim)]
                #modify range
                X = [modify(self.ranges[i][0], x.item(), self.ranges[i][1]) for i,x in enumerate(X)]
                output = self.target_obj.calc(X)
                ranks.append([output, X])
                if output < cur_results:
                    cur_results = output
                    cur_variables = X

            #update heuristics
            ranks.sort(key=lambda x: x[0])
            nextMeans = []
            for _, x in ranks[:lam]:
                nextMeans.append(x)
            means = np.mean(nextMeans, 0) #mean

            #save transtion
            optimal_results.append(cur_results)
            optimal_variables.append(cur_variables)

        return optimal_results, optimal_variables


class WOA(BO_Optimize):
    def __update_params(self, itr_ratio):
        """
        args:
        itr_ratio: ratio for iteration

        return:
        a: decrease from 2 to 0
        A: 2ar - r
        C: 2r
        l: [0, 1]
        """
        #linearly decrease from 2 to 0
        a = np.array([2-2*itr_ratio for n in range(self.ndim)])
        #random vetor in [0, 1)
        r = np.random.rand(self.ndim)

        #compute A
        A = 2*a*r-a
        #compute C
        C = 2*r
        #generate l
        l = np.random.rand()

        return a, A, C, l

    def __optimal_fittness(self, X):
        res = 1e9
        optim = []
        for x in X:
            val = self.target_obj.calc(x)
            if val < res:
                res = val
                optim = x
        return res, optim

    def fit(self, iteration, population, b=0.5):

        print("Whale optimization")
        opt_val = 1e9
        opt_variable = []
        optimal_results = [] #best result per iteration
        optimal_variables = [] #best variables per iteration
        #initialize
        X = [ np.array([uniform_random(self.ranges[n][0], self.ranges[n][1]) for n in range(self.ndim)]) for x in range(population) ]
        _, bestX = self.__optimal_fittness(X) #the best search agent

        for it in range(iteration): #until iteraion end
            cur_X = []
            for x in X: #for each search agent
                #update a, A, C, l
                a, A, C, l = self.__update_params(it/iteration)
                prob = np.random.rand()
                if prob < 0.5: # Eq. 2.6
                    if np.linalg.norm(A) < 1:
                        #update current search agent position Eq. 2.1
                        D = np.abs(C*bestX - x)
                        cur = bestX - A*D
                    else:
                        #update current search agent position Eq. 2.8
                        idx_rnd = np.random.randint(0, population, 1)
                        X_rnd = X[idx_rnd.item()]
                        D = np.abs(C*X_rnd - x)
                        cur = bestX - A*D
                else:
                    #spiral update E.q. 2.5
                    D = np.abs(bestX - x)
                    cur = D*np.exp(b*l)*np.cos(2*np.pi*l) + bestX

                cur_X.append(cur) #next search agent

            #amend positions
            cur_X = [ np.array([modify(self.ranges[i][0], x, self.ranges[i][1]) for i,x in enumerate(cur_X[n])]) for n in range(population)]

            cur_val, bestX = self.__optimal_fittness(cur_X) #fitness of each agent

            X = copy(cur_X)

            if cur_val < opt_val: #for results
                opt_val = cur_val
                opt_variable = bestX

            optimal_results.append(opt_val)
            optimal_variables.append(opt_variable)

        return optimal_results, optimal_variables

class PSO(BO_Optimize):
    def fit(self, iteration, population, w=0.5, c1=0.5, c2=0.5):
        print("Particle Swam optimization")

        opt_val = 1e9
        opt_variable = []
        optimal_results = [] #best result per iteration
        optimal_variables = [] #best variables per iteration

        #Initialize
        X = np.array([ [uniform_random(self.ranges[n][0], self.ranges[n][1]) for n in range(self.ndim)] for x in range(population) ])
        bestX_fitness = np.array([[1e9] for x in range(population)]) #ever best of each particle
        bestX_variable = np.array([[uniform_random(self.ranges[n][0], self.ranges[n][1]) for n in range(self.ndim)] for x in range(population)])
        V = np.array([ [(self.ranges[n][1] - self.ranges[n][0])/10 for n in range(self.ndim)] for x in range(population) ])

        for it in range(iteration):
            for i, x in enumerate(X): #for each particle
                fitness = self.target_obj.calc(x)
                if fitness < bestX_fitness[i]: #update best position
                    bestX_fitness[i] = fitness
                    bestX_variable[i] = x

                #global optim
                if fitness < opt_val:
                    opt_val = fitness
                    opt_variable = x

            nextX = []
            nextV = []
            for x, v, p_variable in zip(X, V, bestX_variable): #for each particle
                nv = w*v + c1*np.random.rand()*(p_variable - x) + c2*np.random.rand()*(opt_variable - x)
                nx = x+v

                nx = [modify(self.ranges[i][0], d, self.ranges[i][1]) for i,d in enumerate(nx)] #modify ranges
                nextX.append(nx)
                nextV.append(nv)

            X = copy(np.array(nextX))
            V = copy(np.array(nextV))

            optimal_results.append(opt_val)
            optimal_variables.append(opt_variable)

        return optimal_results, optimal_variables
