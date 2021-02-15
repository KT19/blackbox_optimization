# blackbox_optimization
Implementing some blackbox optimization methods

## Contents
1. TestFunctions.py: Define test functions
2. BOmethods.py: Define optimization method
3. black_box_optimize.ipynb: notebook for visualization

## Implemented optimization method
1. random search
2. Gaussian search
3. Simulated annealing
4. Whale Optimization Algorithm (WOA)
5. Particle Swam Optimization (PSO)

## Usage
In each test function, needs definition of dimensions of inputs variables.
And in optimization method, you need to set some parameters
such as iteration, population, temperature, and so on.
Let's consider following case:


### Define test function
Test function: Rastrigin

In this case, you write code such as:
```python3
test_obj = Rastrigin(ndim=10)
```
This means test_obj is a Rastrigin function with a dimension of 10.

### Define optimization method
Optimization method: particle swam optimization

In this case, you write code such as:
```python3
optimizer = PSO(test_obj, ndim=10, ranges = [[-5.12,5.12] for n in range(10)])
val, variable = optimizer.fit(iteration=1000, population=10, c1=0.5, c2=0.7)
```
Here, PSO's arguments are target test function, a dimension of search variables, and a range of each dimension. These arguments are the same in each optimization method.
Once you can create an optimization object, then conduct optimization by .fit() with some arguments.
In the above case, PSO is conducted with 1000 iteration and, in each iteration, 10 search agents are evaluated.
And the found value and variables are returned as a list format  (obtain these values per iteration).

For additional example, please see the notebook.
