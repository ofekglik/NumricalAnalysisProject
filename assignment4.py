"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random


def plu(A):
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    P = np.eye(n, dtype=np.double)
    for i in range(n):
        for k in range(i, n):
            if ~np.isclose(U[i, i], 0.0):
                break
            U[[k, k+1]] = U[[k+1, k]]
            P[[k, k+1]] = P[[k+1, k]]
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i]
    return P, L, U

def forward_sub(L, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double)
    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]
    return y


def back_sub(U, y):
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double)
    x[-1] = y[-1] / U[-1, -1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i:], x[i:])) / U[i,i]
    return x

def inverse(A):
    n = A.shape[0]
    b = np.eye(n)
    Ainv = np.zeros((n, n))
    P, L, U = plu(A)
    for i in range(n):
        y = forward_sub(L, np.dot(P, b[i, :]))
        Ainv[i, :] = back_sub(U, y)
    return Ainv


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        pass

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        T = time.time()
        y = f(a)
        sampletime = time.time() - T
        rest_time = maxtime - sampletime
        flag = True
        if rest_time == maxtime:
            xs = np.linspace(a, b, 10000, endpoint=True)
        else:
            xs = np.linspace(a, b, max(1, int(1.4 * rest_time / (sampletime+0.01))), endpoint=True)
        try:
            ys = f(xs)
        except:
            f = np.vectorize(f)
            ys = f(xs)
            flag = False

        initial_time = time.time() - T

        # print("intitial time = ", initial_time)
        rest_time = maxtime - initial_time - 0.4
        i = 1
        if flag:
            while(time.time() - T) <= rest_time:
                ys = ys + f(xs)
                i += 1
        else:
            while (time.time() - T) <= rest_time:
                ys = ys + f(xs)
                i += 1
        # T2 = time.time()
        ys = ys / i
        yy = np.diff(ys)
        h = 0
        while np.sum(yy) >= 0.00001:
            yy = np.diff(yy)
            h += 1
        if h != 0:
            d = h
        if abs(np.mean(ys) - ys[0]) <= 0.001:
            return lambda line: ys[0]
        A = np.vander(xs, d+1)
        AT = A.T
        c = (inverse((AT.dot(A))).dot(AT.dot(ys)))[::-1]
        poly = np.polynomial.Polynomial(coef=c)
        # T = time.time() - T
        # T2 = time.time() - T2
        # print("t2=", T2)
        # print(maxtime)
        # print(F"RunTime = {T}")

        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        # replace these lines with your solution
        result = lambda x: poly(x)

        return result


##########################################################################

#
# import unittest
# from sampleFunctions import *
# from tqdm import tqdm
#
#
# class TestAssignment4(unittest.TestCase):
#
#     def test_return(self):
#         f = NOISY(0.01)(poly(1,1,1))
#         ass4 = Assignment4A()
#         T = time.time()
#         shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
#         T = time.time() - T
#         self.assertLessEqual(T, 5)
#
#     def test_delay(self):
#         f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))
#
#
#         ass4 = Assignment4A()
#         T = time.time()
#         shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
#         T = time.time() - T
#         self.assertGreaterEqual(T, 5)
#
#     def test_err(self):
#         f = poly(1,1,1)
#         nf = NOISY(1)(f)
#
#
#         ass4 = Assignment4A()
#         T = time.time()
#         ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
#         T = time.time() - T
#         mse=0
#         for x in np.linspace(0,1,10):
#
#             self.assertNotEquals(f(x), nf(x))
#             mse+= (f(x)-ff(x))**2
#         mse = mse/10
#         print(mse)
#
#
#
#
#
#
# if __name__ == "__main__":
#     unittest.main()
