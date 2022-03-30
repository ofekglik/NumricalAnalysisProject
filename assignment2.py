"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable



class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def bi_section(self, f, a, b, maxerr):
        c = (b + a) * 0.5
        while abs(b - a) >= maxerr*0.01:
            if np.sign(f(a)) != np.sign(f(c)):
                b = c
            else:
                a = c
            c = (b + a) * 0.5
        return c

    def deriv(self, f, x):
        delta = 0.0000001
        oppdelta = 10000000
        g = (f(x + delta) - f(x)) * oppdelta
        return g

    def newton_raphson(self, f, a, b, max_error):
        x1 = self.bi_section(f, a, b, max_error)
        x0 = x1
        n = 0
        if self.deriv(f, x1) == 0 and abs(f(x1)) < max_error:
            return x1
        while abs(f(x1)) > max_error:
            g = self.deriv(f, x0)
            x1 = x0 - f(x0) / g
            x0 = x1
            n += 1
            if n > 20:
                return
        if b >= x1 >= a:
            return x1

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        intersection_list = []
        # intersec_func = lambda x: f1(x) - f2(x)
        intersec_func = self.get_func(f1, f2)
        intervals = np.linspace(a, b, int(np.ceil(abs(b-a)))*30, endpoint=True)
        for i in range(len(intervals) - 1):
            if np.sign(intersec_func(intervals[i])) != np.sign(intersec_func(intervals[i + 1])) or \
                    np.sign(self.deriv(intersec_func, intervals[i])) != \
                    np.sign(self.deriv(intersec_func, intervals[i + 1])):
                root = self.newton_raphson(intersec_func, intervals[i], intervals[i + 1], maxerr)
                if root:
                    intersection_list.append(root)

        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution

        return intersection_list

    def get_func(self, f1, f2):
        def g(x):
            try:
                return f1(x) - f2(x)
            except:
                return np.inf
        return g


##########################################################################
#
#
# import unittest
# from sampleFunctions import *
# from tqdm import tqdm
#
#
# class TestAssignment2(unittest.TestCase):
#
#     def test_sqr(self):
#
#         ass2 = Assignment2()
#
#         f1 = np.poly1d([-1, 0, 1])
#         f2 = np.poly1d([1, 0, -1])
#
#         X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
#
#         for x in X:
#             self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
#
#     def test_poly(self):
#
#         ass2 = Assignment2()
#
#         f1, f2 = randomIntersectingPolynomials(10)
#         f1 = lambda x: np.cos(x / 6) * x - x - 4
#         f2 = lambda x: 10 * np.sin(6 * x)
#         X = ass2.intersections(f1, f2, -100, 100, maxerr=0.001)
#         print(len(X))
#
#         for x in X:
#             self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
#         print("\n", X)
#
#
# if __name__ == "__main__":
#     unittest.main()
