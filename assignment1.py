"""
In this assignment you should interpolate the given function.
"""
import math
import time
import random
import numpy as np


class Assignment1:
    def __init__(self):
        self.points = None

        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def solver(self, a, b, c, d):

        number = len(d)
        ac, bc, cc, dc = map(np.array, (a, b, c, d))
        for i in range(1, number):
            mc = ac[i - 1] / bc[i - 1]
            bc[i] = bc[i] - mc * cc[i - 1]
            dc[i] = dc[i] - mc * dc[i - 1]

        xc = bc
        xc[-1] = dc[-1] / bc[-1]

        for il in range(number - 2, -1, -1):
            xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

        return xc

    def calc_coef(self, points):
        n = len(points) - 1

        c_matrix = 4 * np.identity(n)
        np.fill_diagonal(c_matrix[1:], 1)
        np.fill_diagonal(c_matrix[:, 1:], 1)
        c_matrix[0, 0] = 2
        c_matrix[n - 1, n - 1] = 7
        c_matrix[n - 1, n - 2] = 2
        a = np.diag(c_matrix, k=-1)
        b = np.diag(c_matrix, k=-0)
        c = np.diag(c_matrix, k=1)

        p_vec = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
        p_vec[0] = points[0] + 2 * points[1]
        p_vec[n - 1] = 8 * points[n - 1] + points[n]
        p_vec = np.array(p_vec)
        x_p = p_vec[:, 0]
        y_p = p_vec[:, 1]

        x_solve = self.solver(a, b, c, x_p)
        y_solve = self.solver(a, b, c, y_p)

        A = np.array((x_solve, y_solve)).T
        B = [0] * n

        for i in range(n - 1):
            B[i] = 2 * points[i + 1] - A[i + 1]
        B[n - 1] = (A[n - 1] + points[n]) / 2

        return A, B

    def calc_spline(self, a, b, c, d):
        return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t,2) * c + np.power(t, 3) * d

    def create_polydict(self, points):
        A, B = self.calc_coef(points)
        return {(points[i][0], points[i][1]): self.calc_spline(points[i], A[i], B[i], points[i + 1]) for i in range(len(points) - 1)}

    def find_func(self, my_dict, x):
        keys = list(my_dict.keys())
        for i in range(len(keys)-1, -1, -1):
            if keys[i][0] <= x:
                func = my_dict[keys[i]]
                normalized_x = (x - self.points[i][0]) / (self.points[i + 1][0] - self.points[i][0])
                return func(normalized_x)[1]

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        if n == 1:
            return lambda x: f(x)
        x_x = np.linspace(a, b, n, endpoint=True)
        #y_y = f(x_x)##TODO: check
        y_y = np.array([f(x) for x in x_x])
        points = np.array([x_x, y_y])
        self.points = points.T
        func_dict = self.create_polydict(self.points)


        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        # replace this line with your solution to pass the second test

        result = lambda x: self.find_func(func_dict, x)

        return result


##########################################################################


# import unittest
# from functionUtils import *
# from tqdm import tqdm
#
#
# class TestAssignment1(unittest.TestCase):
#
#     def test_with_poly(self):
#         T = time.time()
#
#         ass1 = Assignment1()
#         mean_err = 0
#
#         d = 30
#         for i in tqdm(range(100)):
#             a = np.random.randn(d)
#
#             f = np.poly1d(a)
#
#             ff = ass1.interpolate(f, -10, 10, 100)
#
#             xs = np.random.random(200)
#             err = 0
#             for x in xs:
#                 yy = ff(x)
#                 y = f(x)
#                 err += abs(y - yy)
#
#             err = err / 200
#             mean_err += err
#         mean_err = mean_err / 100
#
#         T = time.time() - T
#         print(T)
#         print(mean_err)
#
#     def test_with_poly_restrict(self):
#         ass1 = Assignment1()
#         a = np.random.randn(5)
#         f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
#         ff = ass1.interpolate(f, -10, 10, 10)
#         xs = np.random.random(20)
#         for x in xs:
#             yy = ff(x)

#
# if __name__ == "__main__":
#     unittest.main()
