"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
import time
import random
from functionUtils import AbstractShape
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ThreadPoolExecutor




import matplotlib.pyplot as plt




class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, points):
        self.points = points


        pass

    def contour(self, n: int):
        pass

    def area(self) -> np.float32:
        xs, ys = xy_sorting(self.points)
        area = 0.5*np.abs(np.dot(xs, np.roll(ys, 1))-np.dot(ys, np.roll(xs, 1)))

        return np.float32(area)


def xy_sorting(points):
    x, y = points.T
    x1 = np.mean(x)
    y1 = np.mean(y)
    calc = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    angles = np.where((y - y1) > 0, np.arccos((x - x1) / calc), 2 * np.pi - np.arccos((x - x1) / calc))
    m = np.argsort(angles)
    x_sorted = x[m]
    y_sorted = y[m]
    return x_sorted, y_sorted



class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        points = np.array(contour(400))
        xs = points[:, 0]
        ys = points[:, 1]
        n = len(points)
        sum1 = 0
        sum2 = 0
        for i in range(n-1):
            sum1 += xs[i]*ys[i+1]
            sum2 += xs[i+1]*ys[i]
        area = 0.5*abs(sum1 + xs[-1]*ys[0] - sum2 - xs[0]*ys[-1])



        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        return np.float32(area)

    
    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        n = 20000

        with ThreadPoolExecutor() as executor:
            samples = [executor.submit(sample) for _ in range(n)]
        points = np.array([s.result() for s in samples])
        #points = np.array([sample() for _ in range(n)])
        model1 = NearestNeighbors(n_neighbors=10).fit(points)
        distances = model1.kneighbors(points)[0]
        distances = np.sort(distances, axis=0)[:, 1]
        x = np.arange(0, n)
        data = np.vstack((x, distances)).T
        theta = np.arctan2(data[:, 1].max() - data[:, 1].min(),
                           data[:, 0].max() - data[:, 0].min())
        co = np.cos(theta)
        si = np.sin(theta)
        rotation_matrix = np.array(((co, -si), (si, co)))
        rotated_vector = data.dot(rotation_matrix)
        idx = np.where(rotated_vector == rotated_vector.min())[0][0]
        eps = data[idx][1]

        model = DBSCAN(eps=eps, min_samples=5).fit(points)
        points = model.components_

        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        # replace these lines with your solution
        result = MyShape(points)




        #x, y = sample()

        return result


##########################################################################

#
# import unittest
# from sampleFunctions import *
# from tqdm import tqdm
#
#
# class TestAssignment5(unittest.TestCase):
#
#     def test_return(self):
#         circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
#         ass5 = Assignment5()
#         T = time.time()
#         shape = ass5.fit_shape(sample=circ, maxtime=5)
#         T = time.time() - T
#         self.assertTrue(isinstance(shape, AbstractShape))
#         self.assertLessEqual(T, 5)
#
#     def test_delay(self):
#         circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
#
#         def sample():
#             time.sleep(7)
#             return circ()
#
#         ass5 = Assignment5()
#         T = time.time()
#         shape = ass5.fit_shape(sample=sample, maxtime=5)
#         T = time.time() - T
#         self.assertTrue(isinstance(shape, AbstractShape))
#         self.assertGreaterEqual(T, 5)
#
#     def test_circle_area(self):
#         circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
#         ass5 = Assignment5()
#         T = time.time()
#         shape = ass5.fit_shape(sample=circ, maxtime=30)
#         T = time.time() - T
#         a = shape.area()
#         plt.show()
#         self.assertLess(abs(a - np.pi), 0.01)
#         self.assertLessEqual(T, 32)
#
#
#     def test_bezier_fit(self):
#         circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
#         ass5 = Assignment5()
#         T = time.time()
#         shape = ass5.fit_shape(sample=circ, maxtime=30)
#         T = time.time() - T
#         a = shape.area()
#         self.assertLess(abs(a - np.pi), 0.01)
#         self.assertLessEqual(T, 32)
#
#     def test_area(self):
#         ass5 = Assignment5()
#         res = ass5.area(Circle(np.float32(1), np.float32(1), np.float32(3), np.float32(0)).contour)
#         exp = np.pi * 9
#         print(f'calculated area is: {res}')
#         print(f'relative error is: {abs(res - exp) / exp}')
#         self.assertLessEqual(abs(res - exp) / exp, 0.001)
#
# if __name__ == "__main__":
#     unittest.main()
