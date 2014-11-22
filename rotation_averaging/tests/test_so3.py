#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-10-28 02:40:42
# @Last Modified by:   Rafael Marinheiro
# @Last Modified time: 2014-11-21 01:19:25



import numpy
import numpy.random
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import unittest
import so3
import logging

class TestL1Approximation(unittest.TestCase):

	def setUp(self):
		logging.basicConfig(level=logging.DEBUG)

	def test_zero(self):
		w = numpy.random.rand(3,1)
		w = w/scipy.linalg.norm(w)
		theta = 0

		(w1, theta1) = so3.matrix_to_axis_angle(so3.axis_angle_to_matrix(w, theta))

		self.assertAlmostEqual(theta, theta1, places=7, msg="Theta is not the same!")

	def test_quaternion(self):
		w = numpy.random.rand(3, 1)
		w = w/scipy.linalg.norm(w)
		theta = numpy.random.rand(1, 1)

		(w1, theta1) = so3.quaternion_to_axis_angle(so3.axis_angle_to_quaternion(w, theta))

		self.assertAlmostEqual(theta, theta1, msg="Theta is not the same!")
		self.assertAlmostEqual(scipy.linalg.norm(w-w1), 0, msg="W is not the same!")

	def test_invertability(self):
		w = numpy.random.rand(3, 1)
		w = w/scipy.linalg.norm(w)
		theta = numpy.random.rand(1, 1)

		(w1, theta1) = so3.matrix_to_axis_angle(so3.axis_angle_to_matrix(w, theta))

		# print("theta1: %f, theta2: %f"%(theta, theta1))
		# print w.transpose(), w1.transpose()

		self.assertAlmostEqual(theta, theta1, places=7, msg="Theta is not the same!")
		self.assertAlmostEqual(scipy.linalg.norm(w-w1), 0.0, places=7, msg="W is not the same!")


if __name__ == '__main__':
	unittest.main()