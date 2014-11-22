#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-10-29 15:06:14
# @Last Modified by:   Rafael Marinheiro
# @Last Modified time: 2014-11-21 03:54:12

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-10-28 02:40:42
# @Last Modified by:   Rafael Marinheiro
# @Last Modified time: 2014-10-28 17:52:57



import numpy
import numpy.random
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import unittest
import graph
import compare
import logging
import rotation_averaging

class TestL1Approximation(unittest.TestCase):

	def setUp(self):
		logging.basicConfig(level=logging.INFO)
		graphi = graph.generate_random_so3_graph(200, completeness=0.5, noise=0.2)
		self.global_rotations = graphi[0]
		self.relative_rotations = graphi[1]
		self.indices = graphi[2]

	# def test_random_graph(self):
	# 	compare.compare_global_rotation_to_graph(self.global_rotations, self.relative_rotations, self.indices, plot=False)

	def test_L1RA(self):
		# compare.compare_global_rotation_to_graph(self.global_rotations, self.relative_rotations, self.indices, plot=True)
		initial_guess = graph.compute_initial_guess(len(self.global_rotations), self.relative_rotations, self.indices)
		compare.compare_global_rotation_to_graph(initial_guess, self.relative_rotations, self.indices, plot=True)
		solution = rotation_averaging.L1RA(len(self.global_rotations), self.relative_rotations, self.indices, initial_guess)
		compare.compare_global_rotation_to_graph(solution, self.relative_rotations, self.indices, plot=True)
		# compare.compare_global_rotations_to_baseline(solution, self.global_rotations, plot=True)

	# def test_l1fullA(self):
	# 	m = 100
	# 	n = 10
	# 	k = 2

	# 	A = numpy.random.rand(m, n)
	# 	B = numpy.random.rand(m, k)

	# 	ret = scipy.linalg.lstsq(A, B)
	# 	X0 = ret[0]

	# 	X = l1.l1_msolve(A, B, X0 +numpy.random.rand(n, k)*0.1)

	# 	# print sum(sum(abs(A.dot(X0)-B)))
	# 	# print sum(sum(abs(A.dot(X)-B)))

	# 	self.assertLessEqual(sum(sum(abs(A.dot(X)-B))), sum(sum(abs(A.dot(X0)-B))), "The solution is not better then the lstsq solution!")


	# def test_l1approximation(self):
	# 	m = 1000
	# 	n = 10
	# 	k = 1

	# 	A = scipy.sparse.rand(m, n, format='csr')
	# 	b = numpy.random.rand(m, k)

	# 	ret = scipy.sparse.linalg.lsqr(A, b)
	# 	x0 = ret[0]
	# 	x0 = numpy.array([x0]).transpose()
	# 	# print max(abs(x0))

	# 	x = l1.l1_solve(A, b, x0 +numpy.random.rand(n, k)*0.1)

	# 	# print sum(abs(A.dot(x0)-b))
	# 	# print sum(abs(A.dot(x)-b))
	# 	# print sum(abs(x-x0))
	# 	self.assertLessEqual(sum(abs(A.dot(x)-b)), sum(abs(A.dot(x0)-b)), "The solution is not better then the lstsq solution!")




if __name__ == '__main__':
	unittest.main()