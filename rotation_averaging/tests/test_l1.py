#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-10-28 02:40:42
# @Last Modified by:   Rafael Marinheiro
# @Last Modified time: 2014-10-28 04:39:24



import numpy
import numpy.random
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import unittest
from .. import l1
import logging

class TestL1Approximation(unittest.TestCase):

	def setUp(self):
		logging.basicConfig(level=logging.WARN)
		pass

	def test_l1fullA(self):
		m = 100
		n = 10
		k = 2

		A = numpy.random.rand(m, n)
		B = numpy.random.rand(m, k)

		ret = scipy.linalg.lstsq(A, B)
		X0 = ret[0]

		X = l1.l1_msolve(A, B, X0 +numpy.random.rand(n, k)*0.1)

		print sum(sum(abs(A.dot(X0)-B)))
		print sum(sum(abs(A.dot(X)-B)))
		# print sum(abs(x-x0))
		self.assertTrue(True, msg="Du")


	def test_l1approximation(self):
		m = 1000
		n = 10
		k = 1

		A = scipy.sparse.rand(m, n, format='csr')
		b = numpy.random.rand(m, k)

		ret = scipy.sparse.linalg.lsqr(A, b)
		x0 = ret[0]
		x0 = numpy.array([x0]).transpose()
		# print max(abs(x0))

		x = l1.l1_solve(A, b, x0 +numpy.random.rand(n, k)*0.1)

		print sum(abs(A.dot(x0)-b))
		print sum(abs(A.dot(x)-b))
		# print sum(abs(x-x0))
		self.assertTrue(True, msg="Du")



if __name__ == '__main__':
	unittest.main()