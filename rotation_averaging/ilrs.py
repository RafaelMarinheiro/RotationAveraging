#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Bryce Evans
# @Date:   2014-11-07
# @Last Modified by:   Bryce Evans
# @Last Modified time: 2014-11-08

import logging
import numpy
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg

def ilrs_solve(A, b, x0, tol=1e-3):
	""" Iteratively Reweighted Least Squares

	Computes the solution :math:`x^*` of the :math:`\\ell_1` approximation problem:

		:math:`x^* = \\underset{x}{\\arg\\min} \\quad \\left\\lVert Ax - b\\right\\rVert _1`

	This implementation is entirely based on *l1decode_pd.m*, part of l1 magic software (<http://users.ece.gatech.edu/~justin/l1magic/>)

	:param A: A full rank matrix. Optionally sparse.
	:type A: :math:`M\\times N` matrix

	:param b:
	:type b: :math:`M\\times 1` column-vector

	:param x0: An initial guess of the solution
	:type x0: :math:`N\\times 1` column-vector

	:param tol: Tolerance for primal-dual algorithm (algorithm terminates if the duality gap is less than pdtol)
	:param max_iterations: Maximum number of primal-dual iterations

	:returns: xstar -- the solution to the approximation problem
	"""

	x = x0
	x_prev = scipy.zeros(x)
	while scipy.linalg.norm(x-x_prev) < tol:
		x_prev = x
		e = A*x - b
		phi = phi(e)
		x = inverse(transpose(A)*phi*A)*A*phi*b
		
	return x
