#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Bryce Evans
# @Date:   2014-11-07
# @Last Modified by:   marinheiro
# @Last Modified time: 2014-11-08

import logging
import numpy
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import math

def ilrs_phi(e, sigma=5*math.pi/180):
	diagonal = []
	for el in e:
		diagonal.append(sigma*sigma/(el[0]*el[0] + sigma*sigma))

	return scipy.sparse.diags([diagonal], [0], format='csr')

def ilrs_solve(A, b, x0, tol=1e-3, max_iterations=3):
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
	x_prev = x0*0.0 + 231231.0
	x = x0
	it = 0
	while scipy.linalg.norm(x-x_prev) > tol and it < max_iterations:
		logging.info("IRLS: Iteration %d of %d, Error: %f Tolerance: %f" % (it+1, max_iterations, scipy.linalg.norm(x-x_prev), tol))
		x_prev = x
		e = A.dot(x) - b
		if numpy.isnan(numpy.min(x)):
			logging.warning("NaNs were found. Returning the previous iterate.")
			return x

		phi_m = ilrs_phi(e)

		xt = scipy.sparse.linalg.lsqr(phi_m.dot(A),phi_m.dot(b), iter_lim=1000)
		xt = xt[0]
		xt = numpy.array([xt]).transpose()

		if numpy.isnan(numpy.min(xt)):
			logging.warning("NaNs were found. Returning the previous iterate.")
			return x
		
		x = xt
		it = it+1
	return x

def irls_msolve(A, B, X0, tol=1.0e-3, max_iterations=20):
	"""Returns the solution :math:`X^*` of the :math:`\\ell_1` approximation problem:

		:math:`X^* = \\underset{X}{\\arg\\min} \\quad \\left\\lVert AX - B\\right\\rVert _1`

	:param A: A full rank matrix. Optionally sparse.
	:type A: :math:`M\\times N` matrix

	:param B:
	:type B: :math:`M\\times K` matrix

	:param X0: An initial guess of the solution
	:type X0: :math:`N\\times K` matrix

	:param tol: Tolerance for primal-dual algorithm (algorithm terminates if the duality gap is less than pdtol)
	:param max_iterations: Maximum number of primal-dual iterations

	:returns: Xstar -- the solution to the approximation problem
	"""

	Xstar = numpy.zeros(X0.shape)

	for i in range(Xstar.shape[1]):
		b = numpy.array([B[..., i]]).transpose()
		x0 = numpy.array([X0[..., i]]).transpose()
		Xstar[..., i] = ilrs_solve(A, b, x0, max_iterations=max_iterations).transpose()

	return Xstar