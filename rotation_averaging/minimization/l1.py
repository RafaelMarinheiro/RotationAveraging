#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-10-27 04:32:17
# @Last Modified by:   marinheiro
# @Last Modified time: 2014-12-08 21:48:35

import logging
import numpy
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg

def l1_solve(A, b, x0, tol=1e-3, max_iterations=50):
	"""Computes the solution :math:`x^*` of the :math:`\\ell_1` approximation problem:

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
	alpha = 0.01
	beta = 0.5
	mu = 10.0

	N = len(x0)
	M = len(b)

	gradf0 = scipy.array([[0.0]*N + [1.0]*M]).transpose()

	x = x0
	Ax = A.dot(x)
	u = 0.95*abs(b - Ax) + 0.10*max(abs(b-Ax))

	fu1 = Ax - b - u
	fu2 = -Ax + b - u

	lamu1 = -1.0/fu1
	lamu2 = -1.0/fu2

	Atv = A.transpose().dot(lamu1 - lamu2)

	sdg = -(fu1.transpose().dot(lamu1) + fu2.transpose().dot(lamu2))
	tau = 2.0*mu*M/sdg

	rcent = numpy.vstack((-lamu1*fu1, -lamu2*fu2)) - 1.0/tau
	rdual = gradf0 + numpy.vstack((Atv, -lamu1-lamu2))
	resnorm = scipy.linalg.norm(numpy.vstack((rdual, rcent)))

	pditer = 0
	done = (sdg < tol) or (pditer >= max_iterations)
	
	while not done:
		pditer = pditer + 1

		w2 = -1.0 - (1.0/tau) * (1.0/fu1 + 1.0/fu2)

		sig1 = -(lamu1/fu1) - (lamu2/fu2)
		sig2 = (lamu1/fu1) - (lamu2/fu2)
		sigx = sig1 - ((sig2*sig2)/sig1)

		# ?

		w1 = (-1.0/tau) * (A.transpose().dot(-(1.0/fu1) + (1.0/fu2)))
		w1p = w1 - A.transpose().dot((sig2/sig1)*w2)
		
		# Slow solution
		H11p = A.transpose().dot(scipy.sparse.diags(sigx.transpose(), [0], shape=(M, M)).dot(A))
		dx = scipy.sparse.linalg.spsolve(H11p, w1p)
		dx = numpy.array([dx]).transpose()

		# Fast Solution
		# dx_temp = scipy.sparse.linalg.spsolve(A.transpose(), w1p)
		# dx  = scipy.sparse.linalg.spsolve(scipy.sparse.diags(sigx, 0).dot(A), dx_temp)

		# hcond = numpy.linalg.cond(H11p)
		hcond = 1
		if hcond < 1e-14:
			logging.warning("Matrix is ill-conditioned. Returning the previous iterate.")
			return x

		if numpy.isnan(numpy.min(dx)):
			logging.warning("NaNs were found. Returning the previous iterate.")
			return x

		Adx = A.dot(dx)

		du = (w2 - sig2*Adx)/sig1
		
		dlamu1 = -(lamu1/fu1)*(Adx-du) - lamu1 - (1.0/tau)*(1.0/fu1)
		dlamu2 =  (lamu2/fu2)*(Adx+du) - lamu2 - (1.0/tau)*(1.0/fu2)
		Atdv = A.transpose().dot(dlamu1 - dlamu2)

		# make sure that the step is feasible: keeps lamu1, lamu2 > 0 and fu1,fu2 < 0
		indl = numpy.where(dlamu1 < 0)
		indu = numpy.where(dlamu2 < 0)

		s = min(numpy.hstack((1.0, -lamu1[indl]/dlamu1[indl], -lamu2[indu]/dlamu2[indu])))

		indl = numpy.where((Adx-du) > 0)
		indu = numpy.where((-Adx-du) > 0)
		s = 0.99*min(numpy.hstack((s, -fu1[indl]/(Adx[indl] - du[indl]), -fu2[indu]/(-Adx[indu] - du[indu]))))

		# backtrack

		suffdec = False
		backiter = 0

		while not suffdec:
			if backiter > 32:
				logging.warning("Stuck backtracking, returning last iterate.  (See Section 4 of notes for more information.)")
				return x

			xp = x + s*dx
			up = u + s*du

			Axp = Ax + s*Adx
			Atvp = Atv + s*Atdv

			lamu1p = lamu1 + s*dlamu1
			lamu2p = lamu2 + s*dlamu2

			fu1p = Axp - b - up
			fu2p = -Axp + b - up

			rdp = gradf0 + numpy.vstack((Atvp, -lamu1p - lamu2p))
			rcp = numpy.vstack((-lamu1p*fu1p, -lamu2p*fu2p)) - (1.0/tau)

			suffdec = (scipy.linalg.norm(numpy.vstack((rdp, rcp))) <= (1-alpha*s)*resnorm)

			s = beta*s
			backiter = backiter + 1

		
		# next iteration
		x = xp
		u = up
		Ax = Axp
		Atv = Atvp
		lamu1 = lamu1p
		lamu2 = lamu2p
		fu1 = fu1p
		fu2 = fu2p

		# surrogate duality gap
		sdg = -(fu1.transpose().dot(lamu1) + fu2.transpose().dot(lamu2))
		tau = mu*2.0*M/sdg
		rcent = numpy.vstack((-lamu1*fu1, -lamu2*fu2)) - 1.0/tau
		rdual = rdp
		resnorm = scipy.linalg.norm(numpy.vstack((rdual, rcent)))

		done = (sdg < tol) or (pditer >= max_iterations)

		logging.info("Iteration = %d, tau = %8.3f, Primal = %8.3f, PDGap = %8.3f, Dual res = %8.3f" % (pditer, tau, sum(u), sdg, scipy.linalg.norm(rdual)))
		logging.info("H11p condition number = %8.3f" % hcond)
	
	return x

def l1_msolve(A, B, X0, tol=1e-3, max_iterations=50):
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
		Xstar[..., i] = l1_solve(A, b, x0, tol=tol, max_iterations=max_iterations).transpose()

	return Xstar





















