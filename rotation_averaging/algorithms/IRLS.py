#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: marinheiro
# @Date:   2014-12-08 20:47:41
# @Last Modified by:   marinheiro
# @Last Modified time: 2014-12-09 00:25:20


import common
import rotation_averaging.minimization.irls as irls

import numpy.linalg
import logging

def IRLS(num_nodes, rotations, indices, initial_guess, tol=0.001, max_iterations=100, change_threshold=0.001):
	eps = numpy.spacing(1)
	A = common.create_matrix_from_indices(num_nodes, indices)

	global_rotations = initial_guess

	done = False

	default_estimate = numpy.zeros((num_nodes, 3))

	wdelta = common.compute_relative_log_matrix(global_rotations, rotations, indices)

	if numpy.linalg.norm(wdelta) < tol:
		done = True

	it = 0
	ptol = -1000
	while not done:
		wglobal = irls.irls_msolve(A, wdelta, default_estimate, tol=eps)
		global_rotations = common.update_global_rotation_from_log(global_rotations, wglobal)

		wdelta = common.compute_relative_log_matrix(global_rotations, rotations, indices)

		norm_rel = numpy.linalg.norm(wdelta)
		norm_glob = numpy.linalg.norm(wglobal)

		it = it+1

		logging.info("Iteration number %d. Norm: %f"%(it, norm_rel))

		if norm_rel < tol:
			logging.info("Algorithm converged to the error bound.")
			done = True
		elif it >= max_iterations:
			logging.info("Maximum iterations reached")
			done = True
		else:
			# if norm_glob < change_threshold:
			# 	logging.info("Increasing the number of L1 steps")
			# 	num_l1_steps = 4*num_l1_steps
			# 	change_threshold = change_threshold/100
			if abs(ptol-norm_rel) < tol:
				logging.info("Algorithm converged.")
				done = True

		ptol = norm_rel

	return global_rotations
