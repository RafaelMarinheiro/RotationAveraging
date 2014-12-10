#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-11-22 17:52:15
# @Last Modified by:   marinheiro
# @Last Modified time: 2014-12-09 01:14:31


import numpy
import numpy.random
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import unittest
# import compare
import logging
import rotation_averaging
import scipy.io

def main():
	logging.basicConfig(level=logging.INFO)

	a = scipy.io.loadmat("data/notredame/Notredame.mat")

	I = a['I'].transpose()
	Rgt = a['Rgt']
	RR = a['RR']
	# print a

	testi = rotation_averaging.util.fix_matrix(numpy.array([[1.0, 0.1, 0.0],[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
	# print testi
	# print scipy.linalg.norm(numpy.identity(3) - testi.dot(testi.transpose()))


	indices = [(I[i, 0]-1, I[i, 1]-1) for i in range(I.shape[0])]
	global_rotations = [rotation_averaging.util.fix_matrix(Rgt[:,:,i]) for i in range(Rgt.shape[2])]
	relative_rotations = [rotation_averaging.util.fix_matrix(RR[:,:,i]) for i in range(RR.shape[2])]

	max_err = 0.0
	for mat in global_rotations:
		max_err = max(max_err, scipy.linalg.norm(numpy.identity(3) - mat.dot(mat.transpose())))

	print max_err

	max_err = 0.0
	for mat in relative_rotations:
		max_err = max(max_err, scipy.linalg.norm(numpy.identity(3) - mat.dot(mat.transpose())))

	print max_err
	print global_rotations[0]
	print rotation_averaging.so3.matrix_to_axis_angle(global_rotations[0])
	print rotation_averaging.so3.matrix_to_axis_angle(relative_rotations[0])
	# graphi = graph.generate_random_so3_graph(200, completeness=0.5, noise=0.2)
	# global_rotations = graphi[0]
	# relative_rotations = graphi[1]
	# indices = graphi[2]

	rotation_averaging.compare.compare_global_rotation_to_graph(global_rotations, relative_rotations, indices, plot=True)
	initial_guess = rotation_averaging.graph.compute_initial_guess(len(global_rotations), relative_rotations, indices)

	max_err = 0.0
	for mat in initial_guess:
		max_err = max(max_err, scipy.linalg.norm(numpy.identity(3) - mat.dot(mat.transpose())))

	print max_err

	rotation_averaging.compare.compare_global_rotation_to_graph(initial_guess, relative_rotations, indices, plot=True)
	# print initial_guess[0].shape, len(initial_guess)
	# solution = rotation_averaging.algorithms.L1RA(len(global_rotations), relative_rotations, indices, initial_guess)
	# rotation_averaging.compare.compare_global_rotation_to_graph(solution, relative_rotations, indices, plot=True)
	solution = rotation_averaging.algorithms.IRLS(len(global_rotations), relative_rotations, indices, initial_guess)
	rotation_averaging.compare.compare_global_rotation_to_graph(solution, relative_rotations, indices, plot=True)

if __name__ == '__main__':
	main()