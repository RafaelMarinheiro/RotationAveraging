#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-11-22 17:52:15
# @Last Modified by:   Rafael Marinheiro
# @Last Modified time: 2014-11-22 17:54:31


import numpy
import numpy.random
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import unittest
# import compare
import logging
import rotation_averaging
import rotation_averaging.graph as graph
import rotation_averaging.compare as compare
# impo

def main():
	logging.basicConfig(level=logging.INFO)
	graphi = graph.generate_random_so3_graph(200, completeness=0.5, noise=0.2)
	global_rotations = graphi[0]
	relative_rotations = graphi[1]
	indices = graphi[2]

	# compare.compare_global_rotation_to_graph(global_rotations, relative_rotations, indices, plot=True)
	initial_guess = graph.compute_initial_guess(len(global_rotations), relative_rotations, indices)
	compare.compare_global_rotation_to_graph(initial_guess, relative_rotations, indices, plot=True)
	solution = rotation_averaging.L1RA(len(global_rotations), relative_rotations, indices, initial_guess)
	compare.compare_global_rotation_to_graph(solution, relative_rotations, indices, plot=True)

if __name__ == '__main__':
	main()