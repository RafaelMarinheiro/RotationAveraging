#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-10-29 16:04:21
# @Last Modified by:   Rafael Marinheiro
# @Last Modified time: 2014-10-29 19:15:13

import so3
import numpy
import math
import matplotlib.pyplot as plt

def compare_rotation_matrices(rot1, rot2):
	(n, theta) = so3.matrix_to_axis_angle(rot1.dot(rot2.transpose()))
	return theta

def compare_global_rotation_to_graph(global_rotations, relative_rotations, indices, plot=False):
	dif = []

	for (ind, edge) in enumerate(indices):
		i = edge[0]
		j = edge[1]

		mat = global_rotations[j].dot(global_rotations[i].transpose())
		theta = compare_rotation_matrices(mat, relative_rotations[ind])

		dif.append(theta)

	dif = numpy.array(dif)
	dif = dif*180/math.pi


	ret = (numpy.mean(dif), numpy.median(dif), numpy.std(dif))
	if plot:
		plt.xlabel(("Mean Angular Error (In Degrees): %f\n" % ret[0]) +
					("Median Angular Error (In Degrees): %f\n" % ret[1]) +
					("RMS Angular Error (In Degrees): %f" % ret[2]))

		plt.hist(dif, bins=180)
		plt.show()

	return ret
