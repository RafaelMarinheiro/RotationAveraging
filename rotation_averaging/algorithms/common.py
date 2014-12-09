#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-10-28 18:07:20
# @Last Modified by:   marinheiro
# @Last Modified time: 2014-12-08 21:32:21

import scipy.sparse
import numpy
import rotation_averaging.so3 as so3

def create_matrix_from_indices(num_nodes, indices):
	i = []
	j = []
	val = []

	for line,ind in enumerate(indices):
		i.append(line)
		j.append(ind[0])
		val.append(-1)

		i.append(line)
		j.append(ind[1])
		val.append(1)

	A = scipy.sparse.coo_matrix((val, (i, j)), shape=(len(indices), num_nodes)).tocsr()

	return A

def compute_relative_log_matrix(global_rotations, relative_rotations, indices):
	ret = []

	for (line,ind) in enumerate(indices):
		i = ind[0]
		j = ind[1]
		deltaRot = global_rotations[j].transpose().dot(relative_rotations[line].dot(global_rotations[i]))
		(n, theta) = so3.matrix_to_axis_angle(deltaRot)
		ret.append(so3.axis_angle_to_log(n, theta))

	return numpy.hstack(ret).transpose()

def update_global_rotation_from_log(global_rotations, log_matrix):
	for node in range(len(global_rotations)):
		n, theta = so3.log_to_axis_angle(log_matrix[node])
		n = numpy.array([[n[0]], [n[1]], [n[2]]])
	
		global_rotations[node] = global_rotations[node].dot(so3.axis_angle_to_matrix(n, theta))

	return global_rotations