#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-10-28 18:07:20
# @Last Modified by:   Rafael Marinheiro
# @Last Modified time: 2014-10-28 18:42:54

import scipy.sparse
from .. import so3

def create_matrix_from_indices(num_nodes, indices):
	i = []
	j = []
	val = []

	for line,ind in enumerate(indices):
		i.append(ind[0])
		j.append(line)
		val.append(-1)

		j.append(ind[0])
		j.append(line)
		val.append(1)

	A = scipy.sparse.coo_matrix((val, (i, j)), shape=(len(indices), num_nodes)).to_csr()

	return A

def compute_relative_log_matrix(global_rotations, relative_rotations, indices):
	ret = []

	for (line,ind) in enumerate(indices):
		i = ind[0]
		j = ind[1]
		deltaRot = global_rotations[j].tranpose().dot(relative_rotations[line].dot(global_rotations[i]))
		(n, theta) = so3.matrix_to_axis_angle(deltaRot)
		ret.append(so3.axis_angle_to_log(n, theta))

	return numpy.hstack(ret).transpose()

def update_global_rotation_from_log(global_rotations, log_matrix):
	for node in range(len(global_rotations)):
		n, theta = so3.log_to_axis_angle(log_matrix[node])
		global_rotations[node] = global_rotations[node].dot(so3.axis_angle_to_matrix(n, theta))

	return global_rotations