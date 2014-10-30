#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-10-28 04:41:23
# @Last Modified by:   Rafael Marinheiro
# @Last Modified time: 2014-10-29 15:34:25

import numpy
import numpy.linalg
import scipy
import math

# Axis-Angle <-> Log Conversion

def axis_angle_to_log(n, theta):
	return n*theta

def log_to_axis_angle(w):
	theta = numpy.linalg.norm(w)
	n = numpy.zeros((3, 1))
	if theta != 0.0:
		n = w/theta

	return (n, theta)

# Quaternion <-> Axis-Angle conversion
def quaternion_to_axis_angle(quat):
	theta = 2.0*math.atan2(numpy.linalg.norm(quat[1:]), quat[0])
	n = numpy.zeros((3,1))
	if theta != 0.0:
		n = quat[1:]/math.sin(theta/2)

	return (n, theta)

def axis_angle_to_quaternion(n, theta):
	c = math.cos(theta/2)
	s = math.sin(theta/2)
	quat = numpy.zeros((4,1))

	quat[0] = c
	quat[1:] = n*s

	return quat

# Matrix <-> Quaternion conversion

def matrix_to_quaternion(rot):
	s = math.sqrt(numpy.trace(rot) + 1.0)/2
	quat = numpy.array([[s],
						[(rot[2, 1]-rot[1, 2])/(4*s)],
						[(rot[0, 2]-rot[2, 0])/(4*s)],
						[(rot[1, 0]-rot[0, 1])/(4*s)],
						])
	return quat

def quaternion_to_matrix(quat):
	qw = quat[0][0]
	qx = quat[1][0]
	qy = quat[2][0]
	qz = quat[3][0]

	rot = numpy.array([[1 - 2*qy*qy - 2*qz*qz, 	2*qx*qy - 2*qz*qw, 		2*qx*qz + 2*qy*qw],
					   [2*qx*qy + 2*qz*qw, 		1 - 2*qx*qx - 2*qz*qz, 	2*qy*qz - 2*qx*qw],
					   [2*qx*qz - 2*qy*qw, 		2*qy*qz + 2*qx*qw, 		1 - 2*qx*qx - 2*qy*qy]])
	return rot


# Matrix <-> Axis-Angle conversion
def matrix_to_axis_angle(rot):
	return quaternion_to_axis_angle(matrix_to_quaternion(rot))


def axis_angle_to_matrix(n, theta):
	return quaternion_to_matrix(axis_angle_to_quaternion(n, theta))


