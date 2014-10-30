#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-10-29 14:42:12
# @Last Modified by:   Rafael Marinheiro
# @Last Modified time: 2014-10-29 19:18:00

import networkx as nx

import so3
import numpy
import numpy.linalg
import math

import matplotlib.pyplot as plt

def generate_random_so3_graph(num_nodes, completeness=1.0, noise=None, n_outliers=0):
	global_rotation = []
	for i in range(num_nodes):
		n = numpy.random.rand(3, 1)
		n = n/numpy.linalg.norm(n)
		theta = numpy.random.rand(1, 1)

		global_rotation.append(so3.axis_angle_to_matrix(n, theta))

	graph = nx.Graph()
	graph.add_nodes_from(range(num_nodes))
	for i in range(num_nodes):
		for j in range(i+1,num_nodes):
			if i != j:
				mat = global_rotation[j].dot(global_rotation[i].transpose())
				(n, theta) = so3.matrix_to_axis_angle(mat)
				
				graph.add_edge(i, j, weight=theta)

	graph_edges = sorted(graph.edges(data=True), key=lambda edge: edge[2]['weight'])
	
	relative_rotations = []
	relative_edges = []

	tree = nx.minimum_spanning_tree(graph)

	target = math.ceil(completeness*num_nodes*(num_nodes-1)/2) - len(tree.edges())

	for edge in graph_edges:
		i = edge[0]
		j = edge[1]

		addEdge = False
		if tree.has_edge(i, j):
			addEdge = True
		else:
			if target > 0:
				addEdge = True
				target = target - 1
		
		if addEdge:
			relative_edges.append((i, j))
			mat = global_rotation[j].dot(global_rotation[i].transpose())

			if noise:
				w = noise*numpy.random.rand(3, 1)
				n, theta = so3.log_to_axis_angle(w)
				mat = so3.axis_angle_to_matrix(n, theta).dot(mat)


			relative_rotations.append(mat)

	return (global_rotation, relative_rotations, relative_edges)


	