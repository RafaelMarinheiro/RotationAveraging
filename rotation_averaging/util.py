#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Rafael Marinheiro
# @Date:   2014-12-08 20:30:52
# @Last Modified by:   marinheiro
# @Last Modified time: 2014-12-08 20:34:48

import scipy
import scipy.linalg

def fix_matrix(m):
	sq = scipy.linalg.inv(scipy.linalg.sqrtm(m.transpose().dot(m)))
	return m.dot(sq)