from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from scipy.io import loadmat
from math import cos, sin


class CoarseMorphabelModel(object):
	def __init__(self, model_path,):
		super(CoarseMorphabelModel, self).__init__()
		if not os.path.isfile(model_path):
			print('please set [BFM_COARSE.mat]')
			exit()

		model = loadmat(model_path)
		self.shapePC = model['shapePC']
		self.shapeEV = model['shapeEV']
		self.shapeMU = model['shapeMU']
		self.texPC = model['texPC']
		self.texEV = model['texEV']
		self.texMU = model['texMU']
		self.expPC = model['expPC']
		self.expEV = model['expEV']
		self.expMU = model['expMU']
		self.triangles = model['tri']
		self.full_triangles = np.vstack((model['tri'], model['tri_mouth']))
		self.kpt_ind = model['kpt_ind'][0]

		self.nver = self.shapePC.shape[0] / 3
		self.ntri = self.triangles.shape[0]
		self.n_shape_para = self.shapePC.shape[1]
		self.n_exp_para = self.expPC.shape[1]
		self.n_tex_para = self.texMU.shape[1]

	def generate_vertices(self, shape_para, exp_para):
		"""
		Args:
			shape_para: (n_shape_para, 1)
			exp_para: (n_exp_para, 1)
		Returns:
			vertices: (nver, 3)
		"""
		vertices = self.shapeMU + self.expMU + self.shapePC.dot(shape_para) + self.expPC.dot(exp_para)
		vertices = np.reshape(vertices, [int(3), int(len(vertices) / 3)], 'F').T

		return vertices

	def transform_coarse(self, vertices, s, angles, t2d):
		x, y, z = angles[0], angles[1], angles[2]

		# x
		Rx = np.array([[1,       0,      0],
					   [0,  cos(x), sin(x)],
					   [0, -sin(x), cos(x)]])
		# y
		Ry = np.array([[cos(y), 0, -sin(y)],
					   [     0, 1,       0],
					   [sin(y), 0, cos(y)]])
		# z
		Rz = np.array([[ cos(z), sin(z), 0],
					   [-sin(z), cos(z), 0],
					   [      0,      0, 1]])
		R = Rx.dot(Ry).dot(Rz)
		R = R.astype(np.float32)

		t3d = np.hstack([t2d, np.zeros(1)])

		transform_vertex = s * vertices.dot(R.T) + t3d[np.newaxis, :]

		return transform_vertex

	# save 3D face to obj file
	def save_obj(self, path, v, f):
		f = f.copy()
		f += 1
		with open(path, 'w') as file:
			for i in range(len(v)):
				file.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))

			file.write('\n')

			for i in range(len(f)):
				file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

		file.close()

