import numpy as np


class Cls_Dipconv(object):
	"""
	CLS_DIPCONV Class for unit dipole kernel convolution
	"""

	def __init__(self, imsize, ker):
		self.imsize = imsize
		self.ker = ker  # unit dipole kernel in k-space 'D'

	def ctranspose(self, obj):
		# empty function
		# (fDF)' = fDF
		return obj

	@ property
	def trans(self):
		return Cls_Dipconv(self.imsize, self.ker)

	def mtimes(self, x):
		x = np.reshape(x, self.imsize)
		y = np.fft.ifftn(self.ker * np.fft.fftn(x))
		return y
