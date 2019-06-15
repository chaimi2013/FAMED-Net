#!/usr/bin/python
import caffe
import numpy as np
import json

class spatialShuffleLayer(caffe.Layer):
	def setup(self, bottom, top):
		assert len(bottom) == 1,            'requires a single layer.bottom'
		assert bottom[0].data.ndim >= 3,    'requires image data'
		assert len(top) <= 2,               'requires a single layer.top'
	
		param = json.loads(self.param_str)
		self.kernel_size = param['kernel_size']
	
		
	def reshape(self, bottom, top):
		# difference is shape of inputs
		self.indexShuffle = np.zeros_like(bottom[0].data, dtype=np.int32)
        
		top[0].reshape(*bottom[0].data.shape)
		if len(top) > 1:
				top[1].reshape(*bottom[0].data.shape)
		hei = top[0].data.shape[2]
		wid = top[0].data.shape[3]
		assert (self.kernel_size > 0 & self.kernel_size <= np.minimum(hei,wid))

	def forward(self, bottom, top):
		r = self.kernel_size
		indexSub = range(r**2)
		
		batchSize = top[0].data.shape[0]
		channels = top[0].data.shape[1]
		hei = top[0].data.shape[2]
		wid = top[0].data.shape[3]
		
		top[0].data[...] = bottom[0].data
		for bb in range(batchSize):
			for cc in range(channels):
				for i in range(int(hei/r)):
					for j in range(int(wid/r)):
						np.random.seed(i+j)
						indexSubTmp = np.random.permutation(indexSub)
						top[0].data[bb, cc, i*r:(i+1)*r, j*r:(j+1)*r] = bottom[0].data[bb, cc, i*r:(i+1)*r, j*r:(j+1)*r].reshape(r**2,)[indexSubTmp].reshape([r,r])
						
						self.indexShuffle[bb, cc, i*r:(i+1)*r, j*r:(j+1)*r] = indexSubTmp.reshape([r,r])
						if len(top) > 1:
							top[1].data[bb, cc, i*r:(i+1)*r, j*r:(j+1)*r] = indexSubTmp.reshape([r,r])
							
	def backward(self, top, propogate_down, bottom):
		r = self.kernel_size
		
		batchSize = top[0].data.shape[0]
		channels = top[0].data.shape[1]
        	hei = top[0].data.shape[2]
        	wid = top[0].data.shape[3]
		
		for bb in range(batchSize):
			for cc in range(channels):
				for i in range(int(hei/r)):
					for j in range(int(wid/r)):
						indexSubTmp = self.indexShuffle[bb, cc, i*r:(i+1)*r, j*r:(j+1)*r].reshape(r**2,)
						
						bottom[0].diff[bb, cc, i*r:(i+1)*r, j*r:(j+1)*r] = top[0].diff[bb, cc, i*r:(i+1)*r, j*r:(j+1)*r].reshape(r**2,)[np.argsort(indexSubTmp)].reshape([r,r])

class spatialShuffleRecLayer(caffe.Layer):
	def setup(self, bottom, top):
		assert len(bottom) == 2,            'requires two layer.bottom'
		assert bottom[0].data.ndim >= 3,    'requires image data'
		assert len(top) == 1,               'requires one single layer.top'
	
		param = json.loads(self.param_str)
		self.kernel_size = param['kernel_size']
	
		
	def reshape(self, bottom, top):        
		top[0].reshape(*bottom[0].data.shape)
		hei = top[0].data.shape[2]
		wid = top[0].data.shape[3]
		assert (self.kernel_size > 0 & self.kernel_size <= np.minimum(hei,wid))

	def forward(self, bottom, top):
		r = self.kernel_size
		
		batchSize = top[0].data.shape[0]
		channels = top[0].data.shape[1]
		hei = top[0].data.shape[2]
		wid = top[0].data.shape[3]
		
		top[0].data[...] = bottom[0].data
		for bb in range(batchSize):
			for cc in range(channels):
				for i in range(int(hei/r)):
					for j in range(int(wid/r)):
						np.random.seed(i+j)
						indexSubTmp = bottom[1].data[bb, cc, i*r:(i+1)*r, j*r:(j+1)*r].reshape(r**2)
						top[0].data[bb, cc, i*r:(i+1)*r, j*r:(j+1)*r] = bottom[0].data[bb, cc, i*r:(i+1)*r, j*r:(j+1)*r].reshape(r**2,)[np.argsort(indexSubTmp)].reshape([r,r])
						
							
	def backward(self, top, propogate_down, bottom):
		pass
						
		

