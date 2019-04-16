import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

class layer(object):
        
	
	def __init__(self, in_depth, out_depth, N_clusters, name):
		
		self.name = name
		
		if 'conv' in name:
			self.w = tf.Variable(tf.random.normal([3, 3, in_depth, out_depth], stddev=0.1))
		elif 'fc' in name:
			self.w = tf.Variable(tf.random.normal([in_depth, out_depth], stddev=0.1))
			
		self.w_var = tf.placeholder(tf.float32, self.w.shape) 
		self.assign_w = tf.assign(self.w, self.w_var)
		self.num_total_weights = np.prod(self.w.shape)
		
		# mask for pruning, 1: valid weights, 0: pruned weights
		self.pruning_mask_data = np.ones(self.w.shape, dtype=np.float32)
		self.N_clusters = N_clusters # for quantization
		self.clusters_masks = []


        # Forward pass of a layer of network
	def forward(self,x):
		if 'conv' in self.name:
			return tf.nn.conv2d(x, self.w, strides=[1, 1, 1, 1], padding='SAME')
		elif 'fc' in self.name:		
			return tf.matmul(x, self.w)


        # save histogram of weights				
	def save_weights_histogram(self, sess, directory, iteration):
		w_data = sess.run(self.w).reshape(-1)	
		valid_w_data = [x for x in w_data if x!=0.0]
		plt.grid(True)
		plt.hist(valid_w_data, 100, color='0.4')
		plt.gca().set_xlim([-0.4, 0.4])
		plt.savefig(directory + '/' + self.name + '-' + str(iteration), dpi=100)
		plt.gcf().clear()


        # save weights
	def save_weights(self, sess, directory):
		w_data = sess.run(self.w)		
		np.save(directory + '/' + self.name + '-weights', w_data)
		np.save(directory + '/' + self.name + '-prune-mask', self.pruning_mask_data)
		


	# prune weights
	def prune_weights_ElementWise(self, sess, threshold):
                w_data = sess.run(self.w)
                self.pruning_mask_data = (np.abs(w_data) >= threshold).astype(np.float32)
                print ('\nlayer:', self.name)
                print ('remaining weights:', int(np.sum(self.pruning_mask_data)))
                print ('total weights:', self.num_total_weights)
                print ('compression rate: %', 100*(int(np.sum(self.pruning_mask_data))/int(self.num_total_weights)))
                sess.run(self.assign_w, feed_dict={self.w_var: self.pruning_mask_data*w_data})


        # prune weights gradient
	def prune_weights_gradient_ElementWise(self, grad):
                return grad * self.pruning_mask_data

	
	# update weights gradient
	def prune_weights_update_ElementWise(self, sess):
                w_data = sess.run(self.w)
                sess.run(self.assign_w, feed_dict={self.w_var: self.pruning_mask_data*w_data})
			



	# quantize weights based on K-Means 
	def quantize_weights_KMeans(self, sess):
		w_data = sess.run(self.w)
		max_val = np.max(w_data)
		min_val = np.min(w_data)
		print('\n', w_data.shape, max_val, min_val)
		# linearly initialize centroids between max and min
		self.centroids = np.linspace(min_val, max_val, self.N_clusters)
		w_data = np.expand_dims(w_data, 0)
		centroids_prev = np.copy(self.centroids)
		for i in range(20):  # number of iterations in K-Means
			if 'conv' in self.name:
				distances = np.abs(w_data - np.reshape(self.centroids,(-1, 1, 1, 1, 1)))
				distances = np.transpose(distances, (1,2,3,4,0))
			elif 'fc' in self.name:
				distances = np.abs(w_data - np.reshape(self.centroids,(-1, 1, 1)))
				distances = np.transpose(distances, (1,2,0))				
			classes = np.argmin(distances, axis=-1)
			#self.clusters_masks = []
			for i in range(self.N_clusters):
				cluster_mask = (classes == i).astype(np.float32) * self.pruning_mask_data
				self.clusters_masks.append(cluster_mask) 
				num_weights_assigned = np.sum(cluster_mask)
				if num_weights_assigned!=0:
					self.centroids[i] = np.sum(cluster_mask * w_data) / num_weights_assigned
				else: 
					pass
			if np.array_equal(centroids_prev, self.centroids):
				break
			centroids_prev = np.copy(self.centroids)
		self.quantize_weights_update_KMeans(sess)
		#print('>>> ', len(self.clusters_masks))
		


	# quantize gradient				
	def group_and_reduce_gradient_KMeans(self, grad):
		grad_out = np.zeros(self.w.shape, dtype=np.float32)
		#print('\n\n >>>>>>>>>> ',len(self.clusters_masks))
		for i in range(self.N_clusters):
			cluster_mask = self.clusters_masks[i]
			centroid_grad = np.sum(grad * cluster_mask)
			grad_out = grad_out + cluster_mask * centroid_grad
		return grad_out

		
	# update centroids	
	def quantize_centroids_update_KMeans(self, sess):
		w_data = sess.run(self.w)
		for i in range(self.N_clusters):
			cluster_mask = self.clusters_masks[i]
			cluster_count = np.sum(cluster_mask)
			if cluster_count!=0:
				self.centroids[i] = np.sum(cluster_mask * w_data) / cluster_count
			else: 
				pass

				
	# update weights
	def quantize_weights_update_KMeans(self, sess):
		w_data_updated = np.zeros(self.w.shape, dtype=np.float32)
		for i in range(self.N_clusters):
			cluster_mask = self.clusters_masks[i]
			centroid = self.centroids[i]
			w_data_updated = w_data_updated + cluster_mask * centroid
		sess.run(self.assign_w, feed_dict={self.w_var: self.pruning_mask_data * w_data_updated})







	
