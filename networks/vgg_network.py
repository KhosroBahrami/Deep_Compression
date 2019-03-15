import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from networks.layers import *


class vgg_network(object):

	
	def __init__(self):
                
                # VGG network definition
                # Convolutional layers: 1-10
                self.layer=[]
                self.layer.append(layer(1, 16, N_clusters=5, name='conv1'))
                self.layer.append(layer(16, 16, N_clusters=5, name='conv2'))

                self.layer.append(layer(16, 32, N_clusters=5, name='conv3'))
                self.layer.append(layer(32, 32, N_clusters=5, name='conv4'))

                self.layer.append(layer(32, 64, N_clusters=5, name='conv5'))
                self.layer.append(layer(64, 64, N_clusters=5, name='conv6'))

                self.layer.append(layer(64, 128, N_clusters=5, name='conv7'))
                self.layer.append(layer(128, 128, N_clusters=5, name='conv8'))

                self.layer.append(layer(128, 128, N_clusters=5, name='conv9'))
                self.layer.append(layer(128, 128, N_clusters=5, name='conv10'))

                # Fully connected layers: 11-14
                self.layer.append(layer(1*1*128, 1096, N_clusters=5, name='fc1'))
                self.layer.append(layer(1096, 1096, N_clusters=5, name='fc2'))
                self.layer.append(layer(1096, 1000, N_clusters=5, name='fc3'))
                self.layer.append(layer(1000, 10, N_clusters=5, name='fc4'))


        # Forward pass of the network
	def forward_pass(self, x):

                print(x.shape)
                x = tf.nn.relu(self.layer[0].forward(x))
                print(x.shape)
                x = tf.nn.relu(self.layer[1].forward(x))
                print(x.shape)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2, padding="same")
                print(x.shape)
                
                x = tf.nn.relu(self.layer[2].forward(x))
                print(x.shape)
                x = tf.nn.relu(self.layer[3].forward(x))
                print(x.shape)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2, padding="same")
                print(x.shape)
                
                x = tf.nn.relu(self.layer[4].forward(x))
                print(x.shape)
                x = tf.nn.relu(self.layer[5].forward(x))
                print(x.shape)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2, padding="same")
                print(x.shape)
                
                x = tf.nn.relu(self.layer[6].forward(x))
                print(x.shape)
                x = tf.nn.relu(self.layer[7].forward(x))
                print(x.shape)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2, padding="same")
                print(x.shape)
                
                x = tf.nn.relu(self.layer[8].forward(x))
                print(x.shape)
                x = tf.nn.relu(self.layer[9].forward(x))
                print(x.shape)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2, padding="same")
                print(x.shape)
                
                x = tf.reshape(x, (-1, int(np.product(x.shape[1:])))) 
                print(x.shape)
                x = tf.nn.relu(self.layer[10].forward(x))
                print(x.shape)
                x = tf.nn.relu(self.layer[11].forward(x))
                print(x.shape)
                x = tf.nn.relu(self.layer[12].forward(x))
                print(x.shape)
                logits = self.layer[13].forward(x)
                print('logits: ',logits.shape)
                return logits


        # save histogram of weights				
	def save_weights_histogram(self, sess, directory, iteration):
		for layerN in range(len(self.layer)):
                        self.layer[layerN].save_weights_histogram(sess, directory, iteration)

                        
        # save weights
	def save_weights(self, sess, directory):
		for layerN in range(len(self.layer)):
                        self.layer[N].save_weights(sess, directory)
	
	
	# prune weights
	def prune_weights_ElementWise(self, sess, threshold):
		for layerN in range(len(self.layer)):
                        self.layer[layerN].prune_weights_ElementWise(sess, threshold=0.1)


        # prune weights gradient
	def prune_weights_gradient_ElementWise(self, feed_dict, grads, grads_data):
                index=0
                for grad, grad_data in zip(grads, grads_data):
                        feed_dict[grad] = self.layer[index].prune_weights_gradient_ElementWise(grad_data)
                        index+=1
                return feed_dict

	
	# update weights gradient
	def prune_weights_update_ElementWise(self, sess):
		for layerN in range(len(self.layer)):
                        self.layer[layerN].prune_weights_update_ElementWise(sess)

			
	# quantize weights
	def quantize_weights_KMeans(self, sess):
                for layerN in range(len(self.layer)):
                        self.layer[layerN].quantize_weights_KMeans(sess)


	# quantize gradient				
	def group_and_reduce_gradient_KMeans(self, feed_dict, grads, grads_data):
                index=0
                for grad, grad_data in zip(grads, grads_data):
                        feed_dict[grad] = self.layer[index].group_and_reduce_gradient_KMeans(grad_data)
                        index+=1
                return feed_dict
	
		
	# update centroids	
	def quantize_centroids_update_KMeans(self, sess):
                for layerN in range(len(self.layer)):
                        self.layer[layerN].quantize_centroids_update_KMeans(sess)
		
				
	# update weights
	def quantize_weights_update_KMeans(self, sess):
                for layerN in range(len(self.layer)):
                        self.layer[layerN].quantize_weights_update_KMeans(sess)		
		




	
