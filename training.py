import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys, os, shutil
from networks.vgg_network import *
from configs.config import *




if __name__ == "__main__":

        # Create histogram & weights directories  
        if os.path.exists(FLAGS.histograms_dir+FLAGS.model_name):
            shutil.rmtree(FLAGS.histograms_dir+FLAGS.model_name, ignore_errors=True)
        os.makedirs(FLAGS.histograms_dir+FLAGS.model_name)
        if os.path.exists(FLAGS.weights_dir+FLAGS.model_name):
            shutil.rmtree(FLAGS.weights_dir+FLAGS.model_name, ignore_errors=True)
        os.makedirs(FLAGS.weights_dir+FLAGS.model_name)

        # Define training dataset
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # Define network
        oNetwork = vgg_network()


        # Training, Pruning, Quantization
        x_PH = tf.placeholder(tf.float32, [None, 28, 28, 1])
        logits = oNetwork.forward_pass(x_PH)
        preds = tf.nn.softmax(logits)
        labels = tf.placeholder(tf.float32, [None, 10])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))

        optimizer = tf.train.AdamOptimizer(1e-4)

        layer_w=[]
        for layerN in range(len(oNetwork.layer)):
                 layer_w.append(oNetwork.layer[layerN].w)
                        
        gradients_vars = optimizer.compute_gradients(loss, layer_w)
        grads = [grad for grad, var in gradients_vars]
        train_step = optimizer.apply_gradients(gradients_vars)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        
        iters = []
        iters_acc = []

        for i in range(FLAGS.all_steps):

                batch_x, batch_y = mnist.train.next_batch(50)
                batch_x = np.reshape(batch_x,(-1, 28, 28,1))
                feed_dict={x_PH: batch_x, labels: batch_y}
		
                # network training 
                if i < FLAGS.full_training_steps:
                        sess.run(train_step, feed_dict=feed_dict)
                                                         
                # pruning & fine tunning
                elif  i < FLAGS.pruning_fine_tunning_steps:
                        # prune weights
                        if i%FLAGS.pruning_fine_tunning_step==0:
                                print ('iter:', i, 'prune weights')
                                oNetwork.prune_weights_ElementWise(sess, threshold=0.1)
                        grads_data = sess.run(grads, feed_dict={x_PH: batch_x, labels: batch_y})
                        feed_dict = {}
                        # prune weights gradient
                        feed_dict = oNetwork.prune_weights_gradient_ElementWise(feed_dict, grads, grads_data)
                        sess.run(train_step, feed_dict=feed_dict)
                        # update weights
                        oNetwork.prune_weights_update_ElementWise(sess)
                                 
                # quantization & fine tunning
                else:
                        # quantizate weights
                        if i==FLAGS.quantization_fine_tunning_step:
                                print ('iter:', i, "quantize weights")
                                oNetwork.quantize_weights_KMeans(sess)
                        grads_data = sess.run(grads, feed_dict={x_PH: batch_x, labels: batch_y})
                        feed_dict = {}
                        # update gradients
                        feed_dict = oNetwork.group_and_reduce_gradient_KMeans(feed_dict, grads, grads_data)
                        sess.run(train_step, feed_dict=feed_dict)
                        # update centroids
                        oNetwork.quantize_centroids_update_KMeans(sess)
                        oNetwork.quantize_weights_update_KMeans(sess)
  		
                # evaluation & measure accuracy 
                if i%10 == 0:
                        batches_acc = []
                        for j in range(10):
                                batch_x, batch_y = mnist.test.next_batch(1000)
                                batch_x = np.reshape(batch_x,(-1, 28, 28,1))	
                                batch_acc = sess.run(accuracy,feed_dict={x_PH: batch_x, labels: batch_y})
                                batches_acc.append(batch_acc)
                        acc = np.mean(batches_acc)
                        iters.append(i)
                        iters_acc.append(acc)				
                        print ('iter:', i, 'test accuracy:', acc)
                        oNetwork.save_weights_histogram(sess, FLAGS.histograms_dir+FLAGS.model_name, i)

        oNetwork.save_weights(sess, FLAGS.weights_dir+FLAGS.model_name)
        
        plt.figure(figsize=(10, 4))
        plt.ylabel('accuracy', fontsize=12)
        plt.xlabel('iteration', fontsize=12)
        plt.grid(True)
        plt.plot(iters, iters_acc, color='0.4')
        plt.savefig('./train_acc', dpi=1200)
        print ('Training finished')




