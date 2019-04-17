# Deep Compression: Compression of Deep Neural Network

This is a tensorFlow implementation of Deep Compression [paper](https://arxiv.org/abs/1510.00149) which is inspired by the previous tensorflow and pytorch implementations.

This implementation is designed with the following goals:
- Clear Pipeline: it has pipeline of network compression including pruning and quantization.
- Modularity: This code has been implemented for element-wise pruning and clustering-based quantization. It is modular and easy to be expanded for other types of pruning and quantization methods.
- Extendable: Currently this code has been implemented for VGG16, but it can be extended for other networks.
- To be deployed on Embedded Systems 

This implementation includes pruning & quantization modules in the Deep Compression method:

- Pruning
- Quantization

### Deep Neural Network
I implemented pruning and quantization for convolutional and fully connected layers of VGG16 network. I applied VGG16 network for classification on MNIST database. 

### Network Compression Steps
The network compression has the following steps:

### 1. Network Training 
In this step, I train the whole network (e.g. VGG) for a number iterations (1000 iterations) for classification.

### 2. Pruning
There are different types of prunings. In this code, I implemented element-wise pruning in which the weights of all layers are prunned if they are smaller than a threshold.

### 3. Pruning & fine tunning
After pruning, I train the network for a number of iterations (1000 iterations) while making the prunned values to zero.  

### 4. Quantization
There are different types of quantizations. In this code, I implemented clutering-based quantization based on K-means clustering.
For VGG, I used 5 clusters to get high performance. With less number of clusters, there is drop in the accuracy.

### 5. Quantization & fine tunning
After quantization, I fine tunned the network with the quantized values for a number of iterations (1000 iterations).
For quantization, the clutering is aplied to each layer (connvolutional and fully connected layers) seperatly.


### Compration Rate Calculation:
I calculated the compression rate for VGG:
















