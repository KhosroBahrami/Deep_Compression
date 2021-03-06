# Deep Compression: Compression of Deep Neural Network

This is a tensorFlow implementation of Deep Compression [paper](https://arxiv.org/abs/1510.00149) which is inspired by the previous tensorflow and pytorch implementations.

This implementation is designed with the following goals:
- Clear Pipeline: it has pipeline of network compression including pruning and quantization.
- Modularity: This code has been implemented for element-wise pruning and clustering-based quantization. It is modular and easy to be expanded for other types of pruning and quantization methods.
- Extendable: Currently this code has been implemented for VGG16, but it can be extended for other networks.
- To be deployed on Embedded Systems. 

This implementation includes pruning & quantization modules in the Deep Compression method:

- Pruning
- Quantization

### Deep Neural Network
I used pruning and quantization for convolutional and fully connected layers of VGG16 network. The VGG16 network was used for classification on MNIST database. 

### Network Compression Steps:
The network compression has the following steps:

### 1. Whole Network training 
In this step, the whole network (e.g. VGG) is trained for a number of iterations (1000 iterations) for classification.

### 2. Pruning
There are different types of prunings. In this code, element-wise pruning is used in which the weights of all layers are prunned if they are smaller than a threshold.

### 3. Pruning & fine tunning
After pruning, the network is trained for a number of iterations (1000 iterations) while making the prunned values to zero.  

### 4. Quantization
There are different types of quantizations. In this code, clustering-based quantization is used based on K-means clustering.
For VGG, 20 clusters is used to get high performance. With less number of clusters, there is drop in the accuracy.

### 5. Quantization & fine tunning
After quantization, the network is fine tunned with the quantized values for a number of iterations (1000 iterations).
For quantization, the clustering is applied to each layer (connvolutional and fully connected layers) separately.


### Compression rate calculation:
I calculated the compression rate of pruning & quantization for VGG in the following table.
For pruning, based on the ratio of prunned parameters & non-prunned parameters, compression rate of 3 is achieved for all layers with threshold 0.1.  

For quantization, the following equation is used to calculate the compression rate, 

compression rate = (n*b)/(n*log(k)+k*b)

where n is number of parameters, k is number of clusters and b is number of bits for each parameter (32 bits for float). 

Totally, using pruning & quantization, the size of VGG can be reduced 18 times. 


| Layer  | Filter Size       | Feature size |# of Parameters | Pruning | Quantization | # of bits |
| :---:  |  :---:            |  :---:       | :---:          | :---:   | :---: |  :---: |
| Conv1_1| (3 * 3 * 3)*16    | 28 * 28 * 16 | 432        |  3  |   4   |    5   |
| Conv1_1| (3 * 3 * 16)*16   | 28 * 28 * 16 | 2304       |  3  |   6   |    5   |
| Pool1  |                   | 14 * 14 * 16 |  -         |   - |       |    -   |
| Conv2_2| (3 * 3 * 16)*32   | 14 * 14 * 32 | 4608       |  3  |   6   |    5   |
| Conv2_2| (3 * 3 * 32)*32   | 14 * 14 * 32 | 9216       |  3  |   6   |    5   |
| Pool2  |                   | 7 * 7 * 32   |  -         |   - |       |    -   |
| Conv3_1| (3 * 3 * 32)*64   | 7 * 7 * 64   | 18432      |  3  |   6   |    5   |
| Conv3_2| (3 * 3 * 64)*64   | 7 * 7 * 64   | 36864      |  3  |   6   |    5   |
| Pool3  |                   | 4 * 4 * 64   |   -        |   - |       |    -   |
| Conv4_1| (3 * 3 *64)*128   | 4 * 4 * 128  | 73728      |  3  |   6   |    5   |
| Conv4_2| (3 * 3 * 128)*128 | 4 * 4 * 128  | 147456     |  3  |   6   |    5   |
| Pool4  |                   | 2 * 2 * 128  |   -        |   - |       |    -   |
| Conv5_1| (3 * 3 * 128)*128 | 2 * 2 * 128  | 147456     |  3  |   6   |    5   |
| Conv5_2| (3 * 3 * 128)*128 | 2 * 2 * 128  | 147456     |  3  |   6   |    5   |
| Pool5  |                   | 1 * 1 * 128  |   -        |   - |       |    -   |
| FC1    | (1 * 1 * 128)*1096| 1096         | 140288     |  3  |   6   |    5   |
| FC2    | (1096)*1096       | 1096         | 1201216    |  3  |   6   |    5   |
| FC3    | (1096)*1000       | 1000         | 1096000    |  3  |   6   |    5   |
 











