# Deep Compression: Compression of Deep Neural Network

This is a tensorFlow implementation of Deep Compression [paper](https://arxiv.org/abs/1510.00149). 

This implementation implements 2 methods in the Deep Compression method:

- Pruning
- Quantization

### Pruning
There are different type of pruning. In this code, I implemented element-wise pruning in which the weights are prunned if they are smaller than a threshold.

### Quantization
There are different type of quantization. In this code, I implemented clutering based quantization.


### Deep Neural Network
In this code I used VGG16. I implemented pruning and quantization of convolutional and fully connected layers.


