# Deep Compression: Compression of Deep Neural Network

This is a tensorFlow implementation of Deep Compression [paper](https://arxiv.org/abs/1510.00149) which is inspired by the previous tensorflow and pytorch implementations.

This implementation is designed with the following goals:
- Clear Pipeline: it has pipeline of network compression including pruning and quantization.
- Modularity: This code has been implemented for element-wise pruning and clustering-based quantization. However, it is modular and easy to be expanded for other types of pruning and quantization methods.
- Extendable: Currently this code has been implemented for VGG16, but it can be extended for other networks.
- To be deployed on Embedded Systems 

This implementation includes 2 methods in the Deep Compression method:

- Pruning
- Quantization

### Pruning
There are different types of pruning. In this code, I implemented element-wise pruning in which the weights are prunned if they are smaller than a threshold.

### Quantization
There are different types of quantization. In this code, I implemented clutering-based quantization.


### Deep Neural Network
I implemented pruning and quantization for convolutional and fully connected layers of VGG16 network.


