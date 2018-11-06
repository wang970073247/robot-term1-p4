
## Deep Learning Project ##

The purpose of this project is to train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques that could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./images/show.png
[image_1]: ./images/fcn.png
[image_2]: ./images/loss.png

![alt text][image_0] 

### Concepts


#### Fully Convolutional Network

It is a 1x1 Convolution Layers.To preserve spatial information, as well as to keep the non-linearity of the network, we can replace the Fully-Connected Layers with 1x1 Convolution Layers. There are 2 additional advantages in doing this: one, we can feed images of any size into the resulting trained network, and two, we can easily increase and decrease the complexity of the most compact (usually) representation by varying the number of 1x1 filters.

Disadvantages for these layers include the loss of flexibility since they have less parameters to be learned, which may affect some highly complex environments and applications. Also, pre-trained models that solely rely on Convolution Layers are likely to be computationally slower than similar networks that make use of Fully-Connected Layers.

#### Bilinear Upsampling

To make use of the kept spatial information, we make use of bilinear upsampling to predict pixel probabilities. This requires less overall training, as compared to traditional transpose convolutions, since the bilinear upsampling is just linear interpolations of values.

#### Skip Connections

Another powerful tweak we can make to the network is through the use of skip connections. In essence, this means combining the high-level and low-level information to make spatial predictions more robust and accurate. This is done in this project by concatenating a prior layer that has the same height and width, and then perform convolution on this resulting layer.

#### Encoding and decoding images

With a typical CNN, we are able to encode input images into a spatially-shrunk representation by reducing the height and width with each added convolution layer with a stride larger than 1. As the height and width decreases, it extracts higher and higher level of information from the given image, and thus the depth of the output is also increased so as to not lose too much information. With the above-mentioned 1x1 convolution layers, the resulting representation maintains some spatial information which we can make use of in predictions.

To be able to annotate pixels of the given image with their classification, we can attempt to decode this encoded information through the use of decoder blocks. In these decoder blocks, we attempt to restore the spatial size (gradually, to original input size) by performing bilinear upsampling. We also use skip connections to include lower-level information from prior layers which improves accuracy of pixel classification. In addition, we can include some convolution layers in the decoder blocks to continue extract information from prior layers which may have been missed out earlier.

#### Batch normalization

To optimize network training, we can also make use of batch normalization, which is to normalize the input batch at each layer. This allows us to mitigate vanishing gradient to some extent, as well as provide some level of regularization.

#### Depthwise Separable Convolutions

Instead of the traditional convolution layers, we can instead make use of separable convolution layers. What the layer does is to perform convolution on each of the input channels, and then apply 1x1 convolutions to the result. The advantage is that it uses less parameters which enables fast computation, both in learning and evaluation.

#### Structure of FCN

A typical FCN composed of a encoder section and a decoder section which are connected with a 1 x 1 convolution. The input is pumped though the whole network.
A special technique called skip layers is introduced to improve the performs of FCN. As show in Fig. 2, some of the layers in the encoder section or even the input itself are concatenated to the decoder layers, so that the overall information in the input image can be reserved.
In order to test and tune the performance of the network, a special parameter num_filters is employed to define the number of kernels in each layer.


The purpose of this neural network is to performed scene processing and understanding. This is done by examining and classifying each pixel in a given image.

#### Hyper Parameters

For the FCN, there are mainly 5 hyper-parameters to tune. They are:

* learning_rate: defines the ratio that how much the weights are updated in each propagation.
* batch_size: number of training samples/images that get propagated through the network in a single pass.
* num_epochs: number of times the entire training dataset gets propagated through the network.
* steps_per_epoch: number of batches of training images that go through the network in 1 epoch. We have provided you with a default value. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.
* validation_steps: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. We have provided you with a default value for this as well.

Among these hyper parameters, three of them are critical to the training process of the FCN. They are learning_rate, batch_size and num_epochs. The learning_rate determines how much will the weights be updated in each propagation pass. Although a higher learning rate may fasten the training process, it may also make the results less accurate. The determination of batch size is based on the hardware ability of the GPU. Number of epochs is the number of loops the training process will perform. After trying several different settings, the final hyper paramaters are determined as in the jupyter notebook.


### Results

#### Architecture
  
![alt text][image_1] 
  
This is the resulting model used for training. Each decoder block included 3 layers of separable convolution to sieve out finer details from prior layers. 
A 1x1 convolution layer with 96 filters was found to perform well in many different trials and was thus used.

#### Parameters

After a few test runs, i choose values of parameters as follow:

* learning_rate = 0.001
* batch_size = 20
* num_epochs = 24
* steps_per_epoch = 250
* validation_steps = 50
* workers = 4

#### Performance

![alt text][image_2] 


A total of 4892 training images were used. Finally, a score of 0.4267 was achieved.

### Future Work

Various improvements can be implemented on this model in order to improve its performance. From our observations, we can see that the majority of the errors are produced when either a target is too far or when we have poor lighting conditions that affect the color of the target.
When target is very distant, a very small amount of pixels is associated to the target. This means that the neural network has a harder time associating the small amount off pixel to the actual target; especially because the amount of detail available are heavily degraded in a smaller number of pixels. This is a similar concept that happens in real life, as objects are further, we humans might have a harder time distinguishing unique features between different objects. 
In order to improve classification of very distant objects, we might need to look into more features or specifically more unique attributes. For example, every human as a unique walking pattern. Perhaps we can learning the walking patterns or gaits and use this as a method for classification. 
The lighting issue could be alleviated by applying filters that could potentially remove the illumination changes before the image is fed to the neural network.

