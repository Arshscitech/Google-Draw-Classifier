# Google-Draw-Classifier
A Deep Neural Network To Classify input Drawing among 15 predefined Classes.

The Dataset Used can be found here:

https://github.com/googlecreativelab/quickdraw-dataset


This is more of a testing Project, where i tried to classify images using Deep Neural Networks only rather than Convolutional Neural Networks. Moreover i chose to use Binomial Classification rather than Multiclass Classification just for testing purposes. 

The Entire Neural Network Is Coded From Scrath Without the Use Of any Machine Learning Framework like Keras, Tensorflow, etc.

For testing purposes, Binomial Classification is used and 15 different models are trained, one to identify each class. 

The accuracies of Each Model during Training is shown in training.txt. The accuracies achieved are commparatively good with most of them above 95%

During Testing, The image is tested with all the 15 models and the class of the image is given by the model with the best accuracy.

Some of the images correctly classified by the model are attached. 

This project was made a Year Back, Just to understand Machine Learning firmly and to try something different as a testing purpose.
The performance and the efficiency of the model can be improved by using Convolutional Neural Networks, MultiClass Classification, etc


