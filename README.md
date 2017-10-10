# Tensorflow-Example
An implementation of a custom model function for deep learning in combination with the Estimator API, using Google's Tensorflow.

This example shows how to:
* build a custom deep learning model function with the estimator api
* use a custom decaying learning rate
* log custom metrics during training, evaluation and prediction
* print those metrics
* save custom training metrics for visualization in Tensorboard

For reproducibility, the MNIST dataset is used. Since this example only shows the implementation of a basic deep learning model, no convolutional layers are used (although these would probably improve performance). They can easily be implemented by changing the 'basic' neural layers to convolutional layers in the model function.   
