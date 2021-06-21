# Convolutional-Neural-Network
Classification of handwritten digits using CNN

The Architecture of a CNN network consists of three layers, namely:
1. Convolution Layer
1. Pooling layer
1. Fully Connected Layer
### Convolution Layer: 
The main aim of Convolution Operation is to extract the features such as edges from a given input image. This is done by performing the convolution operation between a part of the selected image with the kernel. There can be multiple convolution layers in a CNN. One layer might be responsible for low-level features such as edges, where as another layer might help in extraction of high-level features. The kernel moves across the entire image with a certain stride value. The result can be of two types i.e. by using Valid Padding or Same Padding. The dimensions of the resulting image are reduced while still maintaining the important features.
### Pooling Layer: 
Same as the convolution layer the pooling layer also reduces the spatial size of the feature. It mainly reduces the computational power required for processing of the data. The 2 types of Polling that we generally observe are Max Pooling and Average Pooling. Max pooling returns the maximum value of the portion of image where as average pooling returns the average. The max pooling performs Noise suppression. But Average pooling only aims to reduce the dimensions.
### Fully Connected Layer: 
After going through multiple Convolution and Pooling layers, the dimensions are flattened. This flattened image data is fed to an MLP/ANN. Here each of the neurons of the previous layer are connected to each of the neurons of the next layer. Over a number of epochs, the model learns to distinguish between the high-level and low-level features and performs classification.

## Algorithm
1. Import NumPy, TensorFlow and other required libraries
2. Load the data set
3. Split the data into training and testing dataset
4. Check the shape and datatype
5. Reshape
6. Using to_categorical in keras to get a binary class matrix from vector class
7. Define the model architecture. Here we use sequential type of model
8. Increasing the dataset using ‘ImageDataGenerator’
9. Compile the model using optimizer Adam
10. Set a learning rate using ‘LearningRateScheduler’
11. Fit the model
12. Evaluate the model
13. Plot the loss and accuracy
14. Print the Confusion Matrix
15. Print the model summary

## Dataset Description
I have considered the MNIST Data Set (Modified Institute of Standards and Technology database) of handwritten digits.
* It contains 60,000 training images set and 10,000 testing image set.
* It consists of 10 classes i.e. 0-9 handwritten digits.
* The resolution of each of these images is 28x28 =784 pixels.
* The images are in Grey-scale meaning the value of each pixel ranges from 0-255.

![image](https://user-images.githubusercontent.com/58825386/122784620-42c26a00-d2d0-11eb-99c4-bd707d95d90d.png)

The data files named train.csv and test.csv contain gray-scale images of hand-written digits, from zero all the way through nine. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.
The training data set, has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.
