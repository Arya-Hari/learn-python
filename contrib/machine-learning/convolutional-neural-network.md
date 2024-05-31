# Overview

<div align="justify">Machine Learning and Artificial Intelligence (AIML) is a popular field of study in computer science today.It is used in a variety of fields, a popular one being Computer Vision. Computer vision is a field of Artificial Intelligence (AI) that enables a computer to understand and interpret the image or visual data. Convolutional Neural Network or CNN is a type of deep learning Neural Network (NN) architecture that is commonly used for Computer Vision tasks.

# Introduction - What are Convolutional Neural Networks?

Convolutional Neural Network or CNN is a type of deep learning Neural Network architecture that is commonly used for Computer Vision tasks. It is an extension of Artificial Neural Networks, and is predominantly used on image-related data. It works very well on visual datasets like images or videos. 

A typical CNN model, like any NN model, consists of different layers. 
- The input layer consists of your image input, which then passes it on to the next layer for processing.
- The next layer is a convolutional layer, whose role is to act like a filter on your image input and extract necessary features from the image, such as edges, shapes, etc.
- The number of features extracted are substantially large and it would require more computational power to process all of these features. Thus, the next layer, called the pooling layer, down samples the features extracted to reduce computation costs while at the same time ensuring that the features very well represent the overall image input.
- After the pooling layer, the fully connected layer or dense layer makes the final prediction based on mathematical algorithms that the model is trained on and the extracted features, and produces the output in the output layer.

There can exist several convolutional and pooling layers, each for a specific type of feature. 

# Key Features - 

- Focus on Grid-like Data - CNNs are designed to work with data arranged in a grid, like pixels in an image. This allows them to capture the spatial relationships between features.
- Feature Extraction through Convolution - A core part of a CNN is the convolutional layer. Here, a filter (kernel) slides across the input data, extracting local features like edges, shapes, and colors.
- Spatial Invariance - CNNs can recognize objects even if they appear in different locations or orientations within the image. This is because they learn features based on their relative positions, not absolute coordinates. 
- Learning Feature Hierarchies - CNNs typically have multiple convolutional layers stacked together. Each layer learns increasingly complex features based on the outputs of the previous layer. This allows the network to build a hierarchy of features, from simple edges to intricate object parts. 
- Pooling for Downsampling - Pooling layers in a CNN help reduce the dimensionality of the data while preserving important features. This makes the network more efficient and less prone to overfitting. 
- Fully Connected Layers for Classification - In the final stages, CNNs often have fully connected layers similar to regular neural networks. These layers take the extracted features and use them for tasks like image classification or object detection.

# Use Cases - 

- Image Classification - This is a classic use case. CNNs can be trained to identify objects, scenes, or even specific entities within images. From classifying products in e-commerce to recognizing animals in wildlife cameras, CNNs power a variety of image recognition tasks. 
- Object Detection - CNNs can not only classify objects but also pinpoint their location within an image. This is crucial for applications like self-driving cars that need to detect and localize obstacles on the road. 
- Facial Recognition - Social media platforms and security systems leverage CNNs for facial recognition. By analyzing facial features, CNNs can identify individuals or verify user identities. 
- Image Segmentation - CNNs can segment images into different regions, assigning labels to each part. This is useful in medical imaging for segmenting tumors or organs, or in self-driving cars to differentiate between road, lane markings, and pedestrians. 
- Medical Image Analysis - CNNs are transforming medical diagnosis by analyzing X-rays, MRIs, and other scans. They can detect abnormalities, classify tumors, and even predict disease progression. 
- Video Analytics -  CNNs can be applied to video data for tasks like action recognition (spotting specific actions in videos), object tracking (following objects as they move), and scene segmentation (understanding the environment in a video). 
- Content Moderation - Social media platforms use CNNs to automatically detect and flag inappropriate content in images or videos. 
- Recommender Systems - CNNs can analyze user preferences based on their interactions with images or videos, recommending similar content or products.

# Example - 

In this example, a simple image classification model will be demonstrated based on the cifar-10 dataset available through TensorFlow. It uses a simple CNN model to classify images into different categories. </div>

```python
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Split your dataset into training and test sub sets 
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the class names as per the dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Define the model type and layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Print a description of the model and its layers
model.summary()

# Train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

```
Now to test whether the model does the required task, run the below script.

```python
# Import necessary libraries
import numpy as np
from PIL import Image

def imageToArray(imageName):
  # Load the image and resize it to the desired dimensions
  image_path = f'/content/{imageName}.jpg'
  width, height = 32, 32  

  image = Image.open(image_path)
  image = image.resize((width, height))
  print(image.width)
  # Convert the image to a NumPy array and normalize the pixel values (if necessary)
  image_array = np.asarray(image)
  image_array = image_array / 255.0  # Normalize the pixel values between 0 and 1

  print(image_array.shape)
  # Reshape the image array to match the input shape of your model
  image_array = image_array.reshape(1, width, height, 3)  # Assumes the input shape is (width, height, 3)

  return image_array

CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog'
                   , 'frog', 'horse', 'ship', 'truck'])

imageName = "image1" # Replace with your .jpg image name
image_array = imageToArray(imageName)
preds = model.predict(image_array) 
preds_single = CLASSES[np.argmax(preds, axis = -1)] 
actual_single = imageName
print(preds_single)
print(actual_single)
```




