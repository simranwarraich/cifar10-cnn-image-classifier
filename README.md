# Neural Network for Image Classification

This project demonstrates how to build a simple neural network for image classification using the CIFAR-10 dataset and TensorFlow/Keras. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

**Note:** This is a beginner project for anyone interested in getting started with neural networks for image classification.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## Project Structure

```
Neural_Network_for_Image_Classification/
├── Building_a_Simple_Neural_Network_for_Image_Classification.ipynb
├── model_architecture.png
└── README.md
```

## Neural Network Architecture

The neural network architecture used in this project consists of the following layers:

![Model Architecture](model_architecture.png)

- Convolutional Layer (32 filters, 3x3 kernel, ReLU activation)
- Max Pooling Layer (2x2 pool size)
- Convolutional Layer (64 filters, 3x3 kernel, ReLU activation)
- Max Pooling Layer (2x2 pool size)
- Convolutional Layer (64 filters, 3x3 kernel, ReLU activation)
- Flatten Layer
- Dense Layer (64 units, ReLU activation)
- Output Layer (10 units, one for each class)

## Usage

1. Clone the repository or download the source code.
2. Install the required dependencies.
3. Run the `building_a_simple_neural_network_for_image_classification.py` script.

The script will perform the following steps:

1. Import the necessary libraries.
2. Load and preprocess the CIFAR-10 dataset.
3. Define the convolutional neural network architecture.
4. Compile the model with the Adam optimizer and sparse categorical cross-entropy loss.
5. Train the model for 10 epochs, using the test data for validation.
6. Evaluate the model on the test data and print the test accuracy.
7. Plot the training and validation accuracy over the epochs.
8. Select an image from the test set, pass it through the trained model, and print the original and predicted labels.

## Results

After training for 10 epochs, the model achieves a test accuracy of approximately 70% on the CIFAR-10 dataset. The training and validation accuracy curves are plotted to visualize the model's performance. It is important to note that the performance can be further improved through hyperparameter tuning, such as adjusting the learning rate, batch size, or model architecture.However, for the purposes of this project, the focus was on implementing and understanding the main architecture. 


## Dataset

The CIFAR-10 dataset is a collection of images that are widely used for machine learning research. You can find more information about the dataset [here](https://www.cs.toronto.edu/~kriz/cifar.html).

```
@techreport{krizhevsky09,
  author       = {Alex Krizhevsky},
  title        = {Learning Multiple Layers of Features from Tiny Images},
  institution  = {},
  year         = {2009},
  howpublished = {http://www.cs.toronto.edu/~kriz/cifar.html}
}
```

## Acknowledgments

This project uses the CIFAR-10 dataset, which is available for free and widely used in the machine learning community.
