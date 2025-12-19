Project Description: Computer Vision with Deep Learning

Welcome to Computer Vision

Course Structure Overview
1️⃣ Introduction to Computer Vision
Learn how computers "see" images and extract information from them.
Work with images using Python and OpenCV.
Explore deep learning applications in computer vision.
2️⃣ Building a Classification Model
Construct a neural network from scratch for image classification.
Improve performance using pre-trained models and transfer learning.
Complete an end-to-end classification project to apply your knowledge.
3️⃣ Hands-On Project Practice
Work on a real-world computer vision project, applying everything learned so far.
Finalize and submit your project.
4️⃣ Expanding Beyond Classification
Explore advanced computer vision tasks, including object detection and segmentation.
Experiment with more complex deep learning models.
What This Course Offers
Core Computer Vision Skills: Learn how to process, analyze, and classify images using deep learning.
Hands-On Projects: Apply your knowledge by building models and working with real datasets.
Industry-Relevant Techniques: Use state-of-the-art neural networks and explore real-world applications.
At the end of this course, you’ll be able to:
Develop and train deep learning models for image classification.
Work with computer vision libraries to process and manipulate images.
Employ detection models to find and identify objects on images.

# Recap & Conclusions
In this series, we’ve explored two powerful approaches for image classification using the Fashion MNIST dataset. To be precise, we learned about:
1. Building a Simple Neural Network
We started by constructing a basic neural network. This type of model works well for simple tasks, but when dealing with image data, it struggles to capture the spatial relationships in the images. The key takeaway from this lesson was understanding the structure of a neural network: input, hidden layers, and the output layer, and how each component plays a role in classification tasks.
However, as datasets become more complex, basic neural networks hit their limitations. This is why we transitioned to a more specialized approach: Convolutional Neural Networks (CNNs).
2. Improving with Pre-Trained CNNs and Transfer Learning
In the next lesson, we learned that CNNs are designed to work specifically with image data, using convolutional layers to extract hierarchical features from images, such as edges and textures. But instead of training a CNN from scratch—a resource-heavy process—we introduced transfer learning.
By using a pre-trained model like EfficientNetB0, we leveraged the knowledge that the network had already acquired from large datasets like ImageNet. We fine-tuned the model by freezing its convolutional layers to retain the learned features, and we built our own custom fully-connected top layer to adapt it to the Fashion MNIST dataset. This strategy improved performance while saving training time and computational resources.
Key Concepts Recap
Let’s now go over the key takeaways and definitions one more time. Here they are:
Basic Neural Networks: Simple models are good for entry-level tasks but struggle with complex data.
CNNs: Designed to process images by recognizing spatial hierarchies.
Transfer Learning: Using pre-trained models like EfficientNet allows us to bypass the need for extensive training and still achieve excellent results by adapting these models to our specific tasks.
Fine-Tuning: By carefully adjusting the pre-trained models through fine-tuning, we can further improve the model’s performance on our dataset.
Final Thoughts
By combining a basic neural network for foundational understanding with the more advanced method of using transfer learning with CNNs, you now have a solid toolkit for tackling a variety of image classification problems. As you continue your learning journey, remember that choosing the right model and leveraging pre-existing knowledge (through transfer learning) can greatly enhance the efficiency and effectiveness of your solutions.


### Introduction & Key Takeaways
You’ve learned many new things in the previous sprint but we guided you a lot. Now it is your time to shine.
In this sprint we are going to give you a new dataset, state the problem that needs to be solved, provide some key milestones (and some hints) and then give you complete freedom to work with it. Let’s start with the dataset description!
The Dataset
Previously, you worked with the Fashion MNIST dataset. This time we’d like to offer to you the dataset called CIFAR-10. What is this dataset about? Here are the key facts about it:
It consists of 60000 32x32 colour images
This data offers 10 classes for classification, with 6000 images per class.
There are 50000 training images and 10000 test images.
Here are the classes in the dataset, as well as 10 random images from each:
notion image
 
As you can see, these are indeed RGB images. This is what differs this dataset from the previous one you worked with.
The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.
Availability of the Dataset
This dataset is also available in TensorFlow & Keras. Here is how you can import it:

from tensorflow.keras.datasets import cifar10

Note that if you face an error while importing the dataset, it most likely means that you need to install Tensorflow. Here is the command that installs Tensorflow for you. Just run in a call of your Jupyter Notebook:

pip install tensorflow 

Next, getting the actual data can be performed in a similar manner like we did for the Fashion MNIST dataset:

# Load CIFAR-10 dataset
(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()

Please, limit the number of data you are going to use for this project to 10 000 samples the training images (and labels). Here is how you do it:
# Limit the training data to 10,000 samples
n = 10000

train_images = train_images[:n]
train_labels = train_labels[:n]

If you do it once at the beginning of your project, there is not need to limit the data anywhere else!
Note that you may get ore data be setting n to a higher value but it will result in longer training time later, so we recommend sticking with 10k images.
Note that there is no need to limit the testing portion of data because it is 10k images by default.
Key Steps to Do in This Project
Let’s go over a step-by-step plan that we want you to do in this project:
Import all of the neccesary libraries
Get the data and save it to the train_images and test_images variables for images and train_labels and test_labels for labels.
Preprocess images, so they can be fed to a neural net that you are going to train. Note that the actual base model that you’ll need to preprare your data for we’ll discuss next.
For this project one Convolutional Neural Net that you can go with is ResNet50. Similarly, it what we did previously, it is available in tensorflow.keras.applications. Here is how you can import it:

from tensorflow.keras.applications import ResNet50

Initialize this model by setting up the weights parameter to imagenet As for the input_shape, set it to (32, 32, 3) - to the size of the images we are going to work with, so no need to resize and add an extra dimension to the input images. This is going to be your base model.
Don’t forget to “freeze” the layers of the base model at this stage
Build the “head” of your model by adding custom top layers to your ResNet50 base model. Feel free to play around here by setting up different number of layers with different number of neurons in them!
☝️ Our recommendation here
For the baseline, consider using a model with two hidden layers. The first hidden layer usually contains more neurons than the second. The number of neurons in each hidden layer is typically a power of two (e.g., 128, 64, 32, etc.).
Compile your model.
Train the “head” of your model for 10 epochs. Note that it’ll take quite a while to be trained! Please, be patient. In the mean time, pay attention how the metric value changes over time!
Unfreeze the base model and train it as well for 10 epochs. Note that this training phase is going to be even slower that the previous one. 
👉 Note that training phase may take hours. This is absolutely fine for the computer vision tasks

Evaluate the performance of your model on the testing data.
Don’t worry if yor end performance isn’t the best. The more you train, the better your model gets. Remember, we limited you to 10 epochs at the step 7 and 10 epochs at the step 8. At real workplace you’d definitely train it for longer, way longer.

Deliverables
To successfully complete and submit your CIFAR-10 image classification project, make sure to include the following:
Google Colab notebook with all your code, from data preprocessing and model creation to training and evaluation. This notebook should be clean, well-commented, and clearly show the steps you followed.
Live presentation recording
You will present your project live during the final session of this sprint. The session will be recorded, and you are required to submit the recording along with your project notebook.
In your presentation, walk us through:
The problem you're solving and the dataset you're using
Your model architecture and training process
Key challenges, decisions, and takeaways from your work
Make sure your notebook is accessible, and all links shared are viewable to reviewers. 