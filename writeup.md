# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./web_signs/0.jpg "Speed limit (20 km/h)"
[image2]: ./web_signs/9.jpg "No passing"
[image3]: ./web_signs/12.jpg "Priority road"
[image4]: ./web_signs/13.jpg "Yield"
[image5]: ./web_signs/14.jpg "Stop sign"
[image6]: ./web_signs/18.jpg "General caution"
[image7]: ./web_signs/26.jpg "Traffic light"
[image8]: ./web_signs/33.jpg "Turn right ahead"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color does not give any useful info for identifying the traffic sign

As a last step, I normalized the image ((image - 127.5) / 127.5) data because it increase the performance of the model.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   				    | 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten	      	    | output 400 				                    |
| Dropout	      	    | keep_prob 0.5 				                |
| Fully connected		| input 400, output 120        					|
| RELU					|												|
| Dropout	      	    | keep_prob 0.5 				                |
| Fully connected		| input 120, outut 84        					|
| RELU					|												|
| Dropout	      	    | keep_prob 0.5 				                |
| Fully connected		| input 84, output 43      						|
| Softmax				|           									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters:
1) rate = 0.0006
2) epochs = 200
3) batch_size = 128
4) one_hot labels

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9964941521479366
* validation set accuracy of 0.9712018140589569

The LeNet model, that was provided during the previous lessons, was chosen for this project. However, the accuracy of the first version of model has not been greater than 91% even with the extended training dataset. That is not acceptable for the project, as it should be greater or equal than 93%. The reason of this maximum number was the overfitting of the model during the training stage. The solution, how to increase the accuracy, I found [here](https://www.geeksforgeeks.org/dropout-in-neural-networks/). Dropout allows to decrease overfitting. By adding it (with a parameter keep_prob = 0.5) I was able to increase the accuracy of the validation set to 97%.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

The mentioned images were classified perfect, as well as
![alt text][image6]

However, the folowing 2 images were missclassified:
![alt text][image7] ![alt text][image8]

I can assume that the model needs to be trained more precisely, because that images look very clear even for classification by human

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Speed limit (20 km/h) | Speed limit (20 km/h)							|
| Yield					| Yield											|
| No passing	      	| No passing					 				|
| Priority road			| Priority road     							|
| General caution		| General caution    							|
| Traffic light			| General caution     							|
| Turn right ahead		| No vehicles     				    			|


The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%.
