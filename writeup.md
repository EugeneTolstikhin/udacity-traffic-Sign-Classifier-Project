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
5) Optimizer - Adam Optimizer.

How these parameters were chosen:
1) Experimentally. The values between 1e-3 - 1e-5 has been considered. The rate 0.0005 - 0.0006 showed the fastest training speed till the maximum accuracy
2) Experimantally. Values 10 - 200 were considered. The values between 150 - 200 has showed the acceptable accuracy. The value greater than 200 showed no additional profit.
3) Experimantally. Considered the values - power of 2. 128 = 2^7
4) One hot encoding - binary representation of all the unique image category. Provide faster access to each category of the image.
5) Adam (= Adaptive Moment Estimation). [Overview of the different optimizers](https://arxiv.org/pdf/1609.04747.pdf) at the paragraph 4.6, 4.10 explains why Adam, which can be considered as a different combinations of the other optimizers, is better for usage. A short summary of why Adam is a good choice: it's computationally efficient and requires very little memory.

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

However, the following 2 images were missclassified:
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

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The result of top 5 softmax probabilities in a raw way looks like that:

INFO:tensorflow:Restoring parameters from ./lenet
TopKV2(values=array([[6.9766802e-01, 1.7054953e-01, 8.8648960e-02, 3.2393917e-02,
        5.4805018e-03],
       [1.0000000e+00, 2.9534826e-11, 3.5334247e-16, 3.3603949e-16,
        1.7622580e-17],
       [1.0000000e+00, 1.3792606e-13, 6.7945612e-15, 6.3335656e-15,
        9.2442201e-16],
       [1.0000000e+00, 5.6577815e-12, 5.0071848e-12, 1.5585289e-12,
        2.2231360e-13],
       [4.4449526e-01, 2.8652456e-01, 1.5980074e-01, 8.0277681e-02,
        1.8021077e-02],
       [4.7033024e-01, 1.9266245e-01, 1.2918560e-01, 8.1047580e-02,
        5.0550640e-02],
       [9.2234945e-01, 2.0823237e-02, 1.0969975e-02, 8.9479005e-03,
        7.6923613e-03],
       [1.0000000e+00, 2.5568959e-21, 1.9284084e-25, 1.6865276e-28,
        4.8183433e-29]], dtype=float32), indices=array([[ 0,  8,  1,  4,  5],
       [12, 40, 13, 35, 25],
       [13, 14, 15, 39,  1],
       [14, 13, 38, 34,  3],
       [18, 24, 26, 27, 20],
       [18, 11, 27, 25, 24],
       [15,  2,  4,  3,  1],
       [ 9, 10, 16, 41, 15]], dtype=int32))

How it should be explained:

1st picture's probabilities:

| Label	| Sign name             | Probability  |
|:-----:|:---------------------:|:------------:|
| 0     | Speed limit (20 km/h) | 0.698        |
| 8     | End of speed limit    | 0.175        |
| 1     | Speed limit (30 km/h) | 0.088        |
| 4     | Speed limit (70 km/h) | 0.032        |
| 5     | Speed limit (80 km/h) | 0.005        |

2nd picture's probabilities:

| Label	| Sign name             | Probability  |
|:-----:|:---------------------:|:------------:|
| 12    | Priority road         |  1           |
| 40    | Roundabout mandatory  |  0           |
| 13    | Yield                 |  0           |
| 35    | Ahead only            |  0           |
| 25    | Road work             |  0           |

3rd picture's probabilities:

| Label	| Sign name             | Probability  |
|:-----:|:---------------------:|:------------:|
| 13    | Yield                 |  1           |
| 14    | Stop                  |  0           |
| 15    | No vehicles           |  0           |
| 39    | Keep left             |  0           |
| 1     | Speed limit (30 km/h) |  0           |

4th picture's probabilities:

| Label	| Sign name             | Probability  |
|:-----:|:---------------------:|:------------:|
| 14    | Stop                  |  1           |
| 13    | Yield                 |  0           |
| 38    | Keep right            |  0           |
| 34    | Turn left ahead       |  0           |
| 3     | Speed limit (60km/h)  |  0           |

5th picture's probabilities:

| Label	| Sign name                    | Probability  |
|:-----:|:----------------------------:|:------------:|
| 18    | General caution              |  0.444       |
| 24    | Road narrows on the right    |  0.287       |
| 26    | Traffic signals              |  0.160       |
| 27    | Pedestrians                  |  0.080       |
| 20    | Dangerous curve to the right |  0.018       |

6th picture's probabilities:

| Label	| Sign name                             | Probability  |
|:-----:|:-------------------------------------:|:------------:|
| 18    | General caution                       |  0.470       |
| 11    | Right-of-way at the next intersection |  0.192       |
| 27    | Pedestrians                           |  0.129       |
| 25    | Road work                             |  0.080       |
| 24    | Road narrows on the right             |  0.051       |

7th picture's probabilities:

| Label	| Sign name            | Probability  |
|:-----:|:--------------------:|:------------:|
| 15    | No vehicles          |  0.922       |
| 2     | Speed limit (50km/h) |  0.021       |
| 4     | Speed limit (70km/h) |  0.011       |
| 3     | Speed limit (60km/h) |  0.009       |
| 1     | Speed limit (30km/h) |  0.008       |

8th picture's probabilities:

| Label	| Sign name                                    | Probability  |
|:-----:|:--------------------------------------------:|:------------:|
| 9     | No passing                                   |  1           |
| 10    | No passing for vehicles over 3.5 metric tons |  0           |
| 16    | Vehicles over 3.5 metric tons prohibited     |  0           |
| 41    | End of no passing                            |  0           |
| 15    | No vehicles                                  |  0           |
