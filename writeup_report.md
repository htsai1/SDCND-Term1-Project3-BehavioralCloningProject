#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/modelarch.jpg "Model Arch"
[image2]: ./images/centraldriving.jpg "Central driving Image"
[image3]: ./images/leftsideimage.jpg "Left side Image"
[image4]: ./images/rightsideimage.jpg "Right side Image"
[image5]: ./examples/placeholder_small.png "Normal Image"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I built my model with reference of Nvidia Architecture. The final model consists of total five convolution neural network (three with 5x5 filter and two with 3x3 filter sizes) and four dense layers.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer . 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets. Totally I used about 15 different datasets to ensure that the model was not overfitting.The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a below training data types:
- Entire Full Track Recording Data: center lane driving of entire track
- Data that focuses more on the curvatures: To train the model to handle the curves more smoothly, I also create some datasets that record only driving during curvatures.  
- Recovering from the left and right sides of the road: This helps the model to learn how to steer back to the central when the vehicle is off to the sides.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

Below is my approach of deriving the final model architecture:

1. My first step was to use a convolution neural network model similar to the Nvidia Architecture. I considered this model might be appropriate because this architecture has been proven by Nvidia team.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout layer.

Then I ran the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like the curve that after the bridge, the car was having trouble to handle that and seems easily get into the dirt road, wich is off the drivable lane. To improve the driving behavior in these cases, I created other 5 datasets that is focusing on this curvature. I also record the data of driving clock-wise particularly on this curvature.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:
![alt text][image1]


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer more to the left when the vehicle sees the right sides of image, and steer more to the right when the car see left side of image. 

Left side image:

![alt text][image3]

Reft side image:

![alt text][image4]


To augment the data sat, I also flipped images and angles thinking that this would create more comprehensive/generalized data to the model for training. 



After the collection process, I had 11875 number of data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used the training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the I used an adam optimizer so that manually training the learning rate wasn't necessary.
