# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/layer_activations.png "Layer Activations Visualization"
[image2]: ./writeup_images/model_summary.png "Keras Model Summary"
[image3]: ./writeup_images/model_architecture_visualization.png "Model Architecture Visualization"
[image4]: ./writeup_images/initial_data_distribution_bias.png "Bias in Initial Dataset"
[image5]: ./writeup_images/flipped_images_sample.png "Flipping Sample Image"
[image6]: ./writeup_images/recovery_sample.png "Recovery Sample Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project submission includes the following files for the project requirements:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained CNN model for driving in autonomous mode on both track 1 and 2
* **track1.mp4** containing a video of the car driving in autonomous mode on track 1
* **README.md** (this file), containing a writeup summarizing the approach taken and the results

*Optional files:*
* *track2.mp4*, containing a video of the car driving in autonomous mode on track2 (promising, though not completing the track successfully), using exactly the same model and dataset
* *track2_tuned_model_full_lap.mp4*, an optional video, showing full lap completion on track2, but with some lane changes, but tried with a slightly tuned model on the same dataset


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously by executing

```sh
python drive.py model.h5

```
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a **convolutional neural network based on the LeNet-5 architecture** with 2 convolutional layers with 5x5 filter sizes, 2 fully connected layers and an output layer. This is described in more detail in the following section named Architecture Documentation. 

The model includes RELU layers in both the convolutional layers as well as the 2 fully connected layers to introduce nonlinearity (code lines 117-127), and the data is cropped to remove the regions normally appearing above the horizon using a Keras Cropped2D layer (code line 112) and also normalized in the model using a Keras lambda layer (code line 115). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after both the fully connected layers, in order to reduce overfitting ([model.py](model.py), lines 122-128). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 77-78, 146-153). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an **adam optimizer**, so the learning rate was not tuned manually ([model.py](model.py), line 136). The batch size, the number of epochs, etc. were tuned by trial and error.

#### 4. Appropriate training data

Training data was chosen such that the vehicle drives as much as possible on the centre of the road. But, in case the car veers from the centre, it would not know how to get back to the centre. So, additional training data was also created by driving a recovery lap, teaching the car how to get back to the road centre, in case it moves close to the edge of the track. **Data augmentation** was done using **flipping, using images from left and right camera, etc.** Data was **preprocessed** to **reduce the image size** to half(for efficiency), **cropped** to remove regions above the road and also **normalized**. Data from **track 2** was also added.

For more details about how I created the training data, please refer to the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a very basic model and keep adding layers and stop once a reasonably good solution was found.

1. My first step was to use a **barebones network** with just one convolutional layer, fully connected layer and an output layer with one neuron outputting the steering angle. This was just to **get the pipeline working**. The vehicle went off the track pretty soon, as expected, with this model. 

2. Once I had a working pipeline, I decided to **increase the depth** with one convolutional layer. Since, this was getting close to the **LeNet-5** architecture, I decided to mimic the LeNet-5 architecture with the same width and depth. This architecture was slightly changed by introducing MaxPooling layers instead of the subsampling in the original paper and I used **'ReLu' activation for non-linearity** for both the convolutional as well as the fully connected layers. The final output layer was modified by having only one neuron (Reason - this is a regression problem requiring just the prediction of one value - the 'Steering Angle', and not a classification problem with 10 classes for which the originial LeNet was designed.)

3. This model was found to be much better and kept the vehicle on the track most of the time. I also wanted to get an idea of what was happening in each of the layers. To do this, I fed in some sample images to the network and observed the activations after each layer. While **visualizing the layer activations**, I observed that the MaxPooling after the second convolution was actually causing loss of information to the fully connected layers as the image size got significantly reduced. So, I **removed the last MaxPooling layer** before training once again. The activations of the final model (fed with one sample image from Track 1 and Track 2 each) are shown in the below images:

![alt text][image1]

4. In order to find out how well the model was working, I **split the data into a training and validation set**. I found the validation loss to be higher than the training loss. This meant the model was overfitting. So, I decided to **add dropout layers** after both the fully connected layers. Also, a **Keras callback** was created to save only the model with the **best validation loss**, in case the validation loss increases after a few Epochs, while the training loss decreases. This also helped to **reduce overfitting.**

5. Even with the changes mentioned above, the model was still not performing satifactorily. These issues were addressed with **changes to the training data**. A detailed description of the training set and training process is given in the respective section below. After improving the training data, the model was found to be working good and the vehicle did not leave the track at all. For most of the time, the vehicle stuck to the centre of the road, and whenever it veered off away from the centre, it quickly came back to the centre before touching the lane lines.

6. Inorder to check if using a better model to further improve the solution, **the NVIDIA model** was also tried out with the same data and the layers and hyperparameters were tuned for the NVIDIA model. But, it was observed that the NVIDIA model was **overfitting and also took much longer time to train**. It probably means that more data is required to make the NVIDIA model work properly(The original paper says it was trained on 72 hours of real world training data, while I just had just around 4 laps of simulation training data on a simple track). I had two options now - either tune the NVIDIA model parameters and add more training data or continue with the LeNet model for track 2. I chose the latter option considering the higher training time for option 1, and the **limitations of not having a GPU** (there were issues with Udacity workspaces getting frequently disconneced and refreshed, so the models were **trained on my laptop without a GPU).**

7. I went **back to the LeNet based model** and decided to **test it on track 2.** As expected, the model trained on track 1 was not working on track 2. It went off the track in a few seconds. I decided to add training data from track 2. And when tested, it worked pretty good for just one forward and reverse lap. Though it was not perfect on track 2, as it could not maintain the right lane throughout, and got off the road in a couple of places, overall it was performing well. I realized that **this model architecture is able to learn well**, and would run smoothly on track 2 too, if provided with more data like a recovery lap.

8. I also wanted to **check if this model trained on both the tracks** could perform well on track 1 again. So, I **tested it on track 1 again and found it to be meeting the requirements.**

#### 2. Final Model Architecture

The final model architecture ([model.py](model.py) lines 108-130) consisted of a convolutional neural network based on the LeNet-architecture (with some modifications as discussed in the previous section), with the following layers (layer shapes shown in the diagram):

![alt text][image2]

Here is a **visualization of the architecture:**

![alt text][image3]

#### 3. Creation of the Training Set & Training Process

These are the steps I took for creating the training set:

1. I first recorded **one normal lap on track 1 driving on the lane centre**. I trained the modified LeNet-architecture on this data and observed that the car was **driving smoothly on straights but cornering sharply or missing the corners and leaving the track**. I realized that this is a reflection of my driving on the simulator - I drove at highest possible speed and cornered sharply at the curves. This meant that probably there was very few data on smooth cornering in the data set. 

2. I decided to check the **distribution of the dataset**. And as expected, I observed that there was very little data on steering angles other than zero. It was a **dataset highly biased towards straight driving** and as a result the car learnt to drive straight, but not turn smoothly. Also, the **bias towards negative steering (left turns)** is more than towards positive steering(right turns), due to nature of the track 1.

![alt text][image4]

3. To correct this bias, I decided to **restrict the amount of straight driving data**, by selecting randomly only a fraction of the images with close to zero steering angles. (This parameter was tuned all throughout the training process). It helped, but due to this restriction, the amount of data was very less, indicating the need for more data. 

4. Rather than creating more training data by driving a reverse lap, I used the **left-right flipping of images** to quickly double the amount of data. For example, here is an image that has then been flipped:

![alt text][image5]

5. Inorder to get more images, I decided to **use left and right camera images with a steering angle correction** of 0.2 (this parameter value was found to be good based on results). This helped triple the amount of data without explicity driving more training laps. I also added **an extra forward lap**.

6. Although the above steps helped improve the perrformance on the track, there were **still a few instances where the vehicle left the track**. I realized that this is because **most of the data was focussed on driving on the track centre**, such that the model **could not learn what should be done when it reaches the edges** in a few cases. So, I decided to drive a **recovery lap**, recording the vehicle recovering from the left side and right sides of the road back to center. These images show what a recovery looks like:

![alt text][image6]


7. Adding the **recovery lap to the training set greatly helped**. The vehicle did not move out of the road, but there was a **new problem observed**. It was driving in a **zig-zag manner** and reaching close to the road edges while cornering. This means that the amount of smooth driving data was far less. I decided to address this issue by **replacing my initial two fast laps of sharp cornering with just a single slow lap of smooth cornering.** Driving during training at a slower speed had 2 advantages:
    a. **Quantity of data** - More data recorded in one lap, since more frames are recorded.
    b. **Quality of data** - The steering is smoother, since at slower speed, the training data was with a wide range of steering angles, instead of just a few sharp turns at higher speed.

8. The **model was already performing pretty good with all these changes and met the requirements on track 1**. I decided to **attempt the track 2**. As expected, the training on track 1 alone was not sufficient to drive properly in autonomous mode on track 2. So, I added some training data from track 2 with 1 forward lap and 1 reverse lap. With this additional data, the model was able to navigate the track 2 to a large extent, though it did not successfully complete the lap without getting out of lane/track.

9. I also wanted to check if adding data from track 2 deteriorated the performance on track 1. But on checking, I found that the model was performing good on track 1 too. So, I finalized this model for submission.

8. With all these changes to the training data, the **final data set** consisted of :
  **a. One slow forward lap on track 1 
    b. One recovery lap on track 1 
    c. One slow forward lap on track 2 
    d. One slow reverse lap on track 2**

All the **data was randomly shuffled** and **20% of this data was set apart as validation set**. The validation set helped determine if the model was over or under fitting. This **common dataset for track 1 and 2** was **trained using the same LeNet-5 architecture model**.


### Simulation

#### Ability of the car to navigate correctly on test data - 'Track 1'

The final model has a modified LeNet-5 architecture trained on both track 1 and 2. This model can be found in the file [model.py](model.py). The trained model is saved as [model.h5](model.h5). This **model is successfully able to navigate track 1 without leaving the track at any point of time**. A recording of the vehicle driving autonomously with this model is available in [track1.mp4](track1.mp4)

#### Ability of the car to navigate correctly on the more difficult 'Track 2'

The same [model](model.py) was tested on track 2. It was found that the vehicle was able to **navigate a significant portion of the track 2** with this model, but faced a few problems such as:

1. Moving out from right lane to left lane in 2 instances
2. Driving on the edge of the lane in a few instances
3. Moving out of the road in a couple of instances

As a result, this model was not able to complete the lap successfully without leaving the track. A video this performance on track 2 can be found in [track2.mp4](track2.mp4)

I tried improving this further by tuning the model and obtained another result, where the vehicle navigated a complete lap without leaving the track (though it changed lanes twice). This can be seen in [track2_tuned_model_full_lap.mp4](track2_tuned_model_full_lap.mp4), but the results were not reproducible, since the vehicle drove differently each time not only when the model was trained again, but also with the same trained model(It seems that the performance changes in the simulator even with the same model.h5 file).

####  Summary

The model is **able to navigate successfully on track one and needs additional data and some parameter tuning to run successfully on track two**. As **next steps, more training data can be provided and the NVIDIA model** can be trained on this larger amount of data to get a robust solution.