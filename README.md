# Behavioral Cloning Project

## Overview

The goals of this project are the following:

* Use the simulator to collect data of good driving behavior.
* Build a convolution neural network in Keras that predicts steering angles from images captured by the simulator's cameras.
* Train and validate the model with a training and a validation set.
* Test that the model sucessfully drives around track one without leaving the road.
* Summarize the resuls with a written report (this writeup).

[//]: # (Image References)
[lake_center_driving]: ./examples/lake_center_driving.png
[jungle_center_driving]: ./examples/jungle_center_driving.png
[lake_reverse_driving]: ./examples/lake_reverse_driving.png
[jungle_reverse_driving]: ./examples/jungle_reverse_driving.png
[jungle_corner]: ./examples/jungle_corner.png
[training_history]: ./examples/train_local.png


## Rubric Points

The rubric points considered in the project can be found [here](https://review.udacity.com/#!/rubrics/432/view).

## Files Submitted

The workspace (a copy of which can be found in [GitHub](https://github.com/sheayun-kmu/CarND-Behavioral-Cloning-P3) contains the following files (and others):

* `README.md`: This writeup.
* `model.py`: Keras implementation of the proposed convolution neural network. A class `DNNModel` is defined, which has methods for initializing, training, plotting, loading and saving the network model.
* `train.py`: Python driver for training the model. It simply instantiates the CNN model, call the training method with given datasets, plot the training results, and finally saves the trained model to a file (`model.h5`).
* `drive.py`: Flask-based WSGI application provided for the project. This application receives requests from the simulator and responds with steering angle and throttle control values according to the camera image given by the argument to the requests. Only slight modifications are made to cope with environment differences.
* `model_local.h5`: The CNN model trained on a local computer (Ubuntu 16.04 laptop).
* `model_remote.h5`: The CNN model trained on the VM provided by Udacity.
* video files
	- `lake_local.mp4`: Autonomous driving recorded by the simulator on track 1 (lake), where the simulation and WSGI both ran on the local machine.
	- `lake_remote.mp4`: Autonomous driving recorded by the simulator on track 1 (lake), where the simulation and WSGI ran on the remote VM.
	- `jungle_local.mp4`: Autonomous driving recorded by the simulator on track 2 (jungle), where the simulation and WSGI both ran on the local machine.
	- `jungle_remote.mp4`: Autonomous driving recorded by the simulator on track 2 (jungle), where the simulation and WSGI both ran on the remote VM.

## Model Architecture and Training Strategy

### 1. The network model

The CNN model described in [[1]](#1) is employed almost as it is. Going through the model, an image frame is first cropped (top 70 rows and bottom 25 rows are sliced out) because we are not much interested in trees and hills, as well as the engine hood. The resulting image dimension is 320 x (160 - 95) x 3. The images are then normalized (roughly with zero mean in the range [-1, 1] for each color channel), and then fed to the convolution network. The network consists of five convolution layers followed by four fully connected layers, which in total has approximately 27 million connections and 250 thousand parameters. Finally, since we want only a scalar value (steering angle) for the output, the final layer outputs a single value.

The network model, written in Keras, looks as follows:

```
class DNNModel:
    def __init__(self):
        self.top_crop, self.bottom_crop = 70, 25
        self.model = Sequential()
        self.model.add(
            Cropping2D(
                cropping=((self.top_crop, self.bottom_crop), (0, 0)),
                input_shape=(160, 320, 3)
            )
        )
        self.model.add(
            Lambda(
                lambda x: x / 127.5 - 1.0
            )
        )
        self.model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
        self.model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
        self.model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(100))
        self.model.add(Dense(50))
        self.model.add(Dense(10))
        self.model.add(Dense(1))
```

### 2. Data acquisition

Using the simulator, the vehicle is driven two laps each on the lake map and on the jungle map. Along with three images captured by cameras (center, left, and right), the simulator records steering angle, throttle, brake, and speed of the vehicle. Among these, we only use steering angle value as the single-dimension label for the training data.

During the initial phase of data acquisition, the vehicle is driven along the center line of the road in order to train the network to determine the steering angle so that the vehicle remains close to the center in autonomous mode. The following sample images illustrate this center driving behavior.

![Center driving - lake map][lake_center_driving]
![Center driving - jungle map][jungle_center_driving]

These data sets (`lake_twolaps` and `jungle_twolaps`) seemed good enough to train the network so that the vehicle could be driven safely in the autonomous mode, but a few glitches were found in steering the vehicle. In sharp turns and steep hills or approaching a shadowy area, sometimes the autonomously simulated driving veered to one of the road sides (mainly to the left). To make the model generalize better, three additional data sets were used; one provided by the Udacity team (`udacity_data`), two recorded by driving the maps one lap each in the reverse direction (`lake_reverse` and `jungle_reverse`). The following sample images were taken while the vehicle heads approximately the same spot (as the above images) but viewed from the oppoiste direction.

![Reverse driving - lake map][lake_reverse_driving]
![Reverse driving - jungle_map][jungle_reverse_driving]

As can be seen from the pictures, even though we use the same tracks, we can effectively apply two different maps (from the image viewpoint) to training the model. Specifically for the lake map, the vehicle drives mostly counter-clockwise (left turns) feeding the model with image data biased towards left turns. By using data collected in reverse driving, we expect the model to balance the steering angles in general images containing curvatures in lane lines, curbs, and so forth.

After testing the trained model a few times, I found that the autonomous driving would from time to time fail to pass a specific corner in the jungle map. Hoping to mitigate this problem, I added another data set (`jungle_corner`) recorded while driving the vehicle passing that specific corner about ten times. The following picture shows the vehicle driven along that corner.

![Difficult corner][jungle_corner]

As a side note, we would expect the model would train better if data is collected by driving with an analog joystick. When we use a keyboard to drive the simulator, we usually hit A (for left) and D (for right) keys intermittently in moderate corners so as to fine-tune the steering angle and control the car. This would produce pairs of ("left turn image", "straight (zero) steering angle") as well as those of ("left turn image", "left (positive) steering angle") within a short period of time while we release the A key for the left turn. (The same goes for right turns, too.)

However, our conjecture after testing the trained model in autonomous mode is that these effects are largely dwarfed out by "a general tendency" to steer left in the left corners and to steer right in the right corners. This is verified by the fact that the vehicle manages to stay on the road while being autonomously driven.

Finally, aiming at stabilizing the vehicle's control even further, we recorded two other sets of driving behavior (`lake_recovery` and `jungle_recovery`). For these two sets, the vehicle is continuously driven to either side of the road and then steered back to close to the center. The recording is switched off while driving to the side and back on while driving to the center. This is done for the purpose of training the model to recover from a situation where the vehicle veered to either side of the road for whatever the reason. However, these two data sets were not used in the final training because we discovered that training with these sets included resulted in a more spurious behavior of the simulated vehicle.

### 3. Training

The data sets collected in the way described above are summarized in the following.

| Data Set | Number of Samples |
|:---:|:---:|
| `lake_twolaps` | 2,369 |
| `jungle_twolaps` | 3,050 |
| `udacity_data` | 8,036 |
| `lake_reverse` | 1,162 |
| `jungle_reverse` | 1,549 |
| `jungle_corner` | 1,198 |
| **total** | 17,364 |

Since we use images obtained from the left and the right cameras as well, we end up with a total of 52,092 (= 17,364 * 3) images and the corresponding steering angles used to drive the vehicle in the training mode. We applied a constant offset of 0.2 to the steering angle, i.e., we associated (measured steering angle + 0.2) to images taken from the left camera, and (measured steering angle - 0.2) to those taken from the right camera.

To further augment the data sets, we used pairs of mirror images and opposite steering angles as well. Although the data sets already include recordings taken by driving the vehicle in the opposite direction, we assume that applying flipped images in the training process would not hurt the model from being properly trained. That augmentation resulted in doubling the number of samples contined in the final data sets, making the total number of samples in the training data 104,184.

The data sets are partitioned into 80% in the training set, and the remaining 20% in the validation set. Since the data sets are quite large, a generator function (`gen()` in `model.py`) is used to feed the network in an incremental manner. A batch size of 128 is used, and the network is trained for five epochs (both of which are parameterized in the interface to the `DNNModel.train()` method). We first tried training the network on a local Linux machine, which resulted in the following:

```
Epoch 1/5
652/652 [==============================] - 321s 492ms/step - loss: 0.0556 - val_loss: 0.0455
Epoch 2/5
652/652 [==============================] - 320s 490ms/step - loss: 0.0444 - val_loss: 0.0420
Epoch 3/5
652/652 [==============================] - 321s 492ms/step - loss: 0.0405 - val_loss: 0.0395
Epoch 4/5
652/652 [==============================] - 322s 493ms/step - loss: 0.0377 - val_loss: 0.0378
Epoch 5/5
652/652 [==============================] - 321s 492ms/step - loss: 0.0353 - val_loss: 0.0370
```

We verified the data sets are exploited correctly by simply comparing the number of training steps by manual calculation of the (number of samples in the training set) divided by the (batch size) against the above output. We have 83,347 (= 104,184 * 0.8) sample in the training set, which gives 651.15 when divided by 128 (the batch size).

![Loss defined in MSE][training_history]

The above figure shows the mean squared error defined in terms of the steering angle measured in the training process. Both the training loss and the validation loss decreases with the epoch, resulting in the final validation error of 0.0370. Seeing that the losses are reduced throughout the entire learning process and that both errors are small, we assume that the model is not overfit. Note that, lthough the data is not recorded and reported here, in early stages of the model development when the data sets were significantly small, we observed overfitting even with only five epochs.

The time consumed in training the model on the local machine was approximately 5.5 minutes per epoch. At first it took much longer even with a smaller data set, due to the fact that the tensorflow library was not properly configured to use the GPU equipped with the laptop computer. The training time could be further reduced by using the VM provided by Udacity (of course, with GPU mode enabled), whose results are shown below.

```
Epoch 1/5
652/652 [==============================] - 119s 183ms/step - loss: 0.0525 - val_loss: 0.0521
Epoch 2/5
652/652 [==============================] - 116s 178ms/step - loss: 0.0431 - val_loss: 0.0479
Epoch 3/5
652/652 [==============================] - 116s 178ms/step - loss: 0.0396 - val_loss: 0.0442
Epoch 4/5
652/652 [==============================] - 116s 178ms/step - loss: 0.0365 - val_loss: 0.0413
Epoch 5/5
652/652 [==============================] - 115s 177ms/step - loss: 0.0343 - val_loss: 0.0400
```

Here we find that the training time was substantially smaller (approximately 2 minutes per epoch). We verify that the training loss and validation loss are comparable to the case of training on the local computer, which is of course expected. Although, due to the random nature of the deep learning process, these two models are not at all equivalent, both the models were tested on the simulator, whose results are described in the following.

## Tests

First, the simulator is run in the autonomous mode on the local computer, while the WSGI server (implemented in `drive.py`) is running with the model trained on the same computer (`model_local.h5`). The simulated vehicle successfully drove (with an increased target speed of 25 mph) a few laps on both the maps (lake and jungle), of which a single lap for each map is recorded in `lake_local.mp4` and `jungle_local.mp4`, respectively. As can be noted from the video clips, the vehicle largely stayed around the center of the road most of the time. One interesting finding is that the autonomously driven vehicle mimicked the driver's behavior very closely even for the maneuvering habit earned by the driver (in trying to get accustomed to controlling the simulated vehicle). For example, when a series of turns is encountered, the driver deliberately took on inner paths around them to drive as straight as possible, and the autonomous driving was able to capture this behavior.

Second, we performed the same simulation tests with the simulator executed on the local computer, but this time using the model trained in the remote VM (`model_remote.h5`). Slight differences in the driving behavior were observed, but the simulated vehicle was successful in both the maps as well (video not recorded).

Third, we managed to obtain an earlier version of the (presumably) same simulator, and tested the models on a different map contained in it. Not surprisingly, both the models were good enough to be used for autonomously driving the simulated vehicle in this new map not previously seen in the learning process. This indicates that the deep neural network model and the training data were general enough to capture a human driver's perception and response to the road condition (although relying entirely on vision here), at least for this simulator's environment and maps.

Finally, we ran the simulator on the remote VM this time, but without great success. We used `model_remote.h5` in steering angle prediction for both maps. The vehicle managed to drive the lake map successfully (recorded in `lake_remote.mp4`) whereas it was not successful in autonomously drive the jungle map (recorded in `jungle_remote.mp4`). As we decreased the target speed to 15 mph and further to 10 mph, the results got slightly better than while being driven at high speed, we did not manage to successfully drive the jungle map on the remote simulator. The environmental differences between the local simulator and the remote one remain to be investigated.

## References

<a id="nvidia">[1]</a>
Bojarski, M. *et al*.
End to end learning for self-driving cars.
NVIDIA Corporation, 2016.
[https://arxiv.org/abs/1604.07316](https://arxiv.org/abs/1604.07316)
