import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, Cropping2D

# A generator function to yield a subset of batch_size
def gen(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample['img_path']
                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if batch_sample['flip'] == True:
                    image = cv2.flip(image, 1)
                angle = float(batch_sample['angle'])
                images.append(image)
                angles.append(angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# A DNN model to capture driving behavior
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

        self.training_history = None

    # Train the network using data contained in the datapaths
    def train(self, datapaths, epochs=5, batch_size=32, train_flipped=False):
        # Collect samples from manual driving recordings.
        steer_offset = 0.2
        samples = []
        for p in datapaths:
            with open(p + '/driving_log.csv') as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    paths = [p + '/IMG/' + x.split('/')[-1] for x in line[:3]]
                    center, left, right = paths
                    steer = float(line[3])
                    samples.extend(
                        [
                            {
                                'img_path': center,
                                'flip': False,
                                'angle': steer,
                            },
                            {
                                'img_path': left,
                                'flip': False,
                                'angle': steer + steer_offset,
                            },
                            {
                                'img_path': right,
                                'flip': False,
                                'angle': steer - steer_offset,
                            },
                        ]
                    )
                    if train_flipped == False:
                        continue
                    samples.extend(
                        [
                            {
                                'img_path': center,
                                'flip': True,
                                'angle': -1.0 * steer,
                            },
                            {
                                'img_path': left,
                                'flip': True,
                                'angle': -1.0 * (steer + steer_offset),
                            },
                            {
                                'img_path': right,
                                'flip': True,
                                'angle': -1.0 * (steer - steer_offset),
                            },
                        ]
                    )

        # Partition the samples into a training set and a validation set.
        train_samples, validation_samples = train_test_split(
                samples,
                test_size=0.2
        )

        train_generator = gen(train_samples, batch_size=batch_size)
        validation_generator = gen(validation_samples, batch_size=batch_size)

        self.model.compile(loss='mse', optimizer='adam')
        self.training_history = self.model.fit_generator(
            train_generator,
            steps_per_epoch=np.ceil(len(train_samples) / batch_size),
            validation_data=validation_generator,
            validation_steps=np.ceil(len(validation_samples) / batch_size),
            epochs=epochs,
            verbose=1
        )

    # Plot training loss and validation loss
    def plot(self):
        if self.training_history:
            plt.plot(self.training_history.history['loss'])
            plt.plot(self.training_history.history['val_loss'])
            plt.title('model mean squared error loss')
            plt.ylabel('mean squared error loss')
            plt.xlabel('epoch')
            plt.legend(['training set', 'validation set'], loc='upper right')
            plt.show()
        else:
            print('No training history to plot.')

    # Dump the trained model to a file
    def save(self, filename):
        self.model.save(filename)

    # Load the model from a stored file
    def load(self, filename):
        self.model.load_weights(filename)

    # Getter method for the model
    def get(self):
        return self.model
