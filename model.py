import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, Cropping2D
import matplotlib.pyplot as plt

datapaths = [
    '../lake_twolaps',
    '../jungle_twolaps',
    '../udacity_data',
    # '../lake_recovery',
    # '../jungle_recovery',
    '../lake_reverse',
    '../jungle_reverse',
    '../jungle_corner',
] 

# Collect samples from manual driving recordings.
steer_offset = 0.2
samples = []
train_flipped = True
for datapath in datapaths:
    with open(datapath + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            paths = [datapath + '/IMG/' + x.split('/')[-1] for x in line[:3]]
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

# A generator function to yield a subset of batch_size
def generator(samples, batch_size=32):
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

# Set batch size and generators for training set and validation set, resp.
batch_size = 32
epochs = 5
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

top_crop, bottom_crop = 70, 25
model = Sequential()
model.add(
    Cropping2D(
        cropping=((top_crop, bottom_crop), (0, 0)),
        input_shape=(160, 320, 3)
    )
)
model.add(
    Lambda(
        lambda x: x / 127.5 - 1.0
    )
)
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
training_history = model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(len(train_samples) / batch_size),
    validation_data=validation_generator,
    validation_steps=np.ceil(len(validation_samples) / batch_size),
    epochs=epochs,
    verbose=1
)

model.save('model.h5')

print(training_history.history.keys())

plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
