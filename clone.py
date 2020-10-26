import csv
import cv2
import numpy as np

datapaths = [
    '/home/sheayun/Desktop/lake_data',
    # '/home/sheayun/Desktop/jungle_data',
]

def collect_samples(images, measurements, img_path, steering_angle):
    img = cv2.imread(img_path)
    images.append(img)
    measurements.append(steering_angle)
    images.append(cv2.flip(img, 1))
    measurements.append(-steering_angle)

images = []
measurements = []
offset = 0.2
for datapath in datapaths:
    lines = []
    with open(datapath + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    for line in lines:
        image_paths = [datapath + '/IMG/' + x.split('/')[-1] for x in line[:3]]
        center_img_path, left_img_path, right_img_path = image_paths
        center_steer = float(line[3])
        collect_samples(images, measurements, center_img_path, center_steer)
        collect_samples(images, measurements, left_img_path, center_steer + offset)
        collect_samples(images, measurements, right_img_path, center_steer - offset)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D

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
        lambda x: x / 255.0 - 0.5
    )
)
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
