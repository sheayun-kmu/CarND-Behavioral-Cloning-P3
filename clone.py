import csv
import cv2
import numpy as np

datapaths = [
    '../lake_data',
    '../jungle_data',
]

images = []
measurements = []
for datapath in datapaths:
    lines = []
    with open(datapath + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = datapath + '/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, AveragePooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(
    filters=6,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(32, 32, 1))
)
model.add(AveragePooling2D())
model.add(Convolution2D(
    filters=16,
    kernel_size=(3, 3),
    activation='relu')
)
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
