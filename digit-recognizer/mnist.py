import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import Sequential
import keras

# read data from file.
data_x = pd.read_csv('data/train.csv')
data_y = data_x["label"]
data_x.drop(["label"], inplace=True, axis=1)
# split training data to training set and test set
x_train, x_test, y_train, y_test = train_test_split(
    data_x.values, data_y.values, test_size=0.1, random_state=42)
# reshape data based on tensorflow convention.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype(np.float32)
y_train = keras.utils.to_categorical(y_train, dtype=np.float32)
y_test = keras.utils.to_categorical(y_test, dtype=np.float32)

print(x_train.shape)
print(y_train.shape)

keras.backend.set_image_data_format('channels_last')

model = Sequential()
model.add(Conv2D(64, kernel_size=3, padding="same",
                 input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=3, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(256, kernel_size=3, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=258, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, 32, epochs=10)

scores = model.evaluate(x_test, y_test, verbose=10)
print(scores)

test = pd.read_csv('data/test.csv')
test_set = (test.values).reshape(-1, 28, 28 , 1).astype('float32')
res = model.predict(test_set)
res = np.argmax(res,axis = 1)
res = pd.Series(res, name="Label")
submission = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"),   res],axis = 1)
submission.to_csv("data/dnn_mnist.csv",index=False)
