import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import Sequential
import keras
import pandas as pd

training_data = pd.read_csv('data/train.csv')
print(training_data.head(5))
data_y = training_data["Survived"]
data_x = training_data.drop(["Survived"], inplace=False, axis=1)
print(data_y.head(5))
print(data_x.head(5))

# Remove unrelated information
for ui in ["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]:
    data_x.drop([ui], inplace=True, axis=1)

data_x["Sex"] = (data_x["Sex"] == "male").astype(pd.np.float32)

train_x = data_x.values
train_x = train_x.astype(pd.np.float32)
train_y = data_y.values.astype(pd.np.float32)
train_y = keras.utils.to_categorical(train_y, dtype=pd.np.float32)

model = keras.Sequential()
model.add(Dense(units=64, input_shape=(6,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, 16, epochs=10)

# accuracy ~ 60%, too low, deprecate this method.
