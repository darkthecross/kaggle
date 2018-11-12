import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import Sequential
import keras
import pandas as pd
import math
import xgboost as xgb


def data_cleaning(data_x):
    # convert obj to one-hot
    for c in ["A", "B", "C", "D", "E", "F", "G", "T"]:
        c_series = pd.Series([0] * data_x.shape[0], name=c)
        for i in range(data_x.shape[0]):
            if not type(data_x["Cabin"][i]) is float:
                c_series[i] = int(c in data_x["Cabin"][i])
        data_x = pd.concat([data_x, c_series], axis=1)
    for c in ["S", "C", "Q"]:
        c_series = pd.Series([0] * data_x.shape[0], name="E" + c)
        for i in range(data_x.shape[0]):
            if not type(data_x["Embarked"][i]) is float:
                c_series[i] = int(c in data_x["Embarked"][i])
        data_x = pd.concat([data_x, c_series], axis=1)
    data_x["Sex"] = (data_x["Sex"] == "male").astype(pd.np.float32)
    # supress Nan
    data_x.fillna(0, inplace=True)
    # Remove unrelated information
    for ui in ["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]:
        data_x.drop([ui], inplace=True, axis=1)
    return data_x


training_data = pd.read_csv('data/train.csv')
print(training_data.head(5))
data_y = training_data["Survived"]
data_x = training_data.drop(["Survived"], inplace=False, axis=1)
# print(data_y)
# print(data_x)
data_x = data_cleaning(data_x)
# print(data_x)

testing_data = pd.read_csv('data/test.csv')
# print(testing_data.head(5))
# print(testing_data)
testing_data = data_cleaning(testing_data)
# print(testing_data)

train_x = data_x.values
train_x = train_x.astype(pd.np.float32)
train_y = data_y.values.astype(pd.np.float32)

dtrain = xgb.DMatrix(train_x, label=train_y)

dtest = xgb.DMatrix(testing_data.values)

xg_params = {"eta": 0.3, "silent":False, "objective":"binary:logistic"}

num_round = 50
bst = xgb.train(xg_params, dtrain, num_round)

res = bst.predict(dtest, ntree_limit=num_round)

print(res)

res[pd.np.where(res>=0.5)] = 1
res[pd.np.where(res<0.5)] = 0

res = res.astype(pd.np.int32)

res = pd.Series(res, name="Survived")

submission = pd.concat(
    [pd.Series(range(892, 1310), name="PassengerId"), res], axis=1)

print(submission)

submission.to_csv("data/xgb.csv",index=False)
submission.head(10)
# accuracy ~ 74%, move on to better methods.
