import xgboost as xgb
import pandas as pd

training_data = pd.read_csv('data/train.csv')
print(training_data.head(5))
data_y = training_data["SalePrice"]
data_x = training_data.drop(["SalePrice"], inplace=False, axis=1)
print(data_x.head(5))

test_data = pd.read_csv('data/test.csv')

data_x.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)
data_y.fillna(0, inplace=True)

dplst = []
def data_cleaning(trd, ted):
    for ss in list(trd):
        if(trd[ss].dtypes == pd.np.object):
            # print(trd[ss].dtypes)
            collect = trd[ss].unique()
            collect = pd.np.concatenate((collect, ted[ss].unique()))
            # print(collect)
            collect = list(set(collect))
            print(ss)
            print(collect)
            # print(collect)
            trd[ss] = trd[ss].astype('category', categories=collect)
            ted[ss] = ted[ss].astype('category', categories=collect)
            trd = pd.concat([trd, pd.get_dummies(trd[ss],prefix=ss)], axis=1)
            # trd.drop([ss], inplace=True, axis=1)
            ted = pd.concat([ted, pd.get_dummies(ted[ss],prefix=ss)], axis=1)
            # ted.drop([ss], inplace=True, axis=1)
            dplst.append(ss)
    trd.drop(dplst, inplace=True, axis=1)
    ted.drop(dplst, inplace=True, axis=1)
    return trd, ted

data_x, test_data = data_cleaning(data_x, test_data)

dtrain = xgb.DMatrix(data_x.values, label=data_y.values)

dtest = xgb.DMatrix(test_data.values)

xg_params = {"eta": 0.3, "silent":False}

num_round = 100
bst = xgb.train(xg_params, dtrain, num_round)
res = bst.predict(dtest, ntree_limit=num_round)

res = pd.Series(res, name="SalePrice")
submission = pd.concat([pd.Series(range(1461, 2920) ,name = "Id"), res],axis = 1)
submission.to_csv("data/xgb_res.csv",index=False)
