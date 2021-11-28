import pandas as pd
import tensorflow
from sklearn import preprocessing


meta_data = pd.read_csv("metadata.csv")
test_data=pd.read_csv("test-v3.csv")
X_test=test_data.drop(['id'],axis=1).values
train_data = pd.read_csv("train-v3.csv")
X_train =train_data.drop(['price','id'],axis=1).values
Y_train=train_data['price'].values
valid_data = pd.read_csv("valid-v3.csv")
X_valid=valid_data.drop(['price','id'],axis=1).values
Y_valid=valid_data['price'].values

scaler=preprocessing.StandardScaler().fit(X_train)
X_train=preprocessing.scale(X_train)
X_valid=scaler.transform(X_valid)
X_test=scaler.transform(X_test)


def test():
    model = tensorflow.keras.models.load_model('housing_model')

    result=model.predict(X_test)

    result=pd.DataFrame(result,columns=['price'])
    result.index+=1
    result.index.name="id"
    result.to_csv('result.csv')
test()
