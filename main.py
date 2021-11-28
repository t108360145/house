import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
from keras.models import Sequential
from keras.layers import *
from sklearn import preprocessing
import keras
import keras.callbacks



meta_data = pd.read_csv("data/metadata.csv")
test_data=pd.read_csv("data/test-v3.csv")
train_data = pd.read_csv("data/train-v3.csv")
valid_data = pd.read_csv("data/valid-v3.csv")
X_test=test_data.drop(['id'],axis=1).values
X_train =train_data.drop(['price','id'],axis=1).values
Y_train=train_data['price'].values
X_valid=valid_data.drop(['price','id'],axis=1).values
Y_valid=valid_data['price'].values

scaler=preprocessing.StandardScaler().fit(X_train)
X_train=preprocessing.scale(X_train)
X_valid=scaler.transform(X_valid)
X_test=scaler.transform(X_test)



def plotLearningCurves(history):
    df = pd.DataFrame(history.history)
    df.plot(figsize=(8, 5))
    plt.grid(True)  # 顯示網格
    # plt.gca().set_ylim(0, 1)
    plt.show()

def train():
    model=Sequential()
    model.add(Dense(units=80,activation="relu",kernel_initializer="normal", input_dim=X_train.shape[1]))
    model.add(Dense(units=70,kernel_initializer="normal"))
    model.add(Dense(units=60,kernel_initializer="normal"))
    model.add(Dense(units=40,kernel_initializer="normal"))
    model.add(Dense(units=30,kernel_initializer="normal"))
    model.add(Dense(units=20,kernel_initializer="normal"))
    model.add(Dense(units=10,kernel_initializer="normal"))
    model.add(Dense(units=5,kernel_initializer="normal"))
    model.add(Dense(units=1,kernel_initializer="normal"))
    model.compile(loss="MAE", optimizer="adam")

    model.summary()
    callbacks=[keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)]

    history=model.fit(X_train,Y_train,validation_data=(X_valid,Y_valid),epochs=150,callbacks=callbacks)

    plotLearningCurves(history)
    model.save("housing_model")


def test():
    model = tensorflow.keras.models.load_model('housing_model')

    result=model.predict(X_test)

    result=pd.DataFrame(result,columns=['price'])
    result.index+=1
    result.index.name="id"
    result.to_csv('result.csv')

train()