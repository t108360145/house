# *House Regression*

班級：電子三甲 學號：108360145 姓名：康彥濬

## 檔案介紹
data資料夾是老師Kaggle給的5個資料檔案

housing_model是訓練完成的模型

main.py是主程式

requirement程式引入的函數

result.csv是輸出的結果

test.py是用來輸出結果的程式

test.sh輸出用腳本，在git bash打`./test.sh`即可

train.sh訓練用腳本，在git bash打`./train.sh`即可



## 心得
我的專題有用到Yolo來辨識圖片，但是Yolo會自己架構模型跟輸出曲線等等，只要給他資料跟做一些基本的設定就可以了，而這是我第一次自己來架構模型，神經層數跟神經元數量該如何設定，我並不是很了解，只好慢慢摸索，測試神經元數量跟層數這兩者之間該如何取捨，成功將loss從十多萬降低到七萬，原本想要繼續優化模型，但礙於這兩周為期中考周，而我兩周後還有一個比賽，時間安排上比較緊湊，無法做出更好的模型來繳交。

## 程式介紹
首先我利用Pandas來讀取資料，把ID跟Price兩個標籤刪除後，再利用sklearing來做資料的前處理，而模型部分我使用keras，並嘗試讓輸出層的神經元數量至少為輸入層的一半，來改善loss的情況，所以總共用了八層，分別有70/60/40/30/20/10/5/1個神經元，損失函數為MAE，優化函數為adam，訓練過程利用callbacks指令來讓模型提前結束，模型訓練完成後再利用matplotlib劃出訓練的過程曲線。

## 程式碼

```code
//引入所使用的函數
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
from keras.models import Sequential
from keras.layers import *
from sklearn import preprocessing
import keras
import keras.callbacks


//從data資料夾讀取資料
meta_data = pd.read_csv("data/metadata.csv")
test_data=pd.read_csv("data/test-v3.csv")
X_test=test_data.drop(['id'],axis=1).values
train_data = pd.read_csv("data/train-v3.csv")
X_train =train_data.drop(['price','id'],axis=1).values
Y_train=train_data['price'].values
valid_data = pd.read_csv("data/valid-v3.csv")
X_valid=valid_data.drop(['price','id'],axis=1).values
Y_valid=valid_data['price'].values

//資料前處理，用StandlerScaler將平均值變為0，標準差為1
scaler=preprocessing.StandardScaler().fit(X_train)
X_train=preprocessing.scale(X_train)
X_valid=scaler.transform(X_valid)
X_test=scaler.transform(X_test)


//劃出訓練的曲線圖
def plotLearningCurves(history):
    df = pd.DataFrame(history.history)
    df.plot(figsize=(8, 5))
    plt.grid(True)  # 顯示網格
    # plt.gca().set_ylim(0, 1)
    plt.show()

//模型
def train():
    model=Sequential()
    //輸入層，神經元數量為80，激活函數為線性
    model.add(Dense(units=80,activation="relu",kernel_initializer="normal",input_dim=X_train.shape[1]))	
    model.add(Dense(units=70,kernel_initializer="normal"))
    model.add(Dense(units=60,kernel_initializer="normal"))
    model.add(Dense(units=40,kernel_initializer="normal"))
    model.add(Dense(units=30,kernel_initializer="normal"))
    model.add(Dense(units=20,kernel_initializer="normal"))
    model.add(Dense(units=10,kernel_initializer="normal"))
    model.add(Dense(units=5,kernel_initializer="normal"))
    model.add(Dense(units=1,kernel_initializer="normal"))
    model.compile(loss="MAE", optimizer="adam")

    model.summary()						//輸出模型摘要

    //設定訓練過程只要超過5次沒有變好就結束
    callbacks=[keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)]	

    history=model.fit(X_train,Y_train,validation_data=(X_valid,Y_valid),epochs=150,callbacks=callbacks)	//開始訓練，次數為150

    plotLearningCurves(history)					//劃出訓練曲線圖
    model.save("housing_model")					//儲存模型

//輸出結果
def test():
    model = tensorflow.keras.models.load_model('housing_model')	//讀取模型

    result=model.predict(X_test)				//預測結果

    result=pd.DataFrame(result,columns=['price'])		//把輸出的Excel調整成符合老師要的格式
    result.index+=1
    result.index.name="id"
    result.to_csv('result.csv')
```