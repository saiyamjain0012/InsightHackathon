# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:28:23 2019

@author: Saiyam_Jain
"""
#importing the required libraries
import pandas as pd

#importing the data
data=pd.read_csv("mldata.csv")

training=data.drop("Unnamed: 0",axis=1)
training=training.iloc[:,1:]


import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout
from sklearn import preprocessing
import numpy as np

from sklearn import preprocessing, model_selection

x=training.iloc[:,:-1]
y=training.iloc[:,-1]

#y_dummy=pd.get_dummies(df['Priority'])



from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)



x['Profit_Margin'].loc[x.Profit_Margin == 'Low'] = 0
x['Profit_Margin'].loc[x.Profit_Margin == 'Medium'] = 1
x['Profit_Margin'].loc[x.Profit_Margin == 'High'] = 2


x['Marketing'].loc[x.Marketing == 'low'] = 2
x['Marketing'].loc[x.Marketing == 'medium'] = 1
x['Marketing'].loc[x.Marketing == 'good'] = 0

x_dummies = pd.get_dummies(x['Type'])

x=x.join(x_dummies)
x=x.drop("Type",axis=1)

x=x.drop("Software",axis=1)

x_dummies2=pd.get_dummies(x['Country'])


x=x.join(x_dummies2)
x=x.drop("Country",axis=1)
x=x.drop("NORTHAMR",axis=1)

x_dummies3 = pd.get_dummies(x['Market_Competition'])
x=x.join(x_dummies3)
x=x.drop("Market_Competition",axis=1)s

x=x.drop("Bad",axis=1)


train_x, test_x, train_y, test_y = model_selection.train_test_split(x,y,test_size = 0.1, random_state = 0)




classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=8, init='uniform', activation='relu', input_dim=7))
classifier.add(Dense(output_dim=8, init='uniform', activation='relu'))
#classifier.add(Dense(output_dim=8, init='uniform', activation='relu'))
#classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=3, init='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train, y_train, batch_size=150, nb_epoch=1500)


scores = classifier.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


que=x_train.head(4)

for index, row in que.iterrows():
    #temp=row.to_frame()
    temp=pd.DataFrame([row])
    order, pri=pred(temp)
    print(pri+" & Code:"+str(order))