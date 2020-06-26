import os
import sys 

import pandas as pd 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

os.chdir(os.path.dirname(os.path.abspath(__file__)))

churn_df = pd.read_csv("./data/Churn_Modelling.csv")

features = [col for col in churn_df.columns if col not in ('RowNumber', 'CustomerId' ,'Surname', 'Exited')]
target = 'Exited'

X = churn_df[features]
y = churn_df[target]

categoricalCols = [ 'Geography', 'Gender']
numericalCols = [col for col in X.columns if col not in ('HasCrCard', 'IsActiveMember', 'Gender', 'Geography')]
preprocess = make_column_transformer( (OneHotEncoder(), categoricalCols), (StandardScaler(), numericalCols), remainder = 'passthrough')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
X_train = preprocess.fit_transform(X_train)
X_test = preprocess.transform(X_test)

ann = Sequential()
ann.add(Dense(units = 6, activation='relu'))
ann.add(Dense(units = 12, activation='relu'))
ann.add(Dense(units = 1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

r = ann.fit(X_train, y_train, batch_size = 32, epochs=100, validation_split=0.2)
p = ann.predict(X_test)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('./output/ann_loss_by_epochs.png')

auc = roc_auc_score(y_test, p[:,0])

print("AUC on test set: {}".format(auc))