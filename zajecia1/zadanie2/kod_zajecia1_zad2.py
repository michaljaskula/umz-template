
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

dataset = pd.read_csv('C:\\Users\\mjaskula\\Desktop\\umz-template\\zajecia1\\zadanie2\\train\\train.tsv', sep='\t', names= ['price', 'isNew', 'rooms', 'floor', 'location', 'sqrmeters'])

Y = dataset['price']
X = dataset['sqrmeters']


X.reshape(len(X),1)
Y.reshape(len(Y),1)

X_train = X
X_test = X
Y_train = Y
Y_test = Y


plt.scatter(X_test, Y_test,  color='blue')
plt.title('Model')
plt.xlabel('Metraz')
plt.ylabel('Cena')
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3)

dataset2=pd.read_csv('C:\\Users\\mjaskula\\Desktop\\umz-template\\zajecia1\\zadanie2\\dev-0\\in.tsv', sep='\t', names=['isNew', 'rooms', 'floor', 'location', 'sqrmeters'])
x_dev=pd.DataFrame(dataset2, columns=['sqrmeters'])
y_dev_predict=reg.predict(x_dev)
pd.DataFrame(y_dev_predict).to_csv('out.tsv', sep='\t', index=False, header=False)

dataset3=pd.read_csv('C:\\Users\\mjaskula\\Desktop\\umz-template\\zajecia1\\zadanie2\\test-A\\in.tsv', sep='\t', names=['isNew', 'rooms', 'floor', 'location', 'sqrmeters'])
x_dev=pd.DataFrame(dataset3, columns=['sqrmeters'])
x_test=pd.DataFrame(dataset3, columns=['sqrmeters'])
y_test_predict=reg.predict(x_test)
pd.DataFrame(y_test_predict).to_csv('out.tsv', sep='\t', index=False, header=False)

