
# coding: utf-8


import pandas as pd
import seaborn as sns
import os 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pylab


print('zbior treningowy')


rtrain = pd.read_csv('train/in.tsv', sep='\t', names=["rezultaty", "B", "C", "D", "E", "F","G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI"])

print(rtrain.describe())

print('rezultaty w % :', end='')
print(sum(rtrain.rezultaty=="g") / len(rtrain))
print('-' * 100)
lr = LogisticRegression()

X = pd.DataFrame(rtrain, columns=['M'])

lr.fit(X, rtrain.rezultaty)

print('True Positives: ', end ='')
TP=sum((lr.predict(X) == rtrain.rezultaty) & (lr.predict(X) == "g"))
print(TP)

print('True Negatives: ', end ='')
TN=sum((lr.predict(X) == rtrain.rezultaty) & (lr.predict(X) == "b"))
print(TN)

print('False Positives: ', end ='')
FP=sum((lr.predict(X) != rtrain.rezultaty) & (lr.predict(X) == "g"))
print(FP)

print('False Negatives: ', end ='')
FN=sum((lr.predict(X) != rtrain.rezultaty) & (lr.predict(X) == "b"))
print(FN)


print('czulosc na zbiorze treningowym:', end = '')
print(TP/(TP+FN))

print('swoistosc na zbiorze treningowym:', end = '')
print(TN/(TN+FP))

print('dokladnosc na zbiorze treningowym:', end = '')
print(str((TP + TN) / len(rtrain)))

print('-' * 100)

rdev = pd.read_csv('dev-0/in.tsv', sep='\t',names=["rezultaty", "B", "C", "D", "E", "F","G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI"])
rdev = pd.DataFrame(rdev,columns = ["M"])
rtest = pd.read_csv('test-A/in.tsv', sep='\t',names=["rezultaty", "B", "C", "D", "E", "F","G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI"])
rtest = pd.DataFrame(rtest,columns = ["M"])
rdev_expected = pd.read_csv('dev-0/expected.tsv', sep='\t', names=['r'])

print('rezultaty w % :', end='')
print(sum(rdev_expected.r=="g") / len(rdev_expected))

print('True Positives: ', end ='')
TP2=sum((lr.predict(rdev) == rdev_expected['r']) & (lr.predict(rdev) == "g"))
print(TP2)

print('True Negatives: ', end ='')
TN2=sum((lr.predict(rdev) == rdev_expected['r']) & (lr.predict(rdev) == "b"))
print(TN2)

print('False Positives: ', end ='')
FP2=sum((lr.predict(rdev) != rdev_expected['r']) & (lr.predict(rdev) == "g"))
print(FP2)

print('False Negatives: ', end ='')
FN2=sum((lr.predict(rdev) != rdev_expected['r']) & (lr.predict(rdev) == "b"))
print(FN2)

print('czulosc na zbiorze deweloperskim:', end = '')
print(TP2/(TP2+FN2))

print('swoistosc na zbiorze deweloperskim:', end = '')
print(TN2/(TN2+FP2))

print('dokladnosc na zbiorze deweloperskim:', end = '')
print(str((TP2 + TN2) / len(rdev)))

print('-'*100)

d = open('dev-0/out.tsv', 'w')
for i in range(0, len(lr.predict(rdev))):
    d.write(str(lr.predict(rdev)[i]))
    d.write('\n')
d.close()

t = open('test-A/out.tsv', 'w')
for i in range(0, len(lr.predict(rtest))):
    t.write(str(lr.predict(rtest)[i]))
    t.write('\n')
t.close()


rdev_expected["y"]=rdev_expected["r"].map( {'g': 0, 'b': 1} ).astype(float)
print(rdev_expected.head())
print('Wykres')
sns.regplot(x=rdev.M, y=rdev_expected.r, logistic=True, y_jitter=.1)
plt.show()
