
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

dataset = pd.read_csv('C:\\Users\\mjaskula\\Desktop\\umz-template\\zajecia1\\zadanie3\\train\\train.tsv', sep='\t', names= ['price', 'isNew', 'rooms', 'floor', 'location', 'sqrMetres'])

dataset.describe()

reg = linear_model.LinearRegression()

reg.fit(pd.DataFrame(dataset, columns=['sqrMetres', 'floor', 'rooms', 'isNew']), dataset['price'])

print(reg.coef_)
print(reg.intercept_)

sns.regplot(y=dataset["price"], x=dataset["rooms"]); plt.show()
sns.regplot(y=dataset["price"], x=dataset["isNew"]); plt.show()
sns.regplot(y=dataset["price"], x=dataset["floor"]); plt.show()
sns.regplot(y=dataset["price"], x=dataset["sqrMetres"]); plt.show()



dataset2=pd.read_csv('C:\\Users\\mjaskula\\Desktop\\umz-template\\zajecia1\\zadanie3\\dev-0\\in.tsv', sep='\t', names=['isNew', 'rooms', 'floor', 'location', 'sqrMetres'])
x_dev=pd.DataFrame(dataset2, columns=['sqrMetres', 'floor', 'rooms', 'isNew'])
y_dev_predict=reg.predict(x_dev)
pd.DataFrame(y_dev_predict).to_csv('out.tsv', sep='\t', index=False, header=False)


dataset3=pd.read_csv('C:\\Users\\mjaskula\\Desktop\\umz-template\\zajecia1\\zadanie3\\test-A\\in.tsv', sep='\t', names=['isNew', 'rooms', 'floor', 'location', 'sqrMetres'])
x_test=pd.DataFrame(dataset3, columns=['sqrMetres', 'floor', 'rooms', 'isNew'])
y_test_predict=reg.predict(x_test)
pd.DataFrame(y_test_predict).to_csv('out.tsv', sep='\t', index=False, header=False)

