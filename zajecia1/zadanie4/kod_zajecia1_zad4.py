
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


dataset = pd.read_csv('C:\\Users\\mjaskula\\Desktop\\umz-template\\zajecia1\\zadanie4\\train\\in.tsv', sep='\t', names= ['price', 'mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])

dataset.describe()

reg = linear_model.LinearRegression()

reg.fit(pd.DataFrame(dataset, columns=['mileage', 'year', 'engineCapacity']), dataset['price'])

print(reg.coef_)
print(reg.intercept_)


sns.regplot(y=dataset["price"], x=dataset["year"]); plt.show()
sns.regplot(y=dataset["year"], x=dataset["mileage"]); plt.show()
sns.regplot(y=dataset["engineCapacity"], x=dataset["mileage"]); plt.show()


dataset2=pd.read_csv('C:\\Users\\mjaskula\\Desktop\\umz-template\\zajecia1\\zadanie4\\dev-0\\in.tsv', sep='\t', names= ['price', 'mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])
x_dev=pd.DataFrame(dataset2, columns=['mileage', 'year', 'engineCapacity'])
y_dev_predict=reg.predict(x_dev)
pd.DataFrame(y_dev_predict).to_csv('out.tsv', sep='\t', index=False, header=False)

dataset3=pd.read_csv('C:\\Users\\mjaskula\\Desktop\\umz-template\\zajecia1\\zadanie4\\test-A\\in.tsv', sep='\t', names= ['price', 'mileage', 'year', 'brand', 'engingeType', 'engineCapacity'])
x_test=pd.DataFrame(dataset2, columns=['mileage', 'year', 'engineCapacity'])
y_test_predict=reg.predict(x_test)
pd.DataFrame(y_test_predict).to_csv('out.tsv', sep='\t', index=False, header=False)
