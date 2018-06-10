import pandas as pd
import graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']

#
#clf = KNeighborsClassifier()
#clf = clf.fit(train_X, train_Y)

# neighbors=3
clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(train_X, train_Y)

# neighbors=5
clf = KNeighborsClassifier(n_neighbors=5)
clf = clf.fit(train_X, train_Y)

# neighbors=10
clf = KNeighborsClassifier(n_neighbors=10)
clf = clf.fit(train_X, train_Y)

# neighbors=15
clf = KNeighborsClassifier(n_neighbors=15)
clf = clf.fit(train_X, train_Y)

# neighbors=17
clf = KNeighborsClassifier(n_neighbors=17)
clf = clf.fit(train_X, train_Y)

print('TRAIN SET')
print(confusion_matrix(clf.predict(train_X), train_Y))
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
print(confusion_matrix(clf.predict(test_X), test_Y))
print(sum(clf.predict(test_X) == test_Y) / len(test_X))