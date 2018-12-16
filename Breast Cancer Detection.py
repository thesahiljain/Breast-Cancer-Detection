import sys
import numpy
import matplotlib
import pandas
import sklearn

import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id',  'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion', 'single_epithelial_size', 
        'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses',  'class']
df = pd.read_csv(url, names=names)

# Preprocessing
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

print(df.axes)
print(df.shape)

# Visualization
print(df.loc[5])
print(df.describe())

# Histogram for each variable
df.hist(figsize = (10, 10))
plt.show()

# Scatter plot matrix
scatter_matrix(df, figsize = (18, 18))
plt.show()

# Create X and Y datasets for training
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

# Specific testing options
seed = 8
scoring = 'accuracy'

# Training models
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
models.append(('SVM', SVC()))

# Evaluate each models
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '{}: {:.4f} ({:.4f})'.format(name, cv_results.mean(), cv_results.std())
    print(msg)

# Make predictions
for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

# SVM customized case
clf = SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example = np.array([[4, 2, 1, 3, 1, 2, 2, 1, 3]])
example = example.reshape(len(example), -1)
prediction = clf.predict(example)
print(prediction)
