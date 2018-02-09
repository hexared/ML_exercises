'''
the dataset consists in a series of data about patients who had an undergone surgery for breast cancer.
Given the details, we want to know if the patient survived or not.
'''
import numpy as np;
import panda as pd;
from matplotlib import pyplot as plt;

from sklearn.naive_bayes import GaussianNB

# import dataset
df = pd.read_csv(sys.argv);

'''
Dataset example:

|  age  |  year  |  nodes  |  survived  |
|   30  |   64   |    1    |      1     |
|   30  |   62   |    3    |      1     |
|   30  |   65   |    0    |      1     |
|   31  |   69   |    2    |      1     |
|   31  |   64   |    4    |      1     |

'''

# plotting the classes to see the dependencies of each feature

plt.xlabel('Feat');
plt.ylabel('Surv');

X = df.loc[:, 'Age'];
Y = df.loc[:, 'Survived'];
plt.scatter(X, Y, color='blue', label='Age');

X = df.loc[:, 'Year'];
Y = df.loc[:, 'Survived'];
plt.scatter(X, Y, color='green', label='Year');

X = df.loc[:, 'Nodes'];
Y = df.loc[:, 'Survived'];
plt.scatter(X, Y, color='red', label='Nodes');

plt.legend(loc=4, prop={'size': 7});
plt.show();

# training
clf=GaussianNB();
x_train = df.loc[:, 'Age', 'Nodes'];
y_train = df.loc[:, 'Survived'];

clf.fit(x_train, y_train);

# test
prediction = clf.predict([[12, 70, 12], [13, 20, 13]]);

print prediction;