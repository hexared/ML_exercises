'''
given car features we need to predict the clas of the car
'''
import numpy as np;
import panda as pd;
from sklearn.tree import DecisionTreeClassifier;

# import and read the csv
df = pd.read_csv(sys.argv);

'''
Dataset example:

| buyng | maint | doors | persons | lug_boot | safety | values | 
|   4   |   4   |   2   |    2    |     1    |   1    | unacc  |
|   4   |   4   |   2   |    2    |     1    |   2    | unacc  |
|   4   |   4   |   2   |    2    |     1    |   3    | unacc  |
|   4   |   4   |   2   |    2    |     2    |   1    | unacc  |
|   4   |   4   |   2   |    2    |     2    |   2    | unacc  |

''' 

# prepare the training set 
x_train = df.loc[:, 'buyng':'safety'];
y_train = df.loc[:, 'values'];

# training
tree.fit(x_train, y_train);

# test
prediction = tree.predict([[4, 3, 2, 1, 2, 3]])

