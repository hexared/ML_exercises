'''
Given the height of Father, we will predict the height of the son.
'''

# math calculation
import numpy as np;
# dataset handler
import pandas as pd;
# graph plotting
from matpltlib import pyplot as plt;

# linear regression lib
from sklearn.linear_model import LinearRegression;

# import the dataset
df = pd.read_csv(sys.argv, delim_withespace=True);

'''
Dataset example:

    | Father |  Son   |
    |  70.4  |  65.2  |
    |  72.3  |  70.1  |
    |  73.0  |  68.8  |
    |  69.4  |  71.2  |
'''

# prepare for training
x_train = df['Father'].values[:, np.newaxis];
y_train = df['Son'].values;

'''
We're assigning the 'Father' feature values to x and the 'Son' feture values to y:
        X        Y
    | Father |  Son   |
    |  70.4  |  65.2  |
    |  72.3  |  70.1  |
    |  73.0  |  68.8  |
    |  69.4  |  71.2  |

With the 'np.newaxis' we're morphing the n-dimensional array in an (n+1)-dimensional array, in this particular case,
in a 2 dimensional Column vector (because of the [:, np.newaxis]).
For a Row vector we sould use [np.newaxis, :].
'''

# choosing the training model
lm = LinearRegression();

# train the dataset
lm.fit(x_train, y_train);

# prepare the test dataset
x_test = [[72.8],[61.1],[67.4],[70.2],[75.6],[60.2],[65.3],[59.2]];

# test
predictions = lm.predict(x_test);
print predictions;

# Plotting
plt.scatter(x_train, y_train, color='red');

# plot the best fit line
plt.plot(x_test, predictions, color='black', linewidth=3);
plt.xlabel('Father height(in)');
plt.ylabel('Son height(in)');
plt.show();