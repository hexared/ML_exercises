import numpy as np;
from matplotlib import pyplot as plt;
from sklearn import svm;
from sklearn.datasets import make_circles;

# create our dataset
df, value = make_circles(n_samples=500, noise=.05, factor=.5);

# plot dataset
plt.scatter(df[:, 0], df[:, 1], c=value);
plt.show();

# training 
x_train = df[:, 0];
y_train = df[:, 1];
z_train = x_train**2 + y**2;

kernals = ['linear', 'poly', 'rbf'];
training_set = np.c_[x_train, y_train];

# train and predict
for kernal in kernals:
    clf = svm.SVC(kernel=kernal.gamma=2);

    # actual train
    clf.fit(training_set, value);

    # test
    prediction = clf.predict([[-0.4, -0.4]]);

    # plotting points, lines and nearest vector
    x = training_set;
    y = value;
    x0 = x_train[np.where(y==0)];
    x1 = y_train[np.where(y==1)];
    plt.figure();

    x_min = x_train[:, 0].min();
    x_max = x_train[:, 0].max();
    y_min = x_train[:, 1].min();
    y_max = x_train[:, 1].max();

    XX, YY = np.magrid[x_min:x_max:200j, y_min:y_max:200j];
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]);

    # put results in a color plot
    Z = Z.reshape(XX.shape);

    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired);

    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'), levels=[-.5, 0, .5]);

    scatter(x0[:, 0], x0[:, 1], c='r', s=50);
    scatter(x1[:, 0], x1[:, 1], c='b', s=50);

    title = ('SVC with {} kernal').format(kernal);
    plt.title(title);
    plt.show();



