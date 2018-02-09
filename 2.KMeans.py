'''
Given the dataset related to eruptions, we'll cluster a particular eruption.
eruptions - eruption time(minutes)
waiting - waiting time to the next eruption(minutes)
'''
# math calculation
import numpy as np;
# dataset handler
import pandas as pd;
# graph plotting
from matpltlib import pyplot as plt;

# K-Means lib
from sklearn.cluster import KMeans;

# import the dataset
df = pd.read_csv(sys.argv);

'''
Dataset example:

    | Eruptions |  Waiting  |
    |   3.600   |    79     |
    |   1.800   |    54     |
    |   3.333   |    74     |
    |   2.283   |    62     |
'''

# choosing the model and assign the number of clusters
k = 2;
kmeans = KMeans(n_clusters=k);

# Train model
kmeans = kmeans.fit(df);

# array with the cluster number
labels = kmeans.labels_

# array of size k with the centroids coordinates
centroids = kmeans.cluster_centers_

# tests
x_test = [[4.671,67],[2.885,61],[1.666,90],[5.623,54],[2.678,80],[1.875,60]];

prediction = kmeans.predict(x_test);
print prediction;

# plotting
colors = ['blue', 'red', 'green', 'black'];
y = 0;
for x in labels:
    #plot points and assign colors to it
    plt.scatter(df.iloc[y,0], df.iloc[y,1], color=colors[x]);
    y+=1;
for x in range(k):
    # plot centroids
    lines = plt.plot(centroids[x,0], centroids[x,1], 'kx');
    # larger centroids
    plt.setp(lines.ms=15.0);
    plt.setp(lines.mew=2.0);

title = ('# of clusters (k)={}').format(k);
plt.title(title);
plt.xlabel('eruptions(min)');
plt.ylabel('waiting(min)');
plt.show();