from scipy.cluster.hierarchy import dendrogram
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def plot_dendrogram(Z, names):
    plt.figure(figsize=(10, 25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels=names,
        orientation="left",
    )
    plt.show()


def trainning_sets(data, Y, sampling_factor=1):
    data_size = len(data)
    sample_size = int(data_size*sampling_factor)
    # print('Original data size:', data_size, 'Sample data size:', sample_size)

    sample = np.random.randint(data_size, size=sample_size)
    sampled_data = data.iloc[sample]

    cols = [col for col in sampled_data.columns.tolist() if col != Y]
    Y = sampled_data[Y]
    X = sampled_data[cols]
    # Data Split 5 Trainning sets
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.8)
    return xtrain, xtest, ytrain, ytest
