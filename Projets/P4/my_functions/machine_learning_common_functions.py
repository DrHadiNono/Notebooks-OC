from scipy.cluster.hierarchy import dendrogram
from sklearn.model_selection import train_test_split
from my_functions.common_functions import MinMax_Scaled, Std_Scaled
import numpy as np
import matplotlib.pyplot as plt


def plot_dendrogram(Z, names):
    plt.figure(figsize=(10, 25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels=names,
        orientation='left',
    )
    plt.show()


def trainning_sets(data, Y, train_size=0.8, random_state=None, shuffle=True, sampling_factor=1, scale=None, scale_y=False):
    if sampling_factor < 1:
        data_size = len(data)
        sample_size = int(data_size*sampling_factor)
        # print('Original data size:', data_size, 'Sample data size:', sample_size)

        sample = np.random.randint(data_size, size=sample_size)
        data = data.iloc[sample]

    cols = [col for col in data.columns.tolist() if col != Y]

    X = data[cols]
    Ycol = Y

    if scale_y:
        Y = data[[Ycol]]
        if scale == 'std':
            Y = Std_Scaled(Y)[:, 0]
        if scale == 'min-max':
            Y = MinMax_Scaled(Y)[:, 0]
    else:
        Y = data[Y].values

    # Data Split 5 Trainning sets
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, Y, train_size=train_size, random_state=random_state, shuffle=shuffle)

    # Data Scaling. Training and Test sets scaled one after another to avoid data leakage
    if scale == 'std':
        xtrain= Std_Scaled(xtrain)
        xtest= Std_Scaled(xtest)
    elif scale == 'min-max':
        xtrain= MinMax_Scaled(xtrain)
        xtest= MinMax_Scaled(xtest)

    return xtrain, xtest, ytrain, ytest
