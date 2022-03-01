from .machine_learning_common_functions import trainning_sets
from sklearn import neighbors
import matplotlib.pyplot as plt


def knn_train(data, Y, k=10, sampling_factor=1):
    # Data Split 5 Trainning sets
    xtrain, xtest, ytrain, ytest = trainning_sets(data, Y, sampling_factor)
    # K-NN
    knn = neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn.fit(xtrain, ytrain)
    error = 100*(1 - knn.score(xtest, ytest))
    return knn, error


def best_knn(data, Y, k=(2, 30), sampling_factor=1, repeat_factor=10):
    errors = []
    best_knn = None
    best_error = None
    kmin = k[0]
    kmax = k[1]
    for k in range(kmin, kmax):
        local_errors = []
        for i in range(repeat_factor):
            knn = knn_train(data, Y, k, sampling_factor)
            local_errors.append(knn[1])
            if best_error == None or knn[1] < best_error:
                best_error = knn[1]
                best_knn = knn[0]
        errors.append(min(local_errors))
    plt.plot(range(kmin, kmax), errors, 'o-')
    plt.show()

    return errors.index(min(errors))+kmin, min(errors), best_knn, errors, range(kmin, kmax)
