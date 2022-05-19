from sklearn.model_selection import train_test_split
from sklearn import metrics, dummy
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from scipy.cluster.hierarchy import dendrogram

from my_functions.common_functions import *
from matplotlib import cm

import timeit


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

        sample = np.random.choice(data_size, size=sample_size, replace=False)
        data = data.iloc[sample]

    cols = data.columns.tolist()
    cols.remove(Y)

    X = data[cols]
    Ycol = Y

    if scale_y:
        Y = data[[Ycol]]
        if scale == 'std':
            Y = Std_Scaled(Y)[:, 0]
        if scale == 'min-max':
            Y = MinMax_Scaled(Y)[:, 0]
        if scale == 'robust':
            Y = Robust_Scaled(Y)[:, 0]
        if scale == 'power':
            Y = PowerTransformer_Scaled(Y)[:, 0]
    else:
        Y = data[Y].values

    # Data Split 5 Trainning sets
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, Y, train_size=train_size, random_state=random_state, shuffle=shuffle)

    # Data Scaling. Training and Test sets scaled one after another to avoid data leakage
    if scale == 'std':
        xtrain = Std_Scaled(xtrain)
        xtest = Std_Scaled(xtest)
    if scale == 'min-max':
        xtrain = MinMax_Scaled(xtrain)
        xtest = MinMax_Scaled(xtest)
    if scale == 'robust':
        xtrain = Robust_Scaled(xtrain)
        xtest = Robust_Scaled(xtest)
    if scale == 'power':
        xtrain = PowerTransformer_Scaled(xtrain)
        xtest = PowerTransformer_Scaled(xtest)

    return xtrain, xtest, ytrain, ytest


def train(X_train, y_train, X_test, y_test, model_name, model, Perfs, scores=None, hue=None):
    print('--------------------', model_name, '--------------------')

    start_time = timeit.default_timer()
    # Entraînement
    model.fit(X_train, y_train)
    # Prédiction sur le jeu de test
    y_pred = model.predict(X_test)
    elapsed = timeit.default_timer() - start_time

    # Evaluatation
    i = len(Perfs)
    Perfs.loc[i, 'Model'] = model_name
    if scores == None:
        scores = ['RMSE', 'R²']
    if 'RMSE' in scores:
        RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        Perfs.loc[i, 'RMSE'] = RMSE
    if 'RMSLE' in scores:
        y_pred[y_pred < 0] = 0
        RMSLE = np.sqrt(metrics.mean_squared_log_error(y_test, y_pred))
        Perfs.loc[i, 'RMSLE'] = RMSLE
    if 'R²' in scores:
        R2 = np.abs(metrics.r2_score(y_test, y_pred))
        if R2 > 1:
            R2 = 0
        Perfs.loc[i, 'R²'] = R2
    Perfs.loc[i, 'Time(s)'] = elapsed
    if hue != None:
        Perfs.loc[i, 'hue'] = hue
    # Convertion de type
    for score in scores:
        Perfs[score] = Perfs[score].astype('float')
    print(Perfs.loc[i])

    # afficher les prédictions
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=y_test, y=y_pred, color='coral',
                    s=20, label='Actual Model', ax=ax)
    sns.lineplot(x=[min(y_test), max(y_test)], y=[
                 min(y_test), max(y_test)], color='gray', label='Ideal Model', ax=ax)
    # étiqueter les axes et le graphique
    plt.xlabel('Target', fontsize=16)
    plt.ylabel('Prediction', fontsize=16)
    plt.title(
        model_name + ' (R²=' + str(round(Perfs.loc[i, 'R²'], 3)) + ')', fontsize=20)
    plt.show()
    return Perfs, model


def train_cv(X_train, y_train, X_test, y_test, model_name, model, Perfs, param_grid, score_cv='neg_mean_squared_log_error', cv=5, scores=None, hue=None):
    # Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
    cv_model = RandomizedSearchCV(
        model,  # un classifieur kNN
        param_grid,     # hyperparamètres à tester
        cv=cv,           # nombre de folds de validation croisée
        scoring=score_cv,   # score à optimiser
        n_jobs=-1
    )

    perf, model = train(X_train, y_train, X_test, y_test,
                        model_name, cv_model, Perfs, scores, hue)
    print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:",
          cv_model.best_params_)
    return perf, cv_model.best_estimator_


def display_scores(Perfs, y=None, yloc=0.92, hue=None):
    Perfs = Perfs.copy()
    scores = Perfs.columns.tolist()
    scores.remove('Model')
    scores.remove('hue')
    if hue != None:
        Perfs = renameCol(Perfs.copy(), 'hue', hue)
    Perfs[colsOfType(Perfs)] = round(Perfs[colsOfType(Perfs)], 3)

    nb_line = len(colsOfType(Perfs))
    nb_col = 1
    height = nb_line * (3.3 if hue == None else 3)
    width = Perfs.shape[0] * (1.5 if hue == None else 0.7)
    wspace = 0.05
    hspace = 0.3

    # Préparation de l'affichage des graphiques sur deux colonnes : une pour les histogrammes et une pour les boxplots
    fig, axes = plt.subplots(nb_line, nb_col, figsize=(
        width, height), sharex=False, sharey=False)
    # ajuster l'espace entre les graphiques.
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    fig.suptitle('Evaluation' + ((' ('+y+')') if y != None and y != ''), x=0.5, y=yloc, fontsize=24,
                 horizontalalignment='center')  # Titre globale de la figure

    for i in range(0, len(scores)):
        ax = axes[i]
        sns.barplot(data=Perfs, ax=ax, x=Perfs['Model'], y=scores[i], hue=hue)
        for i in ax.containers:
            ax.bar_label(i,)
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=(8 if hue == None else 5))
        ax.set_xlabel('', fontsize=0)
    plt.show()


def features_importances(X, y, X_test, y_test, model):
    feature_names = X.columns.tolist()
    feature_names.remove(y)

    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1
    )
    forest_importances = pd.DataFrame(
        result.importances_mean, index=feature_names, columns=['RMAD'])
    forest_importances['RMAD'] = forest_importances['RMAD'].apply(
        lambda x: np.sqrt(x) if x >= 0 else 0)

    fig, ax = plt.subplots(figsize=(10, int(len(feature_names))*0.4))
    sns.barplot(data=forest_importances, y=feature_names,
                x='RMAD', orient='h', ax=ax)
    for i in ax.containers:
        ax.bar_label(i,)
    ax.set_title(
        "Feature importances (" + y + ")", fontsize=20)
    ax.set_xlabel("Root Mean Accuracy Decrease")
    plt.show()
