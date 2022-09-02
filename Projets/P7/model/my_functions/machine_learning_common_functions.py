from sklearn.preprocessing import OrdinalEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, davies_bouldin_score, silhouette_score, calinski_harabasz_score, roc_curve, auc, RocCurveDisplay
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

from my_functions.common_functions import *
from my_functions.dimensionality_reduction_functions import PCA, DisplayManifold

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


def trainning_sets(data, Y, train_size=0.8, validation_size=0, random_state=0, shuffle=True, sampling_factor=1, scale=None, scale_y=False):
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

    # Data Split
    xtrain = xtest = xvalidation = ytrain = yvalidation = ytest = None
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, Y, train_size=train_size+validation_size, random_state=random_state, shuffle=shuffle, stratify=Y)
    if validation_size > 0:
        xtrain, xvalidation, ytrain, yvalidation = train_test_split(
            X, Y, train_size=train_size/(train_size+validation_size), random_state=random_state, shuffle=shuffle, stratify=Y)

    # Data Scaling. Training and Test sets scaled one after another to avoid data leakage
    if scale != None:
        if scale == 'std':
            scaler = Std_Scaled
        if scale == 'min-max':
            scaler = MinMax_Scaled
        if scale == 'robust':
            scaler = Robust_Scaled
        if scale == 'power':
            scaler = PowerTransformer_Scaled
        xtrain = scaler(xtrain)
        xtest = scaler(xtest)
        if validation_size > 0:
            xvalidation = scaler(xvalidation)

    if validation_size > 0:
        return xtrain, xvalidation, xtest, ytrain, yvalidation, ytest
    return xtrain, xtest, ytrain, ytest


def train(X_train, y_train, X_test, y_test, model_name, model, Perfs, scores=None, hue=None):
    print('--------------------', model_name, '--------------------')

    start_time = timeit.default_timer()
    # Entraînement
    model.fit(X_train, y_train)
    # Prédiction sur le jeu de test
    y_pred = model.predict(X_test)
    elapsed = timeit.default_timer() - start_time

    i = len(Perfs)
    Perfs.loc[i, 'Model'] = model_name

    # Evaluatation
    if scores == None:
        scores = ['RMSE', 'R²']
    if 'RMSE' in scores:
        RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
        Perfs.loc[i, 'RMSE'] = RMSE
    if 'RMSLE' in scores:
        y_pred[y_pred < 0] = 0
        RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
        Perfs.loc[i, 'RMSLE'] = RMSLE
    if 'R²' in scores:
        R2 = np.abs(r2_score(y_test, y_pred))
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
    if 'R²' in scores:
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
    if 'ROC' in scores:
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
        model,  # le modèle à Cross Valider
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


def display_scores(Perfs, yloc=0.92, y=None, hue=None, width=None, height=None, rotation=None, hspace=0.3):
    Perfs = Perfs.copy()
    scores = Perfs.columns.tolist()
    scores.remove('Model')
    scores.remove('hue')
    if hue != None:
        Perfs = renameCol(Perfs.copy(), 'hue', hue)
    Perfs[colsOfType(Perfs)] = round(Perfs[colsOfType(Perfs)], 3)

    nb_line = len(scores)
    nb_col = 1
    height = (nb_line * (3.3 if hue == None else 3)
              ) if height == None else height
    width = Perfs.shape[0] * \
        (1.5 if hue == None else 0.7) if width == None else width
    wspace = 0.05

    # Préparation de l'affichage des graphiques sur deux colonnes : une pour les histogrammes et une pour les boxplots
    fig, axes = plt.subplots(nb_line, nb_col, figsize=(
        width, height), sharex=False, sharey=False)
    # ajuster l'espace entre les graphiques.
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    fig.suptitle('Scores' + ((' ('+y+')') if y != None and y != '' else ''), x=0.5, y=yloc, fontsize=24,
                 horizontalalignment='center')  # Titre globale de la figure

    rotation = (8 if hue == None else 5) if rotation == None else rotation
    for i in range(0, len(scores)):
        ax = axes[i]
        sns.barplot(data=Perfs, ax=ax, x=Perfs['Model'], y=scores[i], hue=hue)
        if hue != None:
            ax.legend(loc='center right')
        for i in ax.containers:
            ax.bar_label(i,)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
        ax.set_xlabel('', fontsize=0)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
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
        "Feature importances" + ((' ('+y+')') if y != None and y != '' else ''), fontsize=20)
    ax.set_xlabel("Root Mean Accuracy Decrease")
    plt.show()


def OneHotEncoding(data, categories=None, new_names=None):
    # Encodage des colonnes categorielles
    if categories == None:
        categories = colsOfType(data, 'object')

    df_categories = data.copy()
    if new_names != None:
        # Changer le nom des colonnes
        for i in range(len(categories)):
            df_categories = renameCol(
                df_categories, categories[i], new_names[i])
        categories = new_names
    df_OHE = pd.get_dummies(df_categories, columns=categories)

    return df_OHE


def OrdinalEncoding(data, col):
    dft = data.groupby(col).count()
    data[col] = data[col].apply(lambda x: dft.loc[x][0])
    data[col] = OrdinalEncoder().fit_transform(
        np.array(data[col]).reshape(-1, 1))
    return data


def clustering(data, model, model_name, projection=None, Perfs=None, scale='std', scores=None, hue=None, display_components=3, categories=None):
    print('--------------------', model_name, '--------------------')
    # Nombre de clusters souhaités (taux  de compression)
    data = data[colsOfType(data)]  # garder les colonnes numériques uniquement

    # préparation des données pour le clustering
    X = data.values
    if scale == 'power':
        X_scaled = PowerTransformer_Scaled(X)
    if scale == 'std':
        X_scaled = Std_Scaled(X)
    if scale == 'min-max':
        X_scaled = MinMax_Scaled(X)
    else:
        X_scaled = X

    # Clustering
    start_time = timeit.default_timer()
    if categories == None:
        model = model.fit(X_scaled)
    else:
        model = model.fit(X, categorical=categories)
    elapsed = timeit.default_timer() - start_time
    labels = model.labels_

    stop = False
    try:
        if Perfs == None:
            stop = True
    except:
        pass
    if not stop:
        # Evaluatation
        i = len(Perfs)
        Perfs.loc[i, 'Model'] = model_name
        if scores == None:
            scores = ['Calinski-Harabasz(Var)',
                      'Davies-Bouldin(Sim)', 'Silhouette']
        if 'Calinski-Harabasz(Var)' in scores:
            Perfs.loc[i,
                      'Calinski-Harabasz(Var)'] = calinski_harabasz_score(X, labels)
        if 'Davies-Bouldin(Sim)' in scores:
            Perfs.loc[i,
                      'Davies-Bouldin(Sim)'] = davies_bouldin_score(X, labels)
        if 'Silhouette' in scores:
            Perfs.loc[i, 'Silhouette'] = silhouette_score(X, labels)
        Perfs.loc[i, 'Time(s)'] = elapsed
        if hue != None:
            Perfs.loc[i, 'hue'] = hue
        # Convertion de type
        for score in scores:
            Perfs[score] = Perfs[score].astype('float')
        print(Perfs.loc[i])

    if display_components > 1:
        try:
            if projection != None:
                DisplayManifold(
                    projection, c=model.labels_, title=model_name)
            else:
                DisplayClustering(data, labels, scale, display_components)
        except:
            DisplayManifold(
                projection, c=model.labels_, title=model_name)

    return model


def DisplayClustering(data, labels, scale='std', display_components=3):
    PCA(data, display_components, continuous_illustrative_var=labels, enable_display_circles=False,
        enable_display_scree_plot=False, scale=scale, s=1, display_3D=(display_components == 3))


def clusteringCenters(data, model, scale=None):
    cols = colsOfType(data)
    scaler = None
    if scale == 'power':
        _, scaler = PowerTransformer_Scaled(data[cols], return_scaler=True)
    if scale == 'std':
        _, scaler = Std_Scaled(data[cols], return_scaler=True)
    if scale == 'min-max':
        _, scaler = MinMax_Scaled(data[cols], return_scaler=True)

    centers = model.cluster_centers_
    if scaler != None:
        centers = scaler.inverse_transform(centers)
    else:
        cols = data.columns
    return pd.DataFrame(centers, columns=cols)


def centroids(data, labels):
    n_clusters = len(np.unique(labels))
    d = data.copy()
    d['Cluster'] = labels
    return d.groupby('Cluster').mean()


def OvA_ROC(data, cls, y):
    # Sélection des attributs
    XCols = colsOfType(data)
    # Binarize the output
    data = data.sort_values(by=[y])
    X = data.iloc[:, 1:-1]
    y = label_binarize(data[y], classes=data[y].unique().tolist())
    n_classes = y.shape[1]
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    X_train = Std_Scaled(X_train)
    X_test = Std_Scaled(X_test)

    # Sélection des scores pour l'évaluation des modèles
    score_cv = 'roc_auc_ovr_weighted'

    param_grid = {'random_state': np.arange(5), 'l2_regularization': np.logspace(
        -2, 2, 5), 'learning_rate': np.logspace(-4, -1, 4), 'max_depth': np.arange(1, 5)}
    cv_cls = RandomizedSearchCV(
        cls,
        param_grid,     # hyperparamètres à tester
        cv=5,           # nombre de folds de validation croisée
        scoring=score_cv,   # score à optimiser
        n_jobs=-1
    )

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(cv_cls, n_jobs=-1)
    y_predicted = classifier.fit(X_train, y_train).predict(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test.ravel(), y_predicted.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure(figsize=(7, 6))

    plt.plot(fpr["micro"], tpr["micro"], label="micro-average ROC curve (area = {0:0.2f})".format(
        roc_auc["micro"]), color="deeppink", linestyle=":", linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label="macro-average ROC curve (area = {0:0.2f})".format(
        roc_auc["macro"]), color="navy", linestyle=":", linewidth=4)

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc["micro"]


def plot_hist(hist, metric, metric_val):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Metric plot
    ax1.plot(hist.history[metric])
    ax1.plot(hist.history[metric_val])
    ax1.set_title("model "+metric)
    ax1.set_ylabel(metric)
    ax1.set_xlabel("epoch")
    ax1.legend(["train", "validation"], loc="upper right")

    # Loss Plot
    ax2.plot(hist.history['loss'])
    ax2.plot(hist.history['val_loss'])
    ax2.set_title("model loss")
    ax2.set_ylabel("loss")
    ax2.set_xlabel("epoch")
    ax2.legend(["train", "validation"], loc="upper right")


def AUC(y_test, y_predict):
    n_classes = len(np.unique(y_test))
    if n_classes == 2:
        # calculate inputs for the roc curve
        # fpr, tpr, thresholds = roc_curve(
        #     y_test, y_predict)  # get the best threshold
        # J = tpr - fpr
        # ix = np.argmax(J)
        # best_thresh = thresholds[ix]

        RocCurveDisplay.from_predictions(y_test, y_predict)
    else:
        plt.figure(figsize=(7, 6))
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test.ravel(), y_predict.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        lw = 2

        plt.plot(fpr["micro"], tpr["micro"], label="micro-average ROC curve (area = {0:0.2f})".format(
            roc_auc["micro"]), color="deeppink", linestyle=":", linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"], label="macro-average ROC curve (area = {0:0.2f})".format(
            roc_auc["macro"]), color="navy", linestyle=":", linewidth=4)

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(
            "Some extension of Receiver operating characteristic to multiclass")
        plt.legend(loc="lower right")
        plt.show()