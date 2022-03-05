from .common_functions import *

from sklearn.impute import KNNImputer
from sklearn import decomposition

from matplotlib.collections import LineCollection


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for (d1, d2) in axis_ranks:  # On affiche les n_comp/2 premiers plans factoriels,
        if d2 < n_comp:
            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7, 6))

            # détermination des limites du graphique
            if lims is not None:
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30:
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else:
                xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(
                    pcs[d1, :]), min(pcs[d2, :]), max(pcs[d2, :])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30:
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                           pcs[d1, :], pcs[d2, :],
                           angles='xy', scale_units='xy', scale=1, color='grey')
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                ax.add_collection(LineCollection(
                    lines, axes=ax, alpha=.1, color='black'))

            # affichage des noms des variables
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(x, y, labels[i], fontsize='14', ha='center',
                                 va='center', rotation=label_rotation, color='blue', alpha=0.5)

            # affichage du cercle
            circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(
                d1+1, round(100*pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(
                d2+1, round(100*pca.explained_variance_ratio_[d2], 1)))

            plt.title('Cercle des corrélations (F{} et F{})'.format(d1+1, d2+1))
            plt.show(block=False)


def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, continuous_illustrative_var=None, discrete_illustrative_var=None):
    if continuous_illustrative_var is not None:
        # Generate a custom diverging colormap
        cmap = sns.color_palette('RdYlGn_r', as_cmap=True)

    if discrete_illustrative_var is not None:
        title = discrete_illustrative_var.name

    for (d1, d2) in axis_ranks:  # On affiche les n_comp/2 premiers plans factoriels,
        if d2 < n_comp:
            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7, 6))

            # affichage des points
            if discrete_illustrative_var is None and continuous_illustrative_var is None:
                plt.scatter(X_projected[:, d1],
                            X_projected[:, d2], alpha=alpha)

            if continuous_illustrative_var is not None:
                plt.scatter(X_projected[:, d1],
                            X_projected[:, d2], c=continuous_illustrative_var, cmap=cmap, alpha=alpha)
                plt.colorbar(label=continuous_illustrative_var.name)

            if discrete_illustrative_var is not None:
                discrete_illustrative_var = np.array(
                    discrete_illustrative_var)
                for value in np.unique(discrete_illustrative_var):
                    selected = np.where(discrete_illustrative_var == value)
                    plt.scatter(
                        X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend(title=title)

            # affichage des labels des points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    plt.text(x, y, labels[i], fontsize='14',
                             ha='center', va='center')

            # détermination des limites du graphique
            plt.xlim([np.min(X_projected[:, [d1]])*1.1,
                     np.max(X_projected[:, [d1]])*1.1])
            plt.ylim([np.min(X_projected[:, [d2]])*1.1,
                     np.max(X_projected[:, [d2]])*1.1])

            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(
                d1+1, round(100*pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(
                d2+1, round(100*pca.explained_variance_ratio_[d2], 1)))

            plt.title(
                'Projection des individus (sur F{} et F{})'.format(d1+1, d2+1))
            plt.show(block=False)


def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(), c='red', marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title('Eboulis des valeurs propres')
    plt.show(block=False)


def PCA(data, n_comp=5, cols=None, alpha=1, continuous_illustrative_var=None, discrete_illustrative_var=None, enable_display_scree_plot=True, enable_display_circles=True, enable_display_factorial_planes=True):
    if cols is None:
        cols = data.columns.tolist()

    # choix du nombre de composantes à calculer
    n_comp = min(n_comp, len(cols))
    Fs = [(i, i+1) for i in range(0, n_comp, 2)]

    data_pca = data[cols]

    # préparation des données pour l'ACP
    if data_pca.isnull().sum().sum() > 0:
        knn_imputer = KNNImputer(n_neighbors=max(
            10, int(len(data_pca)*0.1)), weights='distance')
        knn_imputer_transfom = knn_imputer.fit_transform(data_pca)
        data_pca = pd.DataFrame(knn_imputer_transfom,
                                columns=data_pca.columns, index=data_pca.index)

    X = data_pca.values
    names = data.index  # ou data.index pour avoir les intitulés
    features = data[cols].columns

    # Centrage et Réduction
    X_scaled = Std_Scaled(X)

    # Calcul des composantes principales
    pca = decomposition.PCA(n_components=n_comp)
    pca.fit(X_scaled)

    if enable_display_scree_plot:
        # Eboulis des valeurs propres
        display_scree_plot(pca)

    pcs = pca.components_
    if enable_display_circles:
        # Cercle des corrélations
        display_circles(pcs, n_comp, pca, Fs, labels=np.array(features))

    if enable_display_factorial_planes:
        # Projection des individus
        X_projected = pca.transform(X_scaled)
        # , labels = np.array(names))
        display_factorial_planes(X_projected, n_comp, pca,
                                 Fs, alpha=alpha, continuous_illustrative_var=continuous_illustrative_var, discrete_illustrative_var=discrete_illustrative_var)

    return pcs


def PCA_Compression(data, components, cols=None, prefix='comp'):
    if cols == None:
        cols = data.columns.tolist()

    compressed_data = {}
    for i in range(len(components)):
        compressed_data[prefix+str(i+1)] = sum([data[cols[j]]
                                                * components[i][j] for j in range(len(components[i]))])

    compressed_data = pd.DataFrame(compressed_data, index=data.index)
    return compressed_data
