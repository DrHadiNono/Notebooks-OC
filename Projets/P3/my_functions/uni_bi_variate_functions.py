# Librairies pour le traitement des données
from .common_functions import *
import pandas as pd
import numpy as np
import scipy.stats as st

# Librairies pour la visualisation de graphiques
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()  # Définir le style par défaut pour les graphiques


def histo_distribution(data, horizontal=True):
    continues = colsOfType(data, ['float32', 'float64'])
    quantitatives = continues + colsOfType(data, ['int32', 'int64'])

    # Nombre de variables quantitatives
    nb_quantitaves = len(quantitatives)

    # Liste de couleurs
    colors = ['purple', 'green', 'orange',
              'red', 'yellow', 'blue', 'cyan', 'brown']

    height = 0
    width = 0
    nb_line = 0
    nb_col = 0
    wspace = 0.05
    hspace = 0.2

    if horizontal:
        nb_line = 3
        nb_col = nb_quantitaves
        width = nb_quantitaves*9
        height = 3*5
        wspace = 0.07
    else:
        nb_line = nb_quantitaves
        nb_col = 3
        width = 3*8
        height = nb_quantitaves*5
        wspace = 0.025

    # Préparation de l'affichage des graphiques sur deux colonnes : une pour les histogrammes et une pour les boxplots
    fig, axes = plt.subplots(nb_line, nb_col, figsize=(
        width, height), sharex=False, sharey=False)
    # ajuster l'espace entre les graphiques.
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    # fig.suptitle('Distirbutions des variables quantitatives ', y=0.92,
    #              fontsize=26, horizontalalignment='center')  # Titre globale de la figure

    meanprops = {'marker': 'o', 'markeredgecolor': 'black',
                 'markerfacecolor': 'firebrick'}  # Marquage des moyennes en rouge

    for i in range(0, nb_quantitaves):
        data_quantitative = data[quantitatives[i]]
        index = -1

        # Afficher les histogrammes sur la première colonne
        index += 1
        ax = axes[index, i] if horizontal else axes[i, index]
        # Soustitre de colonne
        if i == 0 or horizontal:
            ax.set_title("histogramme", loc='center', y=1, fontsize=18)
        # Ajuster les classes des variables selon leurs types (continues ou discrètes) :
        bins = 'auto' if quantitatives[i] not in continues else 10
        sns.histplot(data_quantitative, ax=ax,
                     color=colors[i % len(colors)], kde=True, bins=bins)
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Afficher les boxplots sans outliers sur la deuxième colonne
        index += 1
        ax = axes[index, i] if horizontal else axes[i, index]
        # Soustitre de colonne
        if i == 0 or horizontal:
            ax.set_title("boxplot avec outliers",
                         loc='center', y=1, fontsize=18)
        sns.boxplot(x=data_quantitative, ax=ax, color=colors[i % len(colors)], orient='h', width=.3, showmeans=True,
                    meanprops=meanprops, showfliers=True)  # afficher avec outliers
        if not horizontal:
            ax.set_xlabel(quantitatives[i], fontsize=20)
        else:
            ax.set_xlabel('')

        # Afficher les boxplots avec outliers sur la deuxième colonne
        index += 1
        ax = axes[index, i] if horizontal else axes[i, index]
        # Soustitre de colonne
        if i == 0 or horizontal:
            ax.set_title("boxplot sans outliers",
                         loc='center', y=1, fontsize=18)
        sns.boxplot(x=data_quantitative, ax=ax, color=colors[i % len(colors)], orient='h', width=.3, showmeans=True,
                    meanprops=meanprops, showfliers=False)  # afficher sans outliers
        if horizontal:
            ax.set_xlabel(quantitatives[i], fontsize=20)
        else:
            ax.set_xlabel('')


def force_mesure(mesure, type):
    """ Renvoie un texte avec la force de la mesure selon les seuils """
    force_text = ''

    # Définition des seuils/intervalles de mesure
    seuils = {
        0: 'aucune', .2: 'faible', .4: 'moyenne', .6: 'forte', .8: 'très forte'}

    for seuil, force in seuils.items():
        if np.abs(mesure) >= seuil:
            force_text = force

    return force_text + ' ' + type


def afficher_correlations(data, variables, categorie=None):
    """ Calcul et affichage des corrélations linéaires entre les 'variables' """
    # Calcul des corrélations
    data = data[(variables+[categorie] if categorie !=
                 None else variables)].copy()
    correlations = data.corr()

    # Afficher les paires de dispersion
    hue_order = data[categorie].sort_values(
    ).unique() if categorie != None else None
    g = sns.PairGrid(data, hue=categorie, hue_order=hue_order)
    g.fig.suptitle('Corrélations linéaires des variables quantitatives ', y=1.05,
                   fontsize=24, horizontalalignment='center')  # Titre globale de la figure

    # Afficher sur la diagonale les dispersions, par rapport à la catégorie si donnée
    g.map_diag(sns.kdeplot, fill=True)

    # Afficher ailleurs les scatter plots avec les régressions linéaires, par rapport à la catégorie si donnée
    g.map_offdiag(sns.regplot, scatter=False, ci=None)
    g.map_offdiag(sns.scatterplot)

    # Afficher la légende (catégorie si donnée)
    g.add_legend()

    # Afficher le résultat des corrélations sur les scatter plots
    for i in range(len(variables)):
        for j in range(len(variables)):
            ax = g.axes[i, j]
            corr = round(correlations.loc[variables[i], variables[j]], 2)
            ax.set_title('r²=' + str(corr) + ' (' + force_mesure(corr,
                         'corrélation') + ')', y=0.99, loc='left')


def correlation_matrix(data, corr_seuil=0):
    # Compute the correlation matrix
    cols = colsOfType(data, ['int64', 'int32', 'float64', 'float32'])
    corr = data[cols].corr()**2

    # Filter weak correlations
    corr = corr[corr >= corr_seuil]
    corr_count = corr[corr.notna()].count()
    cols = [col for col in cols if corr_count[col] > 1]
    corr = data[cols].corr()**2

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(max(11, len(cols)/3), max(9, len(cols)/3)))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=0.01, cbar_kws={"shrink": .5})


def eta_squared(x, y):
    """ Calcul état carré entre X et Y """
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x == classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT


def ANOVA(data, X, Ys, sort=True):
    """ Analyse de la variance des variables en paramètres """
    if sort:
        # Ordonner le data set sur la catégorie permettra éventuellement de voir les possibles corrélations sur les graphiques
        data = data.sort_values(X, ascending=False)

    # Préparation de l'affichage des graphiques (boxplots, dispersions) des variables quantitatives Ys par rapport à la variable qualitative X
    lines = len(Ys)
    cols = 2
    index = 0
    fig, axes = plt.subplots(lines, cols, figsize=(
        cols*12, lines*5), sharex=False, sharey=False)
    # ajuster l'espace entre les graphiques.
    fig.subplots_adjust(wspace=0.025, hspace=0.2)
    meanprops = {'marker': 'o', 'markeredgecolor': 'black',
                 'markerfacecolor': 'firebrick'}  # Marquage des moyennes en rouge

    n2s = 0  # Somme des variances (état carré)

    # Affichages des graphiques pour chaque Y
    for Y in Ys:
        # Calcul de la corrélation entre la variable qualitative X et la variable quantitaive Y
        n2 = round(eta_squared(data[X], data[Y]), 2)
        n2s += n2

        index = Ys.index(Y)*cols+1  # Index des sous figures

        # Afficher les dispersions
        ax = plt.subplot(lines, cols, index)
        ax = sns.kdeplot(data=data, x=Y, hue=X)
        # Afficher et ajuster la position du titre des graphiques (valeur et type de la corrélation)
        ax.set_title('n²=' + str(n2) + ' (' + force_mesure(n2,
                     'variance') + ')', x=1.1, loc='right')
        ax.set_xlabel(Y, fontsize=16)

        index += 1
        # Afficher les boxplots
        ax = plt.subplot(lines, cols, index)
        # Afficher les moyennes et Cacher les outliers
        ax = sns.boxplot(data=data, y=X, x=Y, showmeans=True,
                         meanprops=meanprops, showfliers=False)
        ax.set_xlabel(Y, fontsize=16)
        ax.get_yaxis().set_label_position('right')
        ax.get_yaxis().tick_right()

    mu_n2s = round(n2s/lines, 2)  # moyenne des variances (variance moyenne)
    fig.suptitle('Variance par ' + X + ' (n²=' + str(mu_n2s) + ' ' + force_mesure(mu_n2s, 'variance moyenne') + ')',
                 y=0.92, fontsize=24, horizontalalignment='center')  # Titre globale de la figure


def chi2(data, X, Y):
    c = data[[X, Y]].pivot_table(index=X, columns=Y, aggfunc=len)
    cont = c.copy()

    tx = data[X].value_counts()
    ty = data[Y].value_counts()

    n = len(data)
    cont.loc[:, "Total"] = tx
    cont.loc["total", :] = ty
    cont.loc["total", "Total"] = n
    tx = pd.DataFrame(tx)
    ty = pd.DataFrame(ty)
    tx.columns = ["foo"]
    ty.columns = ["foo"]

    indep = tx.dot(ty.T) / n

    c = c.fillna(0)  # on remplace les valeurs nulles par des 0
    mesure = (c-indep)**2/indep
    xi_n = mesure.sum().sum()
    sns.heatmap(mesure/xi_n, annot=c, cmap=sns.cm.rocket_r)
