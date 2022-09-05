from .common_functions import *
from scipy.stats import chi2_contingency, chi2 as xi2, pearsonr


def histo_distribution(data, horizontal=False):
    ''' Affiche la distribution des colonnes numériques '''
    continues = colsOfType(data, ['float'])
    quantitatives = continues + colsOfType(data, ['int'])

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
        width = 3*6
        height = nb_quantitaves*3
        wspace = 0.025
        hspace = 0.3

    # Préparation de l'affichage des graphiques sur deux colonnes : une pour les histogrammes et une pour les boxplots
    fig, axes = plt.subplots(nb_line, nb_col, figsize=(
        width, height), sharex=False, sharey=False)
    # ajuster l'espace entre les graphiques.
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    meanprops = {'marker': 'o', 'markeredgecolor': 'black',
                 'markerfacecolor': 'firebrick'}  # Marquage des moyennes en rouge

    for i in range(0, nb_quantitaves):
        data_quantitative = data[quantitatives[i]]
        index = -1

        # Afficher les histogrammes sur la première colonne
        index += 1
        if nb_quantitaves == 1:
            ax = axes[index]
        else:
            ax = axes[index, i] if horizontal else axes[i, index]
        # Soustitre de colonne
        if i == 0 or horizontal:
            ax.set_title('histogramme', loc='center', y=1, fontsize=18)
        # Ajuster les classes des variables selon leurs types (continues ou discrètes) :
        bins = 'auto' if quantitatives[i] not in continues else 10
        sns.histplot(data_quantitative, ax=ax,
                     color=colors[i % len(colors)], kde=True, bins=bins)
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Afficher les boxplots sans outliers sur la deuxième colonne
        index += 1
        if nb_quantitaves == 1:
            ax = axes[index]
        else:
            ax = axes[index, i] if horizontal else axes[i, index]
        # Soustitre de colonne
        if i == 0 or horizontal:
            ax.set_title('boxplot avec outliers',
                         loc='center', y=1, fontsize=18)
        sns.boxplot(x=data_quantitative, ax=ax, color=colors[i % len(colors)], orient='h', width=.3, showmeans=True,
                    meanprops=meanprops, showfliers=True)  # afficher avec outliers
        if not horizontal:
            ax.set_xlabel(quantitatives[i], fontsize=14, labelpad=1.)
        else:
            ax.set_xlabel('')

        # Afficher les boxplots avec outliers sur la deuxième colonne
        index += 1
        if nb_quantitaves == 1:
            ax = axes[index]
        else:
            ax = axes[index, i] if horizontal else axes[i, index]
        # Soustitre de colonne
        if i == 0 or horizontal:
            ax.set_title('boxplot sans outliers',
                         loc='center', y=1, fontsize=18)
        sns.boxplot(x=data_quantitative, ax=ax, color=colors[i % len(colors)], orient='h', width=.3, showmeans=True,
                    meanprops=meanprops, showfliers=False)  # afficher sans outliers
        if horizontal:
            ax.set_xlabel(quantitatives[i], fontsize=20)
        else:
            ax.set_xlabel('')


def force_mesure(mesure, type=None):
    ''' Renvoie un texte avec la force de la mesure selon les seuils '''
    force_text = ''

    # Définition des seuils/intervalles de mesure
    seuils = {
        0: 'aucune', .2: 'faible', .4: 'moyenne', .6: 'forte', .8: 'très forte'}

    for seuil, force in seuils.items():
        if np.abs(mesure) >= seuil:
            force_text = force

    return force_text + (' ' + type if type != None else '')


def afficher_correlations(data, categorie=None):
    ''' Calcul et affichage des corrélations linéaires entre les 'variables' '''
    # Garder uniquement les variables numériques
    variables = colsOfType(data)

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
    g.map_diag(sns.kdeplot, fill=True, warn_singular=False)

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
            ax.set_title('r²=' + str(corr) + ' p-value=' + str(round(pearsonr(data[variables[i]], data[variables[j]])[1], 3)) +
                         ' (' + force_mesure(corr) + ')', y=0.99, loc='left')


def correlation_matrix(data, corr_seuil=0, squared=True, triangle=True, sort=False, p_value=False):
    ''' Calcul et affiche la matrice heatmap dezs corrélations de Pearson entre les colonnes '''
    # Compute the correlation matrix
    cols = colsOfType(data)
    corr = data[cols].corr()
    if squared:
        corr = corr**2

    # Filter weak correlations
    corr = corr[abs(corr) >= corr_seuil]
    corr_count = corr[corr.notna()].count()
    cols = [col for col in cols if corr_count[col] > 1]
    corr = data[cols].corr()

    if squared:
        corr = corr**2
    corr = corr[abs(corr) >= corr_seuil]

    if len(corr) > 0:
        if sort:
            corr = corr.sort_index(axis=0).sort_index(axis=1)

        pvalues = None
        if p_value:
            # Calculer les p-values pour la pertinence des corrélations
            pvalues = []
            for col1 in corr.columns:
                p = []
                for col2 in corr.columns:
                    pvalue = round(pearsonr(data[col1], data[col2])[1], 3)
                    p.append(pvalue)
                pvalues.append(p)

        mask = None
        if triangle:
            # Generate a mask for the upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(
            figsize=(max(11, len(cols)/3), max(9, len(cols)/3)))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, annot=pvalues,
                    square=True, linewidths=0.01, cbar_kws={'shrink': .5, 'label': 'Pearson Correlation (r'+('²' if squared else '')+')'})
        plt.title('Correlations', fontsize=24)
        plt.show()
    return corr


def removeCorrelations(corrs):
    cols_drop = set()
    cols_corr = {}
    cols_corr_bis = {}

    sorted_corrs = pd.DataFrame(corrs.count(
        axis=1).sort_values(ascending=False), columns=['Count'])
    cols = sorted_corrs.index.tolist()
    for col in cols:
        cols_corr[col] = set()
        cols_corr_bis[col] = set()
        for col2 in cols:
            if col != col2 and corrs.loc[col, col2] > 0:
                cols_corr[col].add(col2)
                cols_corr_bis[col].add(col2)

    # cols_corr_bis = cols_corr.copy()
    for col in cols:
        for col2 in cols_corr[col]:
            if col2 in cols_corr and col2 not in cols_drop:
                cols_corr_bis[col2].remove(col)
        cols_drop.add(col)

    cols_drop = set()
    for col in cols:
        if len(cols_corr_bis[col]) > 0:
            cols_drop.add(col)

    return list(cols_drop)


def eta_squared(x, y):
    ''' Calcul état carré entre X et Y '''
    moyenne_y = y.mean()
    classes = []
    uniques = x.unique().tolist()
    if np.nan in uniques:
        uniques.remove(np.nan)
    for classe in uniques:
        yi_classe = y[x == classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT


def ANOVA(data, X, Ys=None, sort=True, display_kde=True, yloc=0.9, hspace=0.3, width=8, heigth=0.26):
    ''' Analyse de la variance des variables en paramètres '''
    if sort:
        # Ordonner le data set sur la catégorie permettra éventuellement de voir les possibles corrélations sur les graphiques
        data = data.sort_values(X, ascending=False)
    # Récupérer ou garder uniquement les colonnes numériques
    if Ys == None:
        Ys = colsOfType(data)
    else:
        Ys = colsOfType(data[Ys])

    # Préparation de l'affichage des graphiques (boxplots, dispersions) des variables quantitatives Ys par rapport à la variable qualitative X
    lines = len(Ys)
    cols = 2 if display_kde else 1
    index = 0
    fig, axes = plt.subplots(lines, cols, figsize=(
        cols*width, int(lines*len(data[X].unique())*heigth)), sharex=False, sharey=False)
    # ajuster l'espace entre les graphiques.
    fig.subplots_adjust(wspace=0.025, hspace=hspace)
    meanprops = {'marker': 'o', 'markeredgecolor': 'black',
                 'markerfacecolor': 'firebrick'}  # Marquage des moyennes en rouge

    n2s = 0  # Somme des variances (état carré)

    # Affichages des graphiques pour chaque Y
    for Y in Ys:
        means = {}
        for v in data[X].unique().tolist():
            means[v] = data[data[X] == v][Y].mean()
        data['mean_'+Y] = data[X].apply(lambda x: means[x])
        data = data.sort_values(
            by='mean_'+Y, ascending=False).drop(columns='mean_'+Y)

        index = Ys.index(Y)*cols+1  # Index des sous figures

        if display_kde:
            # Afficher les dispersions
            ax = plt.subplot(lines, cols, index)
            ax = sns.kdeplot(data=data, x=Y, hue=X,
                             warn_singular=False, legend=None)
            ax.set_xlabel(Y, fontsize=16)
            index += 1

        # Afficher les boxplots
        ax = plt.subplot(lines, cols, index)
        # Afficher les moyennes et Cacher les outliers
        ax = sns.boxplot(data=data, y=X, x=Y, orient='h', showmeans=True,
                         meanprops=meanprops, showfliers=False)
        ax.set_xlabel(Y, fontsize=16)
        ax.get_yaxis().set_label_position('right')
        ax.get_yaxis().tick_right()

        # Calcul de la corrélation entre la variable qualitative X et la variable quantitaive Y
        n2 = eta_squared(data[X], data[Y])
        n2s += n2
        # Afficher et ajuster la position du titre des graphiques (valeur et type de la corrélation)
        n2 = round(n2, 3)
        if len(Ys) > 1:
            ax.set_title('n²=' + str(n2) + ' (' + force_mesure(n2,
                                                               'variance') + ')', x=-.1 if display_kde else .4, y=0.985 if display_kde else 1, loc='left')

    mu_n2s = round(n2s/lines, 3)  # moyenne des variances (variance moyenne)
    fig.suptitle('Variance par ' + X + ' (n²=' + str(mu_n2s) + ' ' + force_mesure(mu_n2s, 'variance'+(' moyenne' if len(Ys) > 1 else '')) + ')',
                 x=0.5 if display_kde else 0.8, y=yloc, fontsize=24, horizontalalignment='center')  # Titre globale de la figure


def chi2(data, X, Y, heigth=None, width=None):
    c = pd.crosstab(data[X], data[Y])
    cont = c.copy()

    tx = data[X].value_counts()
    ty = data[Y].value_counts()

    n = len(data)
    cont.loc[:, 'Total'] = tx
    cont.loc['total', :] = ty
    cont.loc['total', 'Total'] = n
    tx = pd.DataFrame(tx)
    ty = pd.DataFrame(ty)
    tx.columns = ['foo']
    ty.columns = ['foo']

    indep = tx.dot(ty.T) / n

    c = c.fillna(0)  # on remplace les valeurs nulles par des 0
    mesure = (c-indep)**2/indep
    xi_n = mesure.sum().sum()

    # Afficher les résultat du test chi²
    chi, pval, dof, exp = chi2_contingency(c)
    print('p-value is: ', pval)
    significance = 0.05
    p = 1 - significance
    critical_value = xi2.ppf(p, dof)
    print('chi=%.6f, critical value=%.6f' % (chi, critical_value))
    if chi > critical_value:
        print("""At %.2f level of significance, we reject the null hypotheses and accept H1. 
    They are not independent.""" % (significance))
    else:
        print("""At %.2f level of significance, we accept the null hypotheses. 
    They are independent.""" % (significance))

    # Afficher la heatmap de contingence
    if width == None:
        width = len(data[X].unique()) * 0.6
    if heigth == None:
        heigth = len(data[Y].unique()) * 0.6
    fig, ax = plt.subplots(figsize=(width, heigth))
    plt.title('CHi2 Contingency', fontsize=24)
    sns.heatmap(data=mesure/xi_n, annot=c*100,
                cmap=sns.cm.rocket_r, fmt='g', cbar_kws={'label': 'CHi2 (%)'}, ax=ax)
    ax.set_xlabel(Y)
    ax.set_ylabel(X)
