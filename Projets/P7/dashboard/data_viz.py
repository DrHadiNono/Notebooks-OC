import matplotlib.pyplot as plt
import re
import seaborn as sns
sns.set()  # Définir le style par défaut pour les graphiques
sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})


def colsOfType(data, types=None):
    ''' Retourne la liste de colonnes dont le type est dans "TYPES" '''
    cols = []
    if types == None:
        types = ['int', 'float']
    dtypes = data.dtypes.to_dict()
    for col_name, type in dtypes.items():
        if (re.split('\d+', str(type))[0] in types):
            cols.append(col_name)
    return cols


def dispersion(data, x, y, z=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(data=data, x=x, y=y,
                    hue=data['Clients'], size=data['Clients'], size_order=['Sélectionné', 'Autres'], style=data['Clients'], sizes=(20, 300), legend='brief', ax=ax)


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
        nb_line = 2
        nb_col = nb_quantitaves
        width = nb_quantitaves*9
        height = 3*5
        wspace = 0.1
    else:
        nb_line = nb_quantitaves
        nb_col = 2
        width = 3*6
        height = nb_quantitaves*3
        wspace = 0.025
        hspace = 0.4

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

        # Afficher les boxplots avec outliers sur la deuxième colonne
        index += 1
        if nb_quantitaves == 1:
            ax = axes[index]
        else:
            ax = axes[index, i] if horizontal else axes[i, index]
        # Soustitre de colonne
        if i == 0 or horizontal:
            ax.set_title('boxplot',
                         loc='center', y=1, fontsize=18)
        sns.boxplot(x=data_quantitative, ax=ax, color=colors[i % len(colors)], orient='h', width=.3, showmeans=True,
                    meanprops=meanprops, showfliers=True)  # afficher avec outliers
        if not horizontal:
            ax.set_xlabel(quantitatives[i], fontsize=14,
                          labelpad=10., x=-.01)
        else:
            ax.set_xlabel('')
