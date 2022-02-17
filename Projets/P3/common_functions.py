# Librairies pour le traitement des données
import pandas as pd
import numpy as np

# Librairies pour la visualisation de graphiques
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()  # Définir le style par défaut pour les graphiques


def verifier_taille(data, dask=True):
    """ Fonction de vérification de la taille d'un data set """
    lignes = data.shape[0]
    if dask:
        lignes = lignes.compute()
    colonnes = data.shape[1]
    print('Le data set contient :')
    print('\t-', lignes, 'lignes et', colonnes, 'colonnes.')

    nb_null = data.isnull().sum().sum()
    if dask:
        nb_null = nb_null.compute()
    taille = lignes*colonnes
    taille_null = 100*nb_null/taille
    taille_non_null = 100*(taille-nb_null)/taille
    print('\t-', nb_null, 'valeurs manquantes, ce qui représente',
          round(taille_null, 2), '% du data set.')

    # Afficher la répartition du taux de valeurs manquantes
    nan_data = pd.DataFrame({'valeurs (%)': [''], 'non-null('+str(round(taille_non_null, 2))+'%)': [
                            taille_non_null], 'null('+str(round(taille_null, 2))+'%)': [taille_null]})
    nan_data.set_index('valeurs (%)').plot(kind='barh', stacked=True, color=[
        'green', 'orange'], figsize=(8, 2), fontsize=12)
    plt.xlabel('%')


def afficher_echantillon(data, n=2):
    """ Afficher un sous-échantillon aléatoire """
    print('Voici un sous-échantillon aléatoire :')
    return data.sample(frac=0.002, random_state=np.random.seed()).head(n)


def valeurs_manquantes(data):
    """ Retourner les valeurs manquantes d'un data frame/set """
    return data[data.isnull().any(axis=1)]


def nan_cols(data, nan_seuil=0):
    n = len(data)
    dict_info = {'Column': [], 'NaN_Count': [], 'NaN_Percent': [],
                 'Not_NaN_Count': [], 'Not_NaN_Percent': [], }

    for col in data.columns:
        nan_count = data[col].isnull().sum()
        nan_percent = 100*nan_count/n
        if nan_percent >= nan_seuil:
            dict_info['Column'].append(col)
            dict_info['NaN_Count'].append(nan_count)
            dict_info['NaN_Percent'].append(nan_percent)
            dict_info['Not_NaN_Count'].append(n-nan_count)
            dict_info['Not_NaN_Percent'].append(100*(n-nan_count)/n)

    return pd.DataFrame(dict_info)


def colsOfType(data, types):
    cols = []
    dtypes = data.dtypes.to_dict()
    for col_name, type in dtypes.items():
        if (type in types):
            cols.append(col_name)
    return cols
