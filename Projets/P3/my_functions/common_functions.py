# Librairies pour le traitement des données
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
# Librairies pour la visualisation de graphiques
sns.set()  # Définir le style par défaut pour les graphiques


def verifier_taille(data, dask=False):
    ''' Fonction de vérification de la taille d'un data set '''
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
    nan_data = pd.DataFrame({'valeurs': [''], 'non-null('+str(round(taille_non_null, 2))+'%)': [
                            taille_non_null], 'null('+str(round(taille_null, 2))+'%)': [taille_null]})
    nan_data.set_index('valeurs').plot(kind='barh', stacked=True, color=[
        'green', 'orange'], figsize=(8, 2), fontsize=12)
    plt.xlabel('%')


def afficher_echantillon(data, n=2):
    ''' Afficher un sous-échantillon aléatoire '''
    print('Voici un sous-échantillon aléatoire :')
    return data.sample(frac=0.002, random_state=np.random.seed()).head(n)


def valeurs_manquantes(data):
    ''' Retourner les valeurs manquantes d'un data frame/set '''
    return data[data.isnull().any(axis=1)]


def nan_cols(data, nan_seuil=0):
    n = len(data)
    dict_info = {'Column': [], '#NaN': [], '%NaN': [],
                 '#Not_NaN': [], '%Not_NaN': [], }

    for col in data.columns:
        nan_count = data[col].isnull().sum()
        nan_percent = 100*nan_count/n
        if nan_percent >= nan_seuil:
            dict_info['Column'].append(col)
            dict_info['#NaN'].append(nan_count)
            dict_info['%NaN'].append(round(nan_percent, 2))
            dict_info['#Not_NaN'].append(n-nan_count)
            dict_info['%Not_NaN'].append(round(100*(n-nan_count)/n, 2))

    return pd.DataFrame(dict_info)


def colsOfType(data, types):
    cols = []
    dtypes = data.dtypes.to_dict()
    for col_name, type in dtypes.items():
        if (type in types):
            cols.append(col_name)
    return cols


# ============================ Scalers ======================================

def scaled(data, scaler, frame=False, return_scaler=False):
    scaler = scaler
    scaled_data = scaler.fit_transform(data)

    res = scaled_data
    if frame:
        res = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

    if return_scaler:
        return res, scaler
    else:
        return res


def reverse_scaled_frame(scaled_data, scaler):
    return pd.DataFrame(scaler.inverse_transform(scaled_data.values), columns=scaled_data.columns, index=scaled_data.index)


def MinMax_Scaled(data, frame=False, return_scaler=False):
    return scaled(data, MinMaxScaler(), frame, return_scaler)


def Std_Scaled(data, frame=False, return_scaler=False):
    return scaled(data, StandardScaler(), frame, return_scaler)

# ============================================================================
