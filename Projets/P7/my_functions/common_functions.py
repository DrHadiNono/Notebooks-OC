# Librairies pour le traitement des données
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import re

pd.set_option('display.max_columns', None)
# Librairies pour la visualisation de graphiques
sns.set()  # Définir le style par défaut pour les graphiques
sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})


def verifier_taille(data, title=None):
    ''' Fonction de vérification de la taille d'un data set '''
    lignes = data.shape[0]
    colonnes = data.shape[1]
    print('Le data set contient :')
    print('\t-', lignes, 'lignes et', colonnes, 'colonnes.')

    nb_null = data.isnull().sum().sum()
    taille = lignes*colonnes
    taille_null = 100*nb_null/taille
    taille_non_null = 100*(taille-nb_null)/taille
    print('\t-', nb_null, 'valeurs manquantes, ce qui représente',
          round(taille_null, 2), '% du data set.')

    # Afficher la répartition du taux de valeurs manquantes
    nan_data = pd.DataFrame({'valeurs': [''], 'non-null('+str(round(taille_non_null, 2))+'%)': [
                            taille_non_null], 'null('+str(round(taille_null, 2))+'%)': [taille_null]})
    nan_data.set_index('valeurs').plot(kind='barh', stacked=True, color=[
        'green', 'orange'], figsize=(8, 2), fontsize=12, title=title)
    plt.xlabel('%')
    plt.show()

    if taille_null > 0:
        print('Voici les colonnes avec NaNs:')
        display(nan_cols(data, -1).T)


def afficher_echantillon(data, n=2, frac=0.002):
    ''' Afficher un sous-échantillon aléatoire '''
    print('Voici un sous-échantillon aléatoire :')
    sample = data.sample(frac=frac, random_state=np.random.seed()).head(n)
    display(sample)


def valeurs_manquantes(data):
    ''' Retourner les valeurs manquantes d'un data frame/set '''
    return data[data.isnull().any(axis=1)]


def doublons(data):
    ''' Recherche les doublons dans le data set '''
    return data[data.duplicated()]


def nan_cols(data, nan_seuil=0):
    ''' Calcul et retourne les taux de remplissage de chaque colonne '''
    n = len(data)
    dict_info = {'Column': [], '#NaN': [], '%NaN': [],
                 '#Not_NaN': [], '%Not_NaN': [], }

    for col in data.columns:
        nan_count = data[col].isnull().sum()
        nan_percent = 100*nan_count/n
        if (nan_seuil > -1 and nan_percent >= nan_seuil) or (nan_seuil == -1 and nan_count > 0):
            dict_info['Column'].append(col)
            dict_info['#NaN'].append(nan_count)
            dict_info['%NaN'].append(round(nan_percent, 2))
            dict_info['#Not_NaN'].append(n-nan_count)
            dict_info['%Not_NaN'].append(round(100*(n-nan_count)/n, 2))

    return pd.DataFrame(dict_info).sort_values(by='#NaN', ascending=False).reset_index(level=0, drop=True)


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


def renameCol(data, old_col, new_col):
    return data.rename(columns={old_col: new_col})


# ============================ Scalers ======================================
# Fonctions de normalisations

def scaled(data, scaler, frame=False, return_scaler=False):
    if frame:
        data = data[colsOfType(data)]
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


def Robust_Scaled(data, frame=False, return_scaler=False):
    return scaled(data, RobustScaler(), frame, return_scaler)


def PowerTransformer_Scaled(data, frame=False, return_scaler=False):
    return scaled(data, PowerTransformer(), frame, return_scaler)

# ============================================================================


# ============================ Plots ======================================
def pivotCount(data, cols, mask=None):
    if len(cols) == 1:
        count = pd.DataFrame(data[cols[0]].value_counts())
    else:
        count = data.pivot_table(
            index=cols[0], columns=cols[1], values=cols[2], aggfunc='count')

    for col in count.columns:
        count[col] *= 100/count[col].sum()
    count = count.replace(np.NaN, 0)

    if mask != None:
        count = count.loc[[mask]]
    count = count.T
    return count.sort_values(ascending=True, by=count.columns[0])


def plotBalance(data, cols, mask=None, stack=True, title=None, width=10, heigth=5, rotation=None, orient='v'):
    cnt = pivotCount(data, cols, mask)
    if mask != None:
        barplot(x=cnt.index, y=cnt.iloc[:, 0].values, xlabel=cols[1], ylabel='Distribution (%)', title=title if title != None else (
            cols[1]+' Distribution'+(' when '+cols[0]+'='+str(mask) if len(cols) > 1 else '')), width=width, heigth=heigth, percent=False, rotation=rotation, orient=orient)
    else:
        cnt.plot(kind='barh', stacked=stack, figsize=(
            8, (1.5+np.log(len(cols))*2) if stack else 1+np.log(len(cnt.T))), fontsize=12)
        plt.title(label=title if title != None else (
            cols[0]+' Distribution'+(' by '+cols[1] if len(cols) > 1 else '')), fontsize=20)
        if len(cols) == 1:
            plt.legend(labels=[str(
                idx)+' ('+str(round(cnt.T.loc[idx, cols[0]], 2))+'%)' for idx in cnt.T.index])
        plt.ylabel('')
        plt.xlabel('Distribution (%)')
        plt.show()


def barplot(x, y, xlabel='', ylabel='#', title=None, width=10, heigth=5, percent=True, rotation=None, orient='v'):
    fig, ax = plt.subplots(figsize=(width, heigth))
    sns.barplot(x=x if orient == 'v' else y,
                y=y if orient == 'v' else x, orient=orient)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(label=title, fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

    for i in ax.containers:
        ax.bar_label(i, labels=[str(round(i.datavalues[j], 2) if not percent else int(i.datavalues[j]))+(('\n' if orient == 'v' else ' ')+'('+str(
            round(i.datavalues[j]/(sum(i.datavalues))*100, 2))+'%)' if percent else '') for j in range(len(i))])
    plt.show()


def barplotDistribution(serie, title=None, width=10, heigth=5, percent=True, rotation=None, orient='v'):
    count = pd.DataFrame(serie.value_counts())
    count = count.sort_values(ascending=True, by=count.columns[0])
    barplot(x=count.index, y=count.iloc[:, 0], xlabel=count.columns[0], ylabel='#', title=title if title !=
            None else count.columns[0]+' Distribution', width=width, heigth=heigth, percent=percent, rotation=rotation, orient=orient)


def lineplot(x, y, xlabel, ylabel, width=10, heigth=5):
    fig, ax = plt.subplots(figsize=(width, heigth))
    sns.lineplot(y=y, x=x)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.xticks(x)
    plt.show()
# ============================================================================

# ============================ Preprocessing ======================================


def SimpleImputation(df, startegy='mean'):
    # Garder les colonnes numériques uniquement
    dfn = df[colsOfType(df)]
    data = dfn.values

    # Imputer les valeurs manquantes par la moyenne
    imputer = SimpleImputer(strategy=startegy)
    imputer_transfom = imputer.fit_transform(data)
    imputed_data = pd.DataFrame(
        imputer_transfom, columns=dfn.columns, index=dfn.index)

    # Remplir le data set original
    dfc = df.copy()
    dfc[colsOfType(df)] = imputed_data[colsOfType(df)]
    return dfc
