import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_excel("Données centre jeunesse.xlsx")
# Exploration des données
'''
print(df.head())
print(df.info())
print(df.describe())
sns.pairplot(data = df)

plt.subplots(figsize = (14,10))
sns.heatmap(df.corr(), annot=True)

plt.show()
'''

# Nettoyage des données
'''
# Vérification valeurs extrêmes pour le nombre d'absences
plt.subplots(figsize = (14,10))
sns.boxplot(x = 'absences', data=df)
plt.title("Dispersion du nombre d'absences")
plt.tight_layout()
plt.show()

# Vérification doublons, aucun doublons
print(df[df.duplicated()].head()) 

# Vérification valeurs manquantes (Aucune)
print(df.isnull().sum())
'''

# Ingénierie des caractéristiques
'''Nous avons envisagés redéfinir des attributs afin d'obtenir des caractéristiques mieux adapté à la conception de notre modèle
mais, les attributs de l'ensemble de données sont déjà sous un bon format.'''
print(df.info())
# Régression logistique
rng = np.random.RandomState(10)
noise = rng.normal(size=(len(df)))
X_w_noise = df
X_w_noise = X_w_noise.drop(['recidive'], axis=1)
X_w_noise['noise'] = noise
print(X_w_noise.head())

X_train, X_test, y_train, y_test = train_test_split(X_w_noise, df.recidive, random_state=0, test_size=.7)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))
