import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier




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
# print(df.info())
'''
# Régression logistique

le = LabelEncoder()
df = df.apply(le.fit_transform)
df_data = df.drop('recidive', axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_data, df.recidive, random_state=11, test_size=.3)

logreg = LogisticRegression(max_iter=1000)

logreg.fit(X_train, y_train)

print(accuracy_score(y_test, logreg.predict(X_test)))
'''

# Réduction de dimensionnalité
le = LabelEncoder()
df = df.apply(le.fit_transform)
df_data = df.drop('recidive', axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_data, df.recidive, random_state=11, test_size=.3)

classifier_model = ExtraTreesClassifier(n_estimators=50)

classifier_model = classifier_model.fit(X_train, y_train)

importance_scores = classifier_model.feature_importances_
print(importance_scores)
data = {'Class': ['center', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'guardian', 'proxDT', 'education', 'internet', 'romantic', 'famrel', 'freetime',
                     'goout', 'Ddalc', 'Walc', 'health', 'absences'], 'importance scores': importance_scores}
df_importance_scores = pd.DataFrame(data)
df_importance_scores = df_importance_scores.set_index(['Class'])
df_importance_scores = df_importance_scores.nlargest(5, 'importance scores')

df_importance_scores.plot(kind ='bar')
plt.tight_layout()
plt.show()

print(df_importance_scores)

score = accuracy_score(y_test, classifier_model.predict(X_test))
print(score)
