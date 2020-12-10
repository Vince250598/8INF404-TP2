import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier

# valeur pour le générateur de nombre aléatoire
random_state = 10

# On ajuste les options pour afficher plus de colonnes dans la console
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_excel("Données centre jeunesse.xlsx")

# Exploration des données
print("--------------EXPLORATION DES DONNÉES--------------")

print("cinqs premières lignes de l'ensemble de données:\n" + str(df.head()))
print("Informations générals sur l'ensemble de données:\n" + str(df.info()))
print("Statistiques sur l'ensemble de données:\n" + str(df.describe()))

sns.pairplot(data=df) # Prend une quinzaine de secondes à rouler, à commenter au besoin
plt.subplots(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True)
plt.show()

# Nettoyage des données
print("--------------NETTOYAGE DES DONNÉES--------------")

# Vérification valeurs extrêmes pour le nombre d'absences
plt.subplots(figsize=(14, 10))
sns.boxplot(x='absences', data=df)
plt.title("Dispersion du nombre d'absences")
plt.tight_layout()
plt.show()

# Vérification doublons, aucun doublons
print("Nombres de doublons dans l'ensemble de données: " + str(len(df[df.duplicated()])))

# Vérification valeurs manquantes (Aucune)
print("Nombre de valeurs manquantes pour chacun des attributs:\n" + str(df.isnull().sum()))

# Ingénierie des caractéristiques
'''Nous avons envisagés redéfinir des attributs afin d'obtenir des caractéristiques mieux adapté à la conception de notre modèle
mais, les attributs de l'ensemble de données sont déjà sous un bon format.'''

# Régression logistique
print("--------------RÉGRESSION LOGISTIQUE--------------")

label_encoder = LabelEncoder()
df_logistic_regression = df.apply(label_encoder.fit_transform)
df_logistic_regression = df_logistic_regression.drop('recidive', axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_logistic_regression, df.recidive, random_state=random_state, test_size=.3)

logistic_regression = LogisticRegression(max_iter=1000)

logistic_regression.fit(X_train, y_train)

logistic_regression_score = accuracy_score(y_test, logistic_regression.predict(X_test))
print("Score de performance du modèle de régression logistique: " + str(logistic_regression_score))

predictions = logistic_regression.predict(X_test)
logistic_regression_confusion_matrix = confusion_matrix(y_test, predictions)
print("Matrice de confusion pour le modèle de régression logistique: ")
print(logistic_regression_confusion_matrix)

sensibility = logistic_regression_confusion_matrix[0][0]/(logistic_regression_confusion_matrix[0][0] + logistic_regression_confusion_matrix[1][0])
print("Sensibilité: " + str(sensibility))
specificity = logistic_regression_confusion_matrix[1][1]/(logistic_regression_confusion_matrix[1][1] + logistic_regression_confusion_matrix[0][1])
print("Spécificité: " + str(specificity))

# Réduction de dimensionnalité
print("--------------RÉDUCTION DE DIMENSIONNALITÉ--------------")

label_encoder = LabelEncoder()
df_extremely_randomized_trees = df.apply(label_encoder.fit_transform)
df_extremely_randomized_trees = df_extremely_randomized_trees.drop('recidive', axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_extremely_randomized_trees, df.recidive, random_state=random_state, test_size=.3)

# Arbre de classification Extremely Randomized Trees
extremely_randomized_trees_model = ExtraTreesClassifier(n_estimators=50)

extremely_randomized_trees_model = extremely_randomized_trees_model.fit(X_train, y_train)

importance_scores = extremely_randomized_trees_model.feature_importances_
data = {'Classe': ['center', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'guardian',
                   'proxDT', 'education', 'internet', 'romantic', 'famrel', 'freetime',
                   'goout', 'Ddalc', 'Walc', 'health', 'absences'], "Scores d'importance": importance_scores}
df_importance_scores = pd.DataFrame(data)
df_importance_scores = df_importance_scores.set_index(['Classe'])
df_importance_scores = df_importance_scores.nlargest(5, "Scores d'importance")

df_importance_scores.plot(kind='bar')
plt.title("Importances des attributs pour prédire la récidive")
plt.tight_layout()
plt.show()

print(df_importance_scores)

print("--------------EXTREMELY RANDOMIZED TREES--------------")

extremely_randomized_trees_score = accuracy_score(y_test, extremely_randomized_trees_model.predict(X_test))
print("Score de performance du modèle Extremely Randomized Trees: " + str(extremely_randomized_trees_score))

predictions = extremely_randomized_trees_model.predict(X_test)
extremely_randomized_trees_confusion_matrix = confusion_matrix(y_test, predictions)
print("Matrice de confusion pour le modèle d'arbre de classification Extremely Randomized Trees: ")
print(extremely_randomized_trees_confusion_matrix)

sensibility = extremely_randomized_trees_confusion_matrix[0][0]/(extremely_randomized_trees_confusion_matrix[0][0] + extremely_randomized_trees_confusion_matrix[1][0])
print("Sensibilité: " + str(sensibility))
specificity = extremely_randomized_trees_confusion_matrix[1][1]/(extremely_randomized_trees_confusion_matrix[1][1] + extremely_randomized_trees_confusion_matrix[0][1])
print("Spécificité: " + str(specificity))
