# Cartographie des Donneurs de Sang

## Description
Ce projet est un tableau de bord interactif développé avec Streamlit pour analyser et visualiser les données liées aux donneurs de sang. Il offre plusieurs fonctionnalités pour mieux comprendre la répartition géographique des donneurs, l'impact des conditions de santé sur l'éligibilité au don, ainsi que des prédictions basées sur un modèle de Machine Learning.

## Fonctionnalités
### 1. **Accueil**
- Affichage des statistiques clés (nombre total de donneurs, âge moyen, taux d'éligibilité, etc.).
- Présentation des 10 premières lignes du jeu de données.

### 2. **Impact des Conditions de Santé**
- Analyse des raisons d'inéligibilité au don de sang.
- Graphiques interactifs sur les principales conditions médicales affectant les donneurs.

### 3. **Profilage des Donneurs Idéaux**
- Clustering des donneurs en fonction de différentes caractéristiques (âge, genre, profession, taille, poids).
- Visualisation des groupes similaires via un algorithme K-Means.

### 4. **Analyse de l’Efficacité des Campagnes**
- Étude des tendances des dons sur une période donnée.
- Analyse démographique des donneurs (répartition par genre, profession, etc.).

### 5. **Fidélisation des Donneurs**
- Identification des profils de donneurs réguliers.
- Exploration des critères influençant la récurrence des dons.
- Cartographie des donneurs fidèles par arrondissement.

### 6. **Cartographie des Donneurs**
- Visualisation géographique des donneurs sur une carte interactive avec Folium.
- Heatmap pour analyser la densité des donneurs par quartier.
- Répartition des donneurs par arrondissement.

### 7. **Prédiction de l'Éligibilité au Don**
- Utilisation d’un modèle de Machine Learning (K-Means et prédiction avec Scikit-learn).
- Interface permettant aux utilisateurs d’entrer des caractéristiques et de recevoir une prédiction sur leur éligibilité au don.

## Installation
### Prérequis
- Python 3.8+
- pip

### Étapes
1. Clonez le dépôt GitHub :
   ```bash
   git clone https://github.com/fabrice002/CBDH-2025-Project.git
   cd CBDH-2025-Project
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Lancez l'application Streamlit :
   ```bash
   streamlit run app.py
   ```
4. Une fois l'application ouverte, sélectionnez `Updated Challenge dataset.xlsx`

## Hypothèses faites lors du développement
- Les quartiers de résidence sont géocodés avec l’API Geopy, mais certaines localisations peuvent ne pas être précises.
- Le clustering K-Means suppose que les donneurs peuvent être regroupés en profils homogènes en fonction des critères choisis.
- Les valeurs aberrantes pour l’âge et le poids sont corrigées en utilisant des médianes ou des moyennes.
- Les données manquantes dans certaines colonnes ont été imputées en utilisant des méthodes statistiques adaptées.


## Configuration
- Le fichier `Excel Updated Challenge dataset.xlsx` et le modèle de machine learning `modele_prediction.pkl` sont présents dans le répertoire et doivent être assurés d'être disponibles.

## Technologies utilisées
- **Streamlit** : Interface utilisateur interactive.
- **Pandas** : Manipulation et nettoyage des données.
- **Plotly** : Visualisation interactive.
- **Folium** : Cartographie des donneurs.
- **scikit-learn** : Modèle de prédiction de l'éligibilité.

## Utilisation de l’API du Modèle de Prédiction
Le modèle de prédiction utilisé permet de déterminer si un donneur est éligible ou non en fonction de ses caractéristiques. Voici les étapes pour l'utiliser :
1. **Sélection des paramètres** : L’utilisateur renseigne ses informations personnelles dans l’interface Streamlit.
2. **Prétraitement des données** : Les valeurs sont normalisées et transformées pour correspondre au format d’entraînement du modèle.
3. **Exécution du modèle** : Le modèle prédit l’éligibilité du donneur sur la base des données fournies.
4. **Affichage des résultats** : La réponse du modèle est affichée à l’utilisateur.


## Auteurs
- Ouandji Hervé Fabrice
- Teng Kana Arielle
- Moussa Adou
- Dongmo Aziz C.

