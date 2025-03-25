# Cartographie des Donneurs de Sang

## Description
Cette application Streamlit permet d'analyser et de visualiser les données relatives aux donneurs de sang, en mettant l'accent sur la cartographie, l'analyse des profils et la prédiction de l'éligibilité des donneurs.

## Fonctionnalités
- **Accueil** : Présentation du tableau de bord interactif.
- **Impact des Conditions de Santé** : Analyse des facteurs influençant l'éligibilité des donneurs.
- **Profilage des Donneurs Idéaux** : Regroupement des donneurs en fonction de divers critères.
- **Analyse de l’Efficacité des Campagnes** : Étude des tendances de dons.
- **Fidélisation des Donneurs** : Analyse des profils les plus récurrents.
- **Cartographie des Donneurs** : Visualisation géographique des donneurs.
- **Prédiction** : Modèle de machine learning pour prédire l'éligibilité des donneurs.

## Installation
### Prérequis
- Python 3.8+
- pip

### Étapes
1. Clonez le dépôt GitHub :
   ```bash
   git clone [https://github.com/votre-utilisateur/votre-repo.git](https://github.com/fabrice002/CBDH-2025-Project/)
   cd votre-repo
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Lancez l'application Streamlit :
   ```bash
   streamlit run app.py
   ```

## Configuration
- Assurez-vous d'avoir un fichier Excel contenant les données des donneurs.
- Ajoutez un fichier `cleaning.py` définissant la fonction `nettoyer_donnees()`.
- Placez un logo `logo.png` dans le dossier racine.
- Un modèle de machine learning `modele_prediction.pkl` doit être disponible pour la prédiction.

## Technologies utilisées
- **Streamlit** : Interface utilisateur interactive.
- **Pandas** : Manipulation et nettoyage des données.
- **Plotly** : Visualisation interactive.
- **Folium** : Cartographie des donneurs.
- **scikit-learn** : Modèle de prédiction de l'éligibilité.

