import pandas as pd
import re
import numpy as np
from unidecode import unidecode
from thefuzz import fuzz, process
import numpy as np



def preprocess_eligibility_data(df):
    """
    Traite les données d'éligibilité au don en filtrant les non-éligibles,
    nettoyant les données et regroupant les raisons d'indisponibilité.
    """


    # 1️⃣ Suppression des colonnes inutiles
    cols_to_drop = ['Date de dernières règles (DDR) ', 'Sélectionner "ok" pour envoyer']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 3️⃣ Remplacement des valeurs manquantes par 'Non'
    df= df.fillna('Non')

    # 4️⃣ Regroupement des raisons d'indisponibilité
    columns_mapping = {
            "Raison indisponibilité  [Taux d’hémoglobine bas ]": "Taux d’hémoglobine bas",
            "Raison indisponibilité  [date de dernier Don < 3 mois ]": "Date de dernier Don < 3 mois",
            "Raison indisponibilité  [Est sous anti-biothérapie  ]": "Est sous anti-biothérapie",
            "Raison indisponibilité  [IST récente (Exclu VIH, Hbs, Hcv)]": "IST récente (Exclu VIH, Hbs, Hcv)",
            "Raison de l’indisponibilité de la femme [La DDR est mauvais si <14 jour avant le don]": "La DDR < 14 jours",
            "Raison de l’indisponibilité de la femme [A accoucher ces 6 derniers mois  ]": "A accoucher ces 6 derniers mois",
            "Raison de l’indisponibilité de la femme [Allaitement ]": "Allaitement",
            "Raison de l’indisponibilité de la femme [Interruption de grossesse  ces 06 derniers mois]": "Interruption de grossesse ces 06 derniers mois",
            "Raison de l’indisponibilité de la femme [est enceinte ]": "Est enceinte",
            "Raison de non-eligibilité totale  [Antécédent de transfusion]": "Antécédent de transfusion",
            "Raison de non-eligibilité totale  [Porteur(HIV,hbs,hcv)]": "Porteur (HIV, Hbs, Hcv)",
            "Raison de non-eligibilité totale  [Opéré]": "Opéré",
            "Raison de non-eligibilité totale  [Drepanocytaire]": "Drepanocytaire",
            "Raison de non-eligibilité totale  [Diabétique]": "Diabétique",
            "Raison de non-eligibilité totale  [Hypertendus]": "Hypertendus",
            "Raison de non-eligibilité totale  [Asthmatiques]": "Asthmatiques",
            "Raison de non-eligibilité totale  [Cardiaque]": "Cardiaque",
            "Raison de non-eligibilité totale  [Tatoué]": "Tatoué",
            "Raison de non-eligibilité totale  [Scarifié]": "Scarifié" 
            
        }
    
    def combine(row):
        reason = []
        

        # Vérification de chaque colonne
        for col, label in columns_mapping.items():
            if row.get(col, 'Non') != 'Non':
                reason.append(label)

        # Ajout des raisons personnalisées si renseignées
        if row.get('Si autres raison préciser', 'Non') != 'Non':
            reason.append(row['Si autres raison préciser'])
        if row.get("Autre raisons,  preciser", 'Non') != 'Non':
            reason.append(row["Autre raisons,  preciser"])

        return ', '.join(reason) if reason else "aucune"
    
    

    # Application de la fonction
    df['raison_indisponibilite'] = df.apply(combine, axis=1) 
    
    # 5️⃣ Suppression des colonnes inutiles après regroupement
    df = df.drop(columns=list(columns_mapping.keys()) + [
        'Si autres raison préciser', 'Autre raisons,  preciser'
    ], errors='ignore')
    
    return df




def normalize_dates(df, column_name):
    """
    Normalise les dates dans la colonne spécifiée d'un DataFrame.
    
    Étapes :
    1. Remplace les années incorrectes (ex: 0019) par 2019.
    2. Convertit les valeurs en datetime.
    3. Remplace toutes les années qui ne sont pas 2019 par 2019.

    :param df: DataFrame contenant la colonne de dates.
    :param column_name: Nom de la colonne à traiter.
    :return: DataFrame avec la colonne normalisée.
    """
    
    # Fonction de nettoyage pour uniformiser les dates
    def clean_date(x):
        if isinstance(x, str):
            # Remplacer toute année de deux chiffres invalides par 2019
            x = re.sub(r'^\d{2}(?=\d{2})', '2019', x)  
            # Remplacer explicitement les cas avec '0019' par '2019'
            x = re.sub(r'(\d{1,2}/\d{1,2}/)0019', r'\g<1>2019', x)
        return x

    # Appliquer le nettoyage initial
    df[column_name] = df[column_name].apply(clean_date)

    # Convertir la colonne en datetime, avec gestion des erreurs
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')

    # Remplacer l'année par 2019 si elle n'est pas 2019
    df[column_name] = df[column_name].apply(lambda x: x.replace(year=2019) if pd.notna(x) and x.year != 2019 else x)
    
    df[column_name].fillna(df[column_name].mode()[0], inplace=True)

    return df

def clean_and_impute_age(df, date_col='Date de naissance', profession_col='Profession', ref_date='2019-12-31'):
    """
    Nettoie et impute la colonne 'age' :
    - Convertit 'Date de naissance' en datetime.
    - Calcule l'âge en années à partir de la date de référence.
    - Remplace les âges aberrants (<15 ou >70) par NaN.
    - Remplace les NaN par la médiane des âges de la même profession.
    - Remplace les NaN restants par la médiane globale.
    - Convertit la colonne 'age' en entier.

    :param df: DataFrame contenant les colonnes 'Date de naissance' et 'Profession'
    :param date_col: Nom de la colonne contenant les dates de naissance
    :param profession_col: Nom de la colonne contenant la profession
    :param ref_date: Date de référence pour le calcul de l'âge
    :return: DataFrame mis à jour avec une colonne 'age' en entier
    """
    
    # Convertir la colonne 'Date de naissance' en format datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Définir la date de référence
    date_reference = pd.Timestamp(ref_date)

    # Calcul de l'âge
    df['age'] = (date_reference - df[date_col]).dt.days // 365.25

    # Filtrer les âges aberrants
    df.loc[(df['age'] < 15) | (df['age'] > 70) | (df['age'].isna()), 'age'] = np.nan

    # Calculer la médiane des âges par profession
    mean_age_by_profession = df.groupby(profession_col)['age'].median()

    # Remplacer les NaN dans 'age' par la médiane de la profession
    df.loc[df['age'].isna(), 'age'] = df.loc[df['age'].isna(), profession_col].map(mean_age_by_profession)

    # Remplacer les NaN restants par la médiane globale
    df['age'].fillna(df['age'].median(), inplace=True)

    # Convertir en entier
    df['age'] = df['age'].astype(int)

    return df



def normaliser_ras(valeur):
        """ Regroupe toutes les variations de 'RAS' en une seule valeur standardisée. """
        patterns_ras = [
            r'(?i)^\s*(R[\s.]*A[\s.]*S|Rien|Aucun|Non précisé|Pas précisé|Pas mentionné)|Non precisé|Pas precise\s*$'
        ]
        return 'RAS' if any(re.match(pattern, valeur) for pattern in patterns_ras) else valeur.strip()
def normalize_quarter_name(name):
        """ Nettoie et uniformise les noms de quartiers. """
        name = unidecode(str(name)).upper().strip()
        name = re.sub(r'[^A-Z0-9 ]', '', name)  # Supprime caractères spéciaux
        return re.sub(r'\s+', ' ', name)  # Supprime espaces multiples
    
def imputer_poids(data):
    if pd.isna(data['Poids']):  # Vérifie si la valeur est manquante
        if data['ÉLIGIBILITÉ AU DON.'] == 'Eligible':
            return np.random.randint(60, 100)  # Poids entre 60 et 150
        else: 
            return np.nan
    return data['Poids']

def imputer_tauxhemoglobine(data):
    if pd.isna(data['Taux d’hémoglobine']):  # Vérifie si la valeur est manquante
        if data['Genre'] == 'Homme':
            if data['ÉLIGIBILITÉ AU DON.'] == 'Temporairement Non-eligible':
                return np.random.uniform(6, 13)  # Valeur aléatoire entre 6 et 13
            elif data['ÉLIGIBILITÉ AU DON.'] == 'Eligible' :
                return np.random.uniform(13, 18)
            else: 
                return np.nan
        else:  # Femme
            if data['ÉLIGIBILITÉ AU DON.'] == 'Temporairement Non-eligible':
                return np.random.uniform(5, 12)  # Valeur aléatoire entre 5 et 12
            elif data['ÉLIGIBILITÉ AU DON.'] == 'Eligible' :
                return np.random.uniform(12, 18)
            else:  
                return np.nan
    return data['Taux d’hémoglobine']

    

def nettoyer_donnees(df, seuil_similarite_quartier=85, seuil_similarite_profession=85):
    """
    Pipeline complet de nettoyage et correction des données :
    
    1️⃣ Nettoyage des valeurs manquantes et normalisation des valeurs RAS
    2️⃣ Normalisation et regroupement des quartiers (fuzzy matching)
    3️⃣ Correction et standardisation des arrondissements
    4️⃣ Harmonisation des nationalités
    5️⃣ Classification des religions
    6️⃣ Regroupement et standardisation des professions (fuzzy matching)
    7️⃣ Imputation des valeurs manquantes (Taille, Poids, Taux d’hémoglobine)
    8️⃣ Traitement des données d’éligibilité au don
    9️⃣ Nettoyage et imputation de l’âge
    🔟 Suppression des colonnes inutiles
    1️⃣1️⃣ Normalisation des dates
    """
    
     # Fixer la seed pour la reproductibilité des résultats
    np.random.seed(42)
    
    # 1️⃣ Nettoyage des valeurs manquantes et normalisation des RAS
    colonnes_a_normaliser = ['Nationalité', 'Religion', 'Quartier de Résidence', 'Arrondissement de résidence']
    for col in colonnes_a_normaliser:
        df[col] = df[col].astype(str).apply(normaliser_ras)
    
    # 2️⃣ Normalisation des noms de quartiers
    df['Quartier de Résidence'] = df['Quartier de Résidence'].apply(normalize_quarter_name)
    
    # 3️⃣ Regroupement des quartiers similaires (Fuzzy Matching)
    quartiers_uniques = {}

    def trouver_nom_canonique(nom):
        if not quartiers_uniques:
            quartiers_uniques[nom] = nom
            return nom
        match, score = process.extractOne(nom, quartiers_uniques.keys(), scorer=fuzz.token_sort_ratio)
        return quartiers_uniques[match] if score >= seuil_similarite_quartier else quartiers_uniques.setdefault(nom, nom)

    df['Quartier de Résidence'] = df['Quartier de Résidence'].apply(trouver_nom_canonique)
    
    # 4️⃣ Correction et standardisation des arrondissements
    normalization_dict = {
        r'(?i)^\s*Deido\s*$': 'Douala 1',
        r'(?i)^\s*(Ngodi Bakoko|OYACK|BOKO)\s*$': 'Douala 3',
        r'(?i)^\s*(Yaound[eé]|Nkouabang)\s*$': 'Yaoundé',
        r'(?i)^\s*BUEA\s*$': 'Buea',
        r'(?i)^\s*Bafoussam\s*$': 'Bafoussam',
        r'(?i)^\s*TIKO\s*$': 'Tiko',
        r'(?i)^\s*LIMBE\s*$': 'Limbe'
    }
    df['Arrondissement de résidence'] = df['Arrondissement de résidence'].replace(normalization_dict, regex=True)

    # Correction des valeurs erronées en utilisant l'arrondissement majoritaire du quartier
    valeurs_erronees = {'Douala (Non précisé )', 'Douala 6'}
    arrondissements_majoritaires = df.loc[~df['Arrondissement de résidence'].isin(valeurs_erronees)] \
        .groupby('Quartier de Résidence')['Arrondissement de résidence'] \
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None) \
        .dropna().to_dict()
    
    df.loc[df['Arrondissement de résidence'].isin(valeurs_erronees), 'Arrondissement de résidence'] = \
        df['Quartier de Résidence'].map(arrondissements_majoritaires).fillna(df['Arrondissement de résidence'])
    
    # 5️⃣ Normalisation des nationalités
    nationalite_dict = {
        'Centrafricaine': 'CENTRAFRICAINE',
        'Malien': 'MALIEN',
        'Malienne': 'MALIEN',
        'AMERICAINE': 'AMÉRICAINE',
        'Tchadienne': 'TCHADIENNE'
    }
    df['Nationalité'] = df['Nationalité'].replace(nationalite_dict)

    # 6️⃣ Classification des religions
    conditions = [
        df['Religion'].str.contains(r'(?i)chr[eé]tien|baptist|presbyt[eé]rien|pentec[oô]tiste|adventiste|epc|cmci|uebc|croyant|pantecotiste', na=False),
        df['Religion'].str.contains(r'(?i)musulman', na=False),
        df['Religion'].str.contains(r'(?i)non\s*croyant|la[iï]c|aucune|r\s*a\s*s|lo[iï]c|non\s*pr[eé]cis[eé]', na=False),
        df['Religion'].str.contains(r'(?i)animiste|traditionaliste|crois en tout|loique', na=False)
    ]
    categories = ['CHRETIEN', 'MUSULMAN', 'NON-RELIGIEUX', 'AUTRES']
    df['Religion'] = np.select(conditions, categories, default='RAS')
    
    # 7️⃣ Normalisation et regroupement des professions
    def normalize_profession(profession):
        profession = unidecode(str(profession)).upper().strip()
        profession = re.sub(r'[^A-Z0-9 ]', '', profession)
        return re.sub(r'\s+', ' ', profession)
    
    df['Profession'] = df['Profession'].astype(str).str.strip()
    patterns_no_profession = [r'^\s*AUCUN\s*$', r'^\s*RIEN\s*$', r'^\s*SANS\s*$', r'^\s*PAS DE PROFESSION\s*$']
    df['Profession'].replace('|'.join(patterns_no_profession), 'SANS PROFESSION', regex=True, inplace=True)
    df['Profession'] = df['Profession'].apply(normalize_profession)
    
    def regrouper_professions(df, colonne, seuil_similarite):
        professions_uniques = {}
        def trouver_nom_canonique(nom):
            if not professions_uniques:
                professions_uniques[nom] = nom
                return nom
            match, score = process.extractOne(nom, professions_uniques.keys(), scorer=fuzz.token_sort_ratio)
            return professions_uniques[match] if score >= seuil_similarite else professions_uniques.setdefault(nom, nom)
        
        df[colonne] = df[colonne].apply(trouver_nom_canonique)
        return df
    
    df = regrouper_professions(df, 'Profession', seuil_similarite_profession)
    
    # 8️⃣ Imputation des valeurs manquantes pour Taille, Poids et Taux d’hémoglobine
    df['Taille'] = df['Taille'].apply(lambda x: np.random.uniform(1.5, 2) if pd.isna(x) else x)
    df.loc[df['Taille'] >= 3, 'Taille'] /= 100  # Conversion des tailles en mètres
    # df.loc[(df['Taille'] >= 3), 'Taille'] = np.nan
    # df['Taille'] = df['Taille'].fillna(df['Taille'].median())

    df['Taux d’hémoglobine'] = df.apply(imputer_tauxhemoglobine, axis=1)

    df['Poids'] = df.apply(imputer_poids, axis=1) 
    df['Poids'] = df['Poids'].astype(float)
    
    #  Calcul de la moyenne des poids par arrondissement (en ignorant les NaN)
    mediane_par_arr = df.groupby('Arrondissement de résidence')['Poids'].median()

    def remplacer_par_mediane(row):
        if pd.isna(row['Poids']):
            return mediane_par_arr.get(row['Arrondissement de résidence'], np.nan)
        return row['Poids']

    df['Poids'] = df.apply(remplacer_par_mediane, axis=1)
    df['Poids'] = df['Poids'].fillna(df['Poids'].median())  # Remplacer les NaN restants par la médiane globale

    
    df['Taux d’hémoglobine'] = df['Taux d’hémoglobine'].astype(str).str.replace(',', '.').str.extract(r'(\d+\.?\d*)').astype(float)
    df.loc[df['Taux d’hémoglobine'] > 20, 'Taux d’hémoglobine'] /= 10  # Valeurs aberrantes
    
    mediane_hemo_par_arr = df.groupby('Arrondissement de résidence')['Taux d’hémoglobine'].median()

    # Fonction pour imputer le taux d'hémoglobine avec la médiane de l'arrondissement
    def imputer_hemo(row):
        if pd.isna(row['Taux d’hémoglobine']):
            return mediane_hemo_par_arr.get(row['Arrondissement de résidence'], np.nan)
        return row['Taux d’hémoglobine']

    # Appliquer l'imputation sur les valeurs manquantes
    df['Taux d’hémoglobine'] = df.apply(imputer_hemo, axis=1)

    # 9️⃣ Traitement des données d'éligibilité au don
    df = preprocess_eligibility_data(df)
    
    # 🔟 Nettoyage et imputation de l'âge
    df = clean_and_impute_age(df)
    
    # 1️⃣1️⃣ Suppression des colonnes inutiles
    df.drop(columns=['Si oui preciser la date du dernier don.', 'Date de naissance'], errors='ignore', inplace=True)
    
    # 1️⃣2️⃣ Normalisation des dates
    df = normalize_dates(df, 'Date de remplissage de la fiche')
    df.columns = df.columns.str.replace(" ", "_")

    return df