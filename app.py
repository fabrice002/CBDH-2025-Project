import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
import time
from folium.plugins import HeatMap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import joblib 
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import pickle 
import os
import locale

# Définir la langue en français
locale.setlocale(locale.LC_TIME, "fr_FR")


from cleaning import nettoyer_donnees

# CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Cartographie des Donneurs", layout="wide")

# Initialisation du géocodeur
geolocator = Nominatim(user_agent="geo_donneur")

# Fonction de géocodage avec mise en cache pour éviter les appels répétés
@st.cache_data
def get_coordinates(quartier):
    geolocator = Nominatim(user_agent="geoapiExercises")
    try:
        location = geolocator.geocode(f"{quartier}, Douala, Cameroun")
        if location:
            return location.latitude, location.longitude
    except GeocoderTimedOut:
        time.sleep(1)
        return get_coordinates(quartier)
    return None, None


# Définir les colonnes attendues
expected_columns = ['Date de remplissage de la fiche', 'Date de naissance',
    "Niveau d'etude", 'Genre', 'Taille', 'Poids',
    'Situation Matrimoniale (SM)', 'Profession',
    'Arrondissement de résidence', 'Quartier de Résidence', 'Nationalité',
    'Religion', 'A-t-il (elle) déjà donné le sang',
    'Si oui preciser la date du dernier don.', 'Taux d’hémoglobine',
    'ÉLIGIBILITÉ AU DON.',
    'Raison indisponibilité  [Est sous anti-biothérapie  ]',
    'Raison indisponibilité  [Taux d’hémoglobine bas ]',
    'Raison indisponibilité  [date de dernier Don < 3 mois ]',
    'Raison indisponibilité  [IST récente (Exclu VIH, Hbs, Hcv)]',
    'Date de dernières règles (DDR) ',
    'Raison de l’indisponibilité de la femme [La DDR est mauvais si <14 jour avant le don]',
    'Raison de l’indisponibilité de la femme [Allaitement ]',
    'Raison de l’indisponibilité de la femme [A accoucher ces 6 derniers mois  ]',
    'Raison de l’indisponibilité de la femme [Interruption de grossesse  ces 06 derniers mois]',
    'Raison de l’indisponibilité de la femme [est enceinte ]',
    'Autre raisons,  preciser', 'Sélectionner "ok" pour envoyer',
    'Raison de non-eligibilité totale  [Antécédent de transfusion]',
    'Raison de non-eligibilité totale  [Porteur(HIV,hbs,hcv)]',
    'Raison de non-eligibilité totale  [Opéré]',
    'Raison de non-eligibilité totale  [Drepanocytaire]',
    'Raison de non-eligibilité totale  [Diabétique]',
    'Raison de non-eligibilité totale  [Hypertendus]',
    'Raison de non-eligibilité totale  [Asthmatiques]',
    'Raison de non-eligibilité totale  [Cardiaque]',
    'Raison de non-eligibilité totale  [Tatoué]',
    'Raison de non-eligibilité totale  [Scarifié]',
    'Si autres raison préciser']


# Chargement du fichier uniquement si les données ne sont pas encore en session
if 'df' not in st.session_state:
    uploaded_file = st.file_uploader("Choisissez un fichier excel", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        # Vérification des colonnes
        missing_cols = [col for col in expected_columns if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in expected_columns]

        if missing_cols or extra_cols:
            st.error("⚠️ Le fichier doit contenir exactement les colonnes spécifiées.")
            if missing_cols:
                st.write("Colonnes manquantes :", missing_cols)
            if extra_cols:
                st.write("Colonnes non attendues :", extra_cols)
        else: 

            if df is not None:
                df = nettoyer_donnees(df)  # fonction à définir ailleurs
                st.session_state.df = df

if 'df' in st.session_state:
    df = st.session_state.df
    
    # --- DASHBOARD AVEC MENU LATÉRAL ---
    st.sidebar.image("logo.png")  # Ajoute un logo (facultatif)
    st.sidebar.title("📌 Menu de Navigation")

    page = st.sidebar.radio("📍CBDH 2025 ", ["Accueil", "Impact des Conditions de Santé", "Profilage des Donneurs Idéaux", "Analyse de l’Efficacité des Campagnes", "Fidélisation des Donneurs", "Cartographie des Donneurs", "Prédiction"])

    # --- PAGE D'ACCUEIL ---
    if page == "Accueil":
        
        # Injecter un fond personnalisé avec du CSS (seulement en mode déploiement)
        background_css = """
        <style>
            .stApp {
                background-image: url("logo.png");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }
        </style>
        """
        st.markdown(background_css, unsafe_allow_html=True)

        # CSS pour centrer le texte
        # CSS pour centrer le texte
        st.markdown(
            """
            <style>
                .centered-text {
                    display: flex;
                    justify-content: center;
                    text-align: center;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Titre centré
        st.markdown('<h1 class="centered-text">🩸 Tableau de Bord Don de Sang</h1>', unsafe_allow_html=True)

        # Texte centré avec guillemets échappés ou guillemets doubles pour éviter le conflit
        st.markdown('<p class="centered-text">Bienvenue sur le tableau de bord interactif du groupe &nbsp;  <strong>CBDH 2025</strong>.</p> <br><br>', unsafe_allow_html=True)

        # Vérification des données chargées
        if df is not None:
            # --- STATISTIQUES CLÉS ---
            total_donneurs = int(df.loc[df['A-t-il_(elle)_déjà_donné_le_sang'] == 'Oui'].shape[0])
            age_moyen = int(round(df["age"].mean(), 1))
            taux_eligibles = round((df["ÉLIGIBILITÉ_AU_DON."].str.lower() == "eligible").mean() * 100, 2)
            femmes = int(round((df["Genre"].str.lower() == "femme").mean() * 100, 2))
            hommes = 100 - femmes

            # --- AFFICHAGE DES INDICATEURS ---
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("🩸 Total Donneurs", total_donneurs)
            col2.metric("📊 Âge Moyen", age_moyen)
            col3.metric("✅ Éligibilité (%)", f"{taux_eligibles}%")
            col4.metric("👩‍🦰 Femmes (%)", f"{femmes}%")
            col5.metric("👨 Hommes (%)", f"{hommes}%")

            # --- APERÇU DES DONNÉES ---
            st.subheader("📋 Aperçu des Données")
            st.write(df.head(10))

            

        else:
            st.error("⚠️ Aucune donnée chargée. Vérifiez le fichier 'cleaned_data.csv'.")
        
        # st.write("🩸 Merci pour votre engagement envers le don de sang !")


    # --- PAGE 1 : Impact des Conditions de Santé ---
    elif page == "Impact des Conditions de Santé":
        if df is not None:
            st.title("📊 Impact des Conditions de Santé sur l'Éligibilité au Don")
            st.write("Sélectionnez un statut pour voir quelles conditions de santé influencent l'éligibilité au don.")
            
            selected_eligibility = st.selectbox("🎯 Choisir un statut", df['ÉLIGIBILITÉ_AU_DON.'].unique())
            df_filtered = df[df["ÉLIGIBILITÉ_AU_DON."] == selected_eligibility]
            
            df_counts = df_filtered["raison_indisponibilite"].value_counts().reset_index()
            df_counts.columns = ["Condition de Santé", "Nombre"]
            
            if df_counts.empty:
                st.warning("⚠️ Aucune donnée pour ce statut.")
            else:
                fig = px.bar(df_counts, x="Condition de Santé", y="Nombre", text="Nombre", color="Nombre",
                            color_continuous_scale="reds", labels={"Nombre": "Nombre de Donneurs"},
                            title=f"Conditions de Santé pour '{selected_eligibility}'")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)

    # --- PAGE 2 : Profilage des Donneurs Idéaux --- 
    elif page == "Profilage des Donneurs Idéaux":
        if df is not None:
            st.title("🔍 Profilage des Donneurs Idéaux")
            st.write("Utilisez des techniques de clustering pour regrouper les donneurs selon leurs caractéristiques.")

            all_features = ["age", "Genre", "Profession", "Taille", "Poids"]
            selected_features = st.multiselect("📌 Sélectionnez les caractéristiques", all_features, default=["age", "Genre"])

            method = st.selectbox("🧠 Méthode de clustering", ["K-Means", "Hierarchical", "DBSCAN"])
            

            if not selected_features:
                st.warning("⚠️ Veuillez sélectionner au moins une caractéristique.")
            else:
                df4 = df.copy()
                # Filtrer les colonnes sélectionnées
                df_cluster = df4[selected_features].copy()

                # Encodage des variables catégorielles
                df_cluster_encoded = pd.get_dummies(df_cluster)
                scaler = StandardScaler()
                df_scaled = scaler.fit_transform(df_cluster_encoded)

                # Clustering
                if method == "K-Means":
                    n_clusters = st.slider("🔢 Nombre de clusters (sauf DBSCAN)", 2, 10, 3)
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = model.fit_predict(df_scaled)
                elif method == "Hierarchical":
                    n_clusters = st.slider("🔢 Nombre de clusters (sauf DBSCAN)", 2, 10, 3)
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                    cluster_labels = model.fit_predict(df_scaled)
                elif method == "DBSCAN":
                    model = DBSCAN(eps=1.5, min_samples=5)
                    cluster_labels = model.fit_predict(df_scaled)

                # Ajout des résultats au dataframe
                df4["Cluster"] = cluster_labels

                # Silhouette score (si applicable)
                if method != "DBSCAN" and len(set(cluster_labels)) > 1:
                    score = silhouette_score(df_scaled, cluster_labels)
                    st.metric("📏 Silhouette Score", round(score, 2))

                    # Interprétation du score
                    if score > 0.7:
                        interpretation = "🌟 Excellent regroupement des clusters."
                    elif 0.5 < score <= 0.7:
                        interpretation = "✅ Regroupement correct."
                    elif 0.25 < score <= 0.5:
                        interpretation = "⚠️ Regroupement moyen, peut être amélioré."
                    else:
                        interpretation = "❌ Mauvais regroupement, clusters peu fiables."

                    st.info(f"**Interprétation :** {interpretation}")
                else:
                    st.info("Silhouette Score non applicable pour DBSCAN ou un seul cluster détecté.")


                # Visualisation 2D avec PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(df_scaled)
                df4['PCA1'] = pca_result[:, 0]
                df4['PCA2'] = pca_result[:, 1]

                fig = px.scatter(df4, x="PCA1", y="PCA2", color=df4["Cluster"].astype(str),
                                title=f"Visualisation des Clusters ({method})",
                                labels={"Cluster": "Groupe"}, hover_data=selected_features)
                st.plotly_chart(fig)

                # Affichage des moyennes par cluster
                st.subheader("📊 Caractéristiques par Cluster")

                # Séparer les colonnes numériques et non numériques
                numeric_cols = df4[selected_features].select_dtypes(include=np.number).columns.tolist()
                non_numeric_cols = [col for col in selected_features if col not in numeric_cols]

                # Agrégation
                agg_dict = {}

                if numeric_cols:
                    agg_numeric = df4.groupby("Cluster")[numeric_cols].mean().round(2)
                    st.write("📈 Moyennes des variables numériques par cluster")
                    st.dataframe(agg_numeric)

                if non_numeric_cols:
                    agg_non_numeric = df4.groupby("Cluster")[non_numeric_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
                    st.write("📋 Modalités les plus fréquentes des variables catégorielles par cluster")
                    st.dataframe(agg_non_numeric)
        else:
            st.warning("Aucune donnée chargée.")

    # --- PAGE 3 : Analyse de l’Efficacité des Campagnes ---
    elif page == "Analyse de l’Efficacité des Campagnes":
        if df is not None:
            st.title("📅 Analyse de l’Efficacité des Campagnes")
            st.write("Étude des tendances des dons de sang au fil du temps et identification des groupes démographiques les plus actifs.")
            df2 = df.copy()
            df2["Année"] = df2["Date_de_remplissage_de_la_fiche"].dt.year
            df2["Mois"] = df2["Date_de_remplissage_de_la_fiche"].dt.strftime("%B")
            
            # Nombre de dons par mois
            # Filtrer uniquement les personnes ayant déjà donné leur sang
            df_filtre = df2[df2["A-t-il_(elle)_déjà_donné_le_sang"] == "Oui"]

            # Nombre de dons par mois
            monthly_donations = df_filtre.groupby("Mois").size().reset_index(name="Nombre de Dons")
            fig1 = px.line(monthly_donations, x="Mois", y="Nombre de Dons", markers=True,
                        title="📈 Nombre de dons par mois")
            st.plotly_chart(fig1)

            # Répartition des dons par Genre
            demographic_counts = df_filtre["Genre"].value_counts().reset_index()
            demographic_counts.columns = ["Genre", "Nombre de Dons"]
            fig2 = px.bar(demographic_counts, x="Genre", y="Nombre de Dons", text="Nombre de Dons",
                        color="Nombre de Dons", color_continuous_scale="blues",
                        title="🌍 Répartition des dons par Genre")
            st.plotly_chart(fig2)
            
            # Répartition par profession
            profession_counts = df["Profession"].value_counts().reset_index().head(10)
            profession_counts.columns = ["Profession", "Nombre de Dons"]
            fig3 = px.bar(profession_counts, x="Profession", y="Nombre de Dons", text="Nombre de Dons",
                        color="Nombre de Dons", color_continuous_scale="greens",
                        title="💼 Top 10 des professions des donneurs")
            st.plotly_chart(fig3)
            

    # --- PAGE 4 : Fidélisation des Donneurs ---
    elif page == "Fidélisation des Donneurs":
        if df is not None:
            st.title("🔄 Fidélisation des Donneurs")

            # Filtres interactifs
            age_range = st.sidebar.slider("📏 Filtrer par âge :", int(df["age"].min()), int(df["age"].max()), (18, 60))
            professions_selection = st.sidebar.multiselect("💼 Filtrer par profession :", df["Profession"].unique(), default=df["Profession"].unique()[:10])
            arrondissements_selection = st.sidebar.multiselect("📍 Filtrer par arrondissement :", df["Arrondissement_de_résidence"].unique(), default=df["Arrondissement_de_résidence"].unique()[:5])

            # Vérification que des filtres ont été sélectionnés
            if not professions_selection or not arrondissements_selection:
                st.warning("⚠️ Veuillez sélectionner au moins une profession et un arrondissement pour continuer l'analyse.")

            
            # Filtrage des données
            df_filtered = df[
                (df["age"].between(age_range[0], age_range[1])) &
                (df["Profession"].isin(professions_selection)) &
                (df["Arrondissement_de_résidence"].isin(arrondissements_selection))
            ]
            
             # Assurer qu'il y a des données après filtrage
            if df_filtered.empty:
                st.info("Aucun donneur ne correspond aux critères sélectionnés. Essayez de modifier les filtres.")
            else:
            
                # --- ANALYSE DE LA RÉCURRENCE ---
                st.subheader("📊 Répartition des donneurs par nombre de dons")
                don_rec_counts = df_filtered["A-t-il_(elle)_déjà_donné_le_sang"].value_counts()
                fig1 = px.bar(don_rec_counts, x=don_rec_counts.index, y=don_rec_counts.values,
                            labels={"x": "Nombre de dons", "y": "Nombre de donneurs"},
                            title="Distribution du nombre de dons par personne",
                            color=don_rec_counts.index)
                st.plotly_chart(fig1)
                
                # --- ANALYSE ÂGE ET PROFESSION ---
                st.subheader("📌 Analyse de l’âge et de la profession sur la fidélisation")
                fig2 = px.box(df_filtered, x="A-t-il_(elle)_déjà_donné_le_sang", y="age",
                            title="Analyse de l’âge des donneurs selon leur récurrence",
                            labels={"age": "Âge", "A-t-il_(elle)_déjà_donné_le_sang": "A déjà donné le sang"},
                            color="A-t-il_(elle)_déjà_donné_le_sang")
                st.plotly_chart(fig2)

                # Sélection des professions les plus représentées
                top_n = 15  # Nombre de professions à afficher
                profession_counts = df_filtered['Profession'].value_counts().head(top_n).index
                df_filtered = df_filtered[df_filtered['Profession'].isin(profession_counts)]

                # Création du graphique amélioré
                fig = px.bar(df_filtered, 
                            y="Profession",  # Graphique horizontal
                            color="A-t-il_(elle)_déjà_donné_le_sang",
                            title="Professions des donneurs et récurrence",
                            labels={"A-t-il_(elle)_déjà_donné_le_sang": "A déjà donné le sang"},
                            category_orders={"Profession": profession_counts},
                            # text_auto=True,  # Affichage des valeurs sur les barres
                            orientation="h")  # Affichage horizontal

                # Mise en forme du graphique
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})

                st.plotly_chart(fig, use_container_width=True)
                
                # --- ANALYSE GÉOGRAPHIQUE ---
                st.subheader("🌍 Analyse géographique des donneurs fidèles")
        
        if "Arrondissement_de_résidence" in df.columns and "A-t-il_(elle)_déjà_donné_le_sang" in df.columns:
            df_filtered = df.dropna(subset=["Arrondissement_de_résidence", "A-t-il_(elle)_déjà_donné_le_sang"])
            
            don_fidelity = df_filtered.groupby("Arrondissement_de_résidence")["A-t-il_(elle)_déjà_donné_le_sang"].value_counts(normalize=True).unstack()
            don_fidelity["Taux_de_fidélisation"] = don_fidelity.get("Oui", 0) * 100
            
            try:
                fig4 = px.choropleth(
                    don_fidelity.reset_index(),
                    geojson="arrondissements.geojson",
                    locations="Arrondissement_de_résidence",
                    featureidkey="properties.nom",
                    color="Taux_de_fidélisation",
                    title="Carte de fidélisation des donneurs par arrondissement",
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig4)
            except Exception as e:
                st.error(f"❌ Erreur lors de l'affichage de la carte choropleth: {e}")
        else:
            st.warning("⚠️ Les colonnes nécessaires ne sont pas présentes dans les données.")
    
    elif page == "Cartographie des Donneurs":
        
        if df is not None:
        # Vérifier les colonnes nécessaires
            required_columns = ["Arrondissement_de_résidence", "Quartier_de_Résidence"]

            if all(col in df.columns for col in required_columns):

                # Copie de sécurité
                df1 = df.copy()

                # Initialisation du géocodeur
                geolocator = Nominatim(user_agent="don_blood_mapper")

                # --- Cache local sur disque ---
                CACHE_FILE = "quartiers_coords.pkl"

                # Charger les coordonnées si elles existent
                if os.path.exists(CACHE_FILE):
                    with open(CACHE_FILE, "rb") as f:
                        coord_cache = pickle.load(f)
                else:
                    coord_cache = {}

                # Fonction de géocodage avec mise en cache
                def get_coordinates(location):
                    if location in coord_cache:
                        return coord_cache[location]
                    try:
                        time.sleep(1)  # Évite les limitations de Nominatim
                        location_data = geolocator.geocode(location + ", Cameroun")
                        if location_data:
                            coords = (location_data.latitude, location_data.longitude)
                        else:
                            coords = (None, None)
                    except:
                        coords = (None, None)

                    coord_cache[location] = coords

                    # Sauvegarde automatique
                    with open(CACHE_FILE, "wb") as f:
                        pickle.dump(coord_cache, f)

                    return coords

                # Géocodage des quartiers uniques uniquement
                unique_quartiers = df["Quartier_de_Résidence"].dropna().unique()
                quartier_coords = {q: get_coordinates(q) for q in unique_quartiers}

                # Attribution des coordonnées à chaque ligne
                df1["latitude"] = df["Quartier_de_Résidence"].apply(lambda x: quartier_coords.get(x, (None, None))[0])
                df1["longitude"] = df["Quartier_de_Résidence"].apply(lambda x: quartier_coords.get(x, (None, None))[1])

                # --- Carte interactive ---
                m = folium.Map(location=[3.848, 11.502], zoom_start=12)

                # Marqueurs individuels
                for _, row in df1.iterrows():
                    if pd.notnull(row["latitude"]) and pd.notnull(row["longitude"]):
                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=5,
                            popup=f"Quartier: {row['Quartier_de_Résidence']}<br>Donateur: {row['A-t-il_(elle)_déjà_donné_le_sang']}",
                            color="blue",
                            fill=True,
                            fill_color="blue",
                        ).add_to(m)

                # Heatmap
                heat_data = df1[["latitude", "longitude"]].dropna().values.tolist()
                HeatMap(heat_data).add_to(m)

                # Interface Streamlit
                st.title("🌍 Cartographie des Donneurs de Sang")
                st.write("Visualisation interactive de la répartition des donneurs selon leur lieu de résidence.")
                folium_static(m)

            # else:
            #     st.error("❌ Les colonnes nécessaires ne sont pas présentes dans le fichier.")
                
                
                # --- GRAPHIQUE DE RÉPARTITION DES DONNEURS ---
                st.subheader("📌 Répartition des Donneurs par Arrondissement")
                fig4 = px.histogram(
                    df,
                    x="Arrondissement_de_résidence",
                    title="Distribution des Donneurs par Arrondissement",
                    color="Genre"
                )
                st.plotly_chart(fig4, use_container_width=True)

            else:
                st.error("❌ Les colonnes requises ne sont pas présentes dans le fichier de données.")
    elif page == "Prédiction":
        if df is not None:
            st.title("🔮 Prédiction de l'Éligibilité au Don de Sang")

            # Chargement du modèle de prédiction
            model = joblib.load("modele_prediction.pkl")

            # Création d'un formulaire pour saisir les données
            if df is not None:
                st.write("Modifiez les valeurs des paramètres pour voir l'impact sur la prédiction")
                
                colonnes_model = [col for col in df.columns if col not in ["ÉLIGIBILITÉ_AU_DON.", 'latitude', 'longitude', 'raison_indisponibilite']]
                valeurs_utilisateur = {}
                
                for col in colonnes_model:
                    if (col == 'Arrondissement_de_résidence' ):  # Si la colonne est Arrondissement_de_résidence
                        arrondissement = st.selectbox(f"{col}", df[col].unique())
                        # Appliquer un filtre pour Quartier_de_Résidence selon l'Arrondissement choisi
                        quartiers_filtrés = df.loc[df['Arrondissement_de_résidence'] == arrondissement, 'Quartier_de_Résidence'].unique()
                        # Filtrer Quartier_de_Résidence
                        quartier = st.selectbox(f"Quartier de Résidence", quartiers_filtrés)
                        valeurs_utilisateur[col] = arrondissement
                        valeurs_utilisateur['Quartier_de_Résidence'] = quartier
                    elif df[col].dtype == 'object':
                        if col == 'Quartier_de_Résidence' :
                            continue
                        valeurs_utilisateur[col] = st.selectbox(f"{col}", df[col].unique())
                    elif df[col].dtype == 'datetime64[ns]':
                        continue  # Ignorer les colonnes Timestamp
                    else:
                        min_val, max_val = df[col].min(), df[col].max()
                        valeurs_utilisateur[col] = st.slider(f"{col}", float(min_val), float(max_val), float(df[col].median()))
                
                nouvelles_donnees = pd.DataFrame([valeurs_utilisateur])
                
                # Vérifier si les colonnes latitude et longitude existent avant de les supprimer
                cols_a_supprimer = [col for col in ['latitude', 'longitude'] if col in nouvelles_donnees.columns]
                nouvelles_donnees = nouvelles_donnees.drop(columns=cols_a_supprimer, errors='ignore') 
                
                predictions = model.predict(nouvelles_donnees)
                
                # Mapping des classes numériques vers les étiquettes
                label_mapping = {
                    0: "Définitivement non-eligible",
                    1: "Eligible",
                    2: "Temporairement Non-eligible"
                }
                
                st.subheader("📌 Résultat de la prédiction")
                st.write(f"La prédiction du modèle est : **{label_mapping[predictions[0]]}**")
                # st.write(f"La prédiction du modèle est : **{label_mapping[predictions[0]]}**")
                
                # st.write(nouvelles_donnees)
