import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
import time
from folium.plugins import HeatMap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
# from sklearn.preprocessing import LabelEncoder

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


# Chargement du fichier uniquement si les données ne sont pas encore en session
if 'df' not in st.session_state:
    uploaded_file = st.file_uploader("Choisissez un fichier excel", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df = nettoyer_donnees(df)
        st.session_state.df = df 

if 'df' in st.session_state:
    df = st.session_state.df
    
    # --- DASHBOARD AVEC MENU LATÉRAL ---
    st.sidebar.image("logo.png")  # Ajoute un logo (facultatif)
    st.sidebar.title("📌 Menu de Navigation")

    page = st.sidebar.radio("📍 ", ["Accueil", "Impact des Conditions de Santé", "Profilage des Donneurs Idéaux", "Analyse de l’Efficacité des Campagnes", "Fidélisation des Donneurs", "Cartographie des Donneurs", "Prédiction"])

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
            st.write("Sélectionnez des critères pour regrouper les donneurs en profils similaires.")
            
            all_features = ["age", "Genre", "Profession", "Taille", "Poids"]
            selected_features = st.multiselect("📌 Sélectionnez les caractéristiques", all_features, default=["age", "Genre"])
            
            n_clusters = st.slider("🔢 Choisissez le nombre de clusters", 2, 10, 3)
            
            if not selected_features:
                st.warning("⚠️ Veuillez sélectionner au moins une caractéristique.")
            else:
                df_cluster = df[selected_features].copy()
                df_cluster = pd.get_dummies(df_cluster)
                scaler = StandardScaler()
                df_scaled = scaler.fit_transform(df_cluster)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df["Cluster"] = kmeans.fit_predict(df_scaled)
                
                x_axis, y_axis = selected_features[:2] if len(selected_features) >= 2 else (selected_features[0], selected_features[0])
                fig = px.scatter(df, x=x_axis, y=y_axis, color=df["Cluster"].astype(str), title=f"Clustering ({n_clusters} groupes)",
                                labels={"Cluster": "Groupe de Donneurs"}, hover_data=selected_features)
                st.plotly_chart(fig)

    # --- PAGE 3 : Analyse de l’Efficacité des Campagnes ---
    elif page == "Analyse de l’Efficacité des Campagnes":
        if df is not None:
            st.title("📅 Analyse de l’Efficacité des Campagnes")
            st.write("Étude des tendances des dons de sang au fil du temps et identification des groupes démographiques les plus actifs.")
            
            df["Année"] = df["Date_de_remplissage_de_la_fiche"].dt.year
            df["Mois"] = df["Date_de_remplissage_de_la_fiche"].dt.strftime("%B")
            
            # Nombre de dons par mois
            # Filtrer uniquement les personnes ayant déjà donné leur sang
            df_filtre = df[df["A-t-il_(elle)_déjà_donné_le_sang"] == "Oui"]

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

            # Filtrage des données
            df_filtered = df[
                (df["age"].between(age_range[0], age_range[1])) &
                (df["Profession"].isin(professions_selection)) &
                (df["Arrondissement_de_résidence"].isin(arrondissements_selection))
            ]
            
            
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

                # Initialisation du géocodeur
                geolocator = Nominatim(user_agent="don_blood_mapper")

                @st.cache_data
                def get_coordinates(location):
                    try:
                        time.sleep(1)  # Éviter les limitations de l'API
                        location_data = geolocator.geocode(location + ", Cameroun")
                        if location_data:
                            return location_data.latitude, location_data.longitude
                    except:
                        return None, None

                # Géocodage des quartiers (exécution unique grâce à @st.cache_data)
                coordinates = df["Quartier_de_Résidence"].apply(lambda x: get_coordinates(x) if pd.notnull(x) else (None, None))
                df["latitude"] = coordinates.apply(lambda x: x[0] if x is not None else None)
                df["longitude"] = coordinates.apply(lambda x: x[1] if x is not None else None)


                # Création de la carte centrée sur une localisation moyenne
                m = folium.Map(location=[3.848, 11.502], zoom_start=12)

                # Ajouter des marqueurs pour chaque quartier
                for _, row in df.iterrows():
                    if pd.notnull(row["latitude"]) and pd.notnull(row["longitude"]):
                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=5,
                            popup=f"Quartier: {row['Quartier_de_Résidence']}<br>Donateurs: {row['A-t-il_(elle)_déjà_donné_le_sang']}",
                            color="blue",
                            fill=True,
                            fill_color="blue",
                        ).add_to(m)

                # Ajouter une Heatmap pour visualiser la densité des donneurs
                heat_data = df[["latitude", "longitude"]].dropna().values.tolist()
                HeatMap(heat_data).add_to(m)

                # Interface Streamlit
                st.title("🌍 Cartographie des Donneurs de Sang")
                st.write("Visualisation interactive de la répartition des donneurs selon leur lieu de résidence.")

                # Affichage de la carte
                folium_static(m)
                
                
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
                
                st.subheader("📌 Résultat de la prédiction")
                st.write(f"La prédiction du modèle est : **{predictions[0]}**")
                # st.write(f"La prédiction du modèle est : **{label_mapping[predictions[0]]}**")
                
                # st.write(nouvelles_donnees)
