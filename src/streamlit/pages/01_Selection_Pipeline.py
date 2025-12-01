import streamlit as st
import os 
import joblib
import time
# Ajouter le répertoire racine du projet au chemin Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
src_path = os.path.join(project_root, 'src')

# print(f"Project root: {project_root}")
# print(f"Source path: {src_path}")

# from src.features.Pipelines.transformateurs.BaseTransform import *
from features.Pipelines.transformateurs.BaseTransform import *
from features.Pipelines.transformateurs.Image_path_loader import *
from features.Pipelines.transformateurs.Image_Analyser import *
from features.Pipelines.transformateurs.tuple_to_df import *

save_dir_paths = os.path.join(project_root, 'models')


# Container Titre 
container_Title = st.container(border=True)
with container_Title:
    st.title("Sélection Pipeline")
    st.markdown("Bienvenue dans la page de sélection du pipeline ! Utilisez la barre latérale pour naviguer.")
    

# container Sélection Pipeline
container_Selection_Pipeline = st.container(border=True)
with container_Selection_Pipeline:
    
    pkl_files = [ f for f in os.listdir(save_dir_paths) if f.endswith('.pkl')]
    # st.write(pkl_files)

    col1, col2 = st.columns([1,3])
    if pkl_files:
        with col1:
            st.markdown("### Séléction du pipeline :")
            selected_pipeline = st.selectbox("Sélectionnez un pipeline enregistré :", pkl_files)
            st.success(f"Vous avez sélectionné le pipeline : **{selected_pipeline}**")


        with col2:
            st.markdown("### Détails du pipeline sélectionné :")
            # Afficher les détails du pipeline sélectionné
            job_to_load = os.path.join(save_dir_paths, selected_pipeline)
            st.info(f"Chemin absolu du pipeline : {job_to_load}")

            
            try:
                st.write(job_to_load)
                loaded_pipeline = joblib.load(job_to_load)
                st.code(loaded_pipeline)
                                  
            except Exception as e:
                st.error(f"Erreur lors du chargement du pipeline : {e}")
                loaded_pipeline = None
        
        with col1:
            launch_button = st.button(f"Lancer Pipeline : {selected_pipeline}")
        
        
            
    else:
        st.warning(f"Aucun pipeline enregistré trouvé dans le dossier '{save_dir_paths}'. Veuillez enregistrer un pipeline avant de le sélectionner.")

if launch_button and loaded_pipeline is not None:         
    container_Execution_Pipe = st.container(border=True)
    with container_Execution_Pipe:
        st.title(f"Exécution du pipeline : **{selected_pipeline}**")
        st.divider()
        with st.spinner(f"Exécution du pipeline : **{selected_pipeline}** en cours..."):
            # time.sleep(1)  # Simuler un délai pour l'effet visuel
            # st.success(f"Exécution du pipeline : **{selected_pipeline}**")
            # st.info("Chargement des données et exécution du pipeline en cours...")
            X = loaded_pipeline.fit_transform(X=None)
            st.divider()
            st.success(f"Exécution du pipeline : **{selected_pipeline}** terminée avec succès!")
            # st.info(f"Résultat de la transformation : {X.__class__} avec {len(X)} éléments.")             
    