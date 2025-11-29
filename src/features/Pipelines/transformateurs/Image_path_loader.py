import os
import glob
import streamlit as st
import pandas as pd
import plotly.express as px

from .BaseTransform import BaseTransform

# # Définir une palette de couleurs personnalisée pour correspondance entre les graphiques
# color_map = {
#             'covid': '#636EFA',
#             'lung_opacity': '#EF553B',
#             'normal': '#00CC96',
#             'viral pneumonia': '#AB63FA'
#         }

class ImagePathLoader(BaseTransform):
    def __init__(self, root_dir="../data/raw/COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/", verbose=True, use_streamlit=True):
        super().__init__(verbose=verbose, use_streamlit=use_streamlit)
        self.root_dir = root_dir

    def fit(self, X=None, y=None):
        return super().fit(X, y)

    def transform(self, X=None):
        if self.use_streamlit :
            st.divider()
            st.info("🔄 Chargement des chemins d'images...")
        self.log("Chargement des chemins d'images...")
        # super().transform(X)
        X = self.load_paths_data_raw(self.root_dir)
        self.plot_distribution_labels(X[2])
        if self.use_streamlit :
            st.success(f"Nombre d'images chargées : {len(X[0])}")
        return X
    
    def load_paths_data_raw(self, root_dir):
        self.image_paths = []
        self.mask_paths = []
        self.labels = []

        labels_list = [label for label in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, label, 'images'))]
        
        for label in self.progress(labels_list, desc="Chargement des chemins d'images"):
            class_img_dir = os.path.join(root_dir, label, 'images')
            class_mask_dir = os.path.join(root_dir, label, 'masks')
            
            for img_path in glob.glob(os.path.join(class_img_dir, '*.png')):
                filename = os.path.basename(img_path)
                mask_path = os.path.join(class_mask_dir, filename)
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(label.lower())

        return self.image_paths, self.mask_paths, self.labels
    
    def plot_distribution_labels(self, labels,color_map=None):
        df = pd.DataFrame(labels, columns=['Label'])
        label_counts = df['Label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']

        if color_map is None:
            unique_labels = label_counts['Label'].unique()
            default_colors = px.colors.qualitative.Plotly
            color_map = {label: default_colors[i % len(default_colors)] for i, label in enumerate(unique_labels)}


        fig = px.bar(label_counts, x='Label', y='Count', title='Distribution des labels',
                     color='Label', color_discrete_map=color_map)
        
        circle_fig = px.pie(label_counts, names='Label', values='Count', 
                           title='Distribution des labels (Pie Chart)',
                           color='Label', color_discrete_map=color_map)
        
        # Améliorer la lisibilité du texte sur le pie chart
        circle_fig.update_traces(textposition='outside', textinfo='percent+label', 
                                 textfont_size=14)

        st.title("Distribution des labels:")
        col1,col2,col3 = st.columns([1,2,2])

        with col1:
            st.title("Distribution des labels:")
            st.table(label_counts)
        with col2:
            st.plotly_chart(fig, width='stretch')
        with col3:
            st.plotly_chart(circle_fig, width='stretch')
        return fig, circle_fig, pd.DataFrame(label_counts)