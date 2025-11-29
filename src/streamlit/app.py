import streamlit as st

st.set_page_config(
    page_title="Mon Application Streamlit",
    layout="wide",
    initial_sidebar_state="expanded",
)   

# Container Titre 
container_Title = st.container(border=True)
with container_Title:
    st.title("Mon Application Streamlit")
    st.markdown("Bienvenue dans mon application Streamlit ! Utilisez la barre latérale pour naviguer.")
    

