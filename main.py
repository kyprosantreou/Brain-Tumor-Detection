import streamlit as st

about_page = st.Page("welcome_page.py", title="About", icon="ℹ️")
prediction_page = st.Page("predict.py", title="Brain MRI Analysis", icon="🧠")
clustering_page = st.Page("clustering.py", title="Clustering", icon="📊")

pg = st.navigation({
    "Informations":[about_page],
    "Tools":[prediction_page,clustering_page]
    })

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")

pg.run()