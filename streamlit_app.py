import streamlit as st
import leafmap.foliumap as leafmap

st.set_page_config(layout="wide")

st.sidebar.title("Contato")
st.sidebar.info(
    """
    Quasar Space: <https://quasarspace.com.br>
    """
)

# Customize page title
st.title("Análise de qualidade da água")

m = leafmap.Map(minimap_control=True)
m.add_basemap("OpenTopoMap")
m.to_streamlit(height=500)
