# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 04:47:14 2023

@author: caion
"""

import os
import leafmap.foliumap as leafmap
import leafmap.colormaps as cm
import streamlit as st
import geemap

st.set_page_config(layout="wide")

st.sidebar.title("Sobre")
st.sidebar.info(
    """
    Web App URL: <https://watergeospatial.streamlit.app/>
    """
)

st.sidebar.title("Contato")
st.sidebar.info(
    """
    Quasar Space: <https://quasarspace.com.br>
    """
)


@st.cache
def load_cog_list():
    cog_list = ['Chlorofila-a','Profundidade de Secchi','Turbidez','Batimetria','Oxigênio Dissolvido']
    return cog_list


@st.cache
def get_palettes():
    return list(cm.palettes.keys())
    # palettes = dir(palettable.matplotlib)[:-16]
    # return ["matplotlib." + p for p in palettes]

@st.cache(persist=True)
def ee_authenticate(token_name="EARTHENGINE_TOKEN"):
    geemap.ee_initialize(token_name=token_name)

st.title("Visualizando parâmetros hídricos")
st.markdown(
    """
Um aplicativo para análise de corpos de água e obtenção de seus parâmetros de qualidade.
"""
)

ee_authenticate()
cog_list = load_cog_list()
cog = st.selectbox("Selecione um parâmetro de qualidade.", cog_list)

Map = geemap.Map(center=[40,-100], zoom=4)
Map