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

row1_col1, row1_col2 = st.columns([2, 1])

with row1_col1:
    cog_list = load_cog_list()
    cog = st.selectbox("Selecione um parâmetro de qualidade.", cog_list)

with row1_col2:
    empty = st.empty()

    url = empty.text_input(
        "Enter a HTTP URL to a Cloud Optimized GeoTIFF (COG)",
        cog,
    )

    if url:
        try:
            options = leafmap.cog_bands(url)
        except Exception as e:
            st.error(e)
        if len(options) > 3:
            default = options[:3]
        else:
            default = options[0]
        bands = st.multiselect("Select bands to display", options, default=options)

        if len(bands) == 1 or len(bands) == 3:
            pass
        else:
            st.error("Please select one or three bands")

    add_params = st.checkbox("Add visualization parameters")
    if add_params:
        vis_params = st.text_area("Enter visualization parameters", "{}")
    else:
        vis_params = {}

    if len(vis_params) > 0:
        try:
            vis_params = eval(vis_params)
        except Exception as e:
            st.error(
                f"Invalid visualization parameters. It should be a dictionary. Error: {e}"
            )
            vis_params = {}

    submit = st.button("Submit")

m = leafmap.Map(latlon_control=False)

if submit:
    if url:
        try:
            m.add_cog_layer(url, bands=bands, **vis_params)
        except Exception as e:
            with row1_col2:
                st.error(e)
                st.error("Work in progress. Try it again later.")

with row1_col1:
    m.to_streamlit()