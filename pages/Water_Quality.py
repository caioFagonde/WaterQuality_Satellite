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
import ee
import rasterio

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

Map = leafmap.Map()
Map.add_raster("dissolved_oxygen.tif",layer_name = "Dissolved Oxygen")
Map
