# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 04:47:14 2023

@author: caion
"""

import os
import time
import folium
from streamlit_folium import st_folium, folium_static
from localtileserver import get_leaflet_tile_layer
import leafmap.foliumap as leafmap
import leafmap.colormaps as cm
import streamlit as st
import geemap
import ee
import rasterio
import pandas
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ipyleaflet
import pickle
from ipyleaflet import Map, basemaps

from os import listdir
from os.path import isfile, join

from pyproj import Transformer


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

class Ponto(object):
    __slots__ = ('lat', 'lon','dep','do','name')
    def _init_(self):
        self.do = []
        self.dep = []

# Collection points
B = Ponto()
Six6 = Ponto()
P = Ponto()
Z = Ponto()

B.lat = -20.04405168135771 #[-50.933795227068416,-20.04405168135771]
B.lon = -50.933795227068416 #[-50.933795227068416,-20.04405168135771]
B.do = [5.4,6.5]
B.dep = [4.5, 4.1]
B.name = "Batelão"

Six6.lat = -20.041955238728764 #[-50.933795227068416,-20.04405168135771]
Six6.lon = -50.931821121233455 #[-50.933795227068416,-20.04405168135771]
Six6.do = [6.2,6.7]
Six6.dep = [3.8, 4]
Six6.name = "6x6"

P.lat = -20.04198547608097 #[-50.933795227068416,-20.04405168135771]
P.lon = -50.93346263315057 #[-50.933795227068416,-20.04405168135771]
P.do = [5.8,6.5]
P.dep = [4, 4.5]
P.name = "Porto"

Z.lat = -20.0431748073166 #[-50.933795227068416,-20.04405168135771]
Z.lon = -50.93011523629998 #[-50.933795227068416,-20.04405168135771]
Z.do = [5.9,6.7]
Z.dep = [4, 4.8]
Z.name = "Zippy"

points = [B,Six6,P,Z]

absolute_path = os.getcwd()
relative_path = "PuroPeixe/files/"
mypath = os.path.join(absolute_path, relative_path) 
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Load models
regr = pickle.load(open('model_do.pkl', 'rb'))
regr_secc = pickle.load(open('model_secc.pkl', 'rb'))
coefficients = np.transpose(regr.coef_) #coefficients = pandas.concat([pandas.DataFrame(X.columns),pandas.DataFrame()], axis = 1)    
coefficients_secc = np.transpose(regr_secc.coef_)
#@st.cache(allow_output_mutation=True)

def generate_tif(depth = 4, filename = "",img = ""):
    
    if filename == "":
        filename = mypath + onlyfiles[1]
    if img == "":
        img= onlyfiles[1]
    # List of filenames for tifs
    
    
    src = rasterio.open(filename)
    
    # Load red and NIR bands - note all PlanetScope 4-band images have band order BGRN
    with rasterio.open(filename) as src:
        band_red = src.read(6) * 0.0001
    
    with rasterio.open(filename) as src:
        band_nir = src.read(8) * 0.0001
    
    with rasterio.open(filename) as src:
        band_green = src.read(4) * 0.0001
        
    with rasterio.open(filename) as src:
        band_blue = src.read(2) * 0.0001
    
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    ## Some functions to obtain remote sensing reflectance
    strAux = img.split('/')[0]
    year = strAux[0:4]
    mon = strAux[4:6]
    day = strAux[6:8]
    H = strAux[9:11]
    M = strAux[11:13]
    fDate = day + "/" + mon + "/" + year + " " + H +":" + M
    fDate = datetime.datetime.strptime(fDate, '%d/%m/%Y %H:%M')
    doy = fDate.timetuple().tm_yday
    
    def surface_to_remote_reflectance(surface_reflectance, solar_zenith_angle):
        remote_sensing_reflectance = surface_reflectance / np.cos(solar_zenith_angle)
        return remote_sensing_reflectance
    
    declination = 23.45 * np.sin(np.deg2rad(360 * (284 + doy) / 365))
    solar_hour_angle = 15 * (float(H) - 12)
    latitude = np.deg2rad(B.lat)
    longitude = np.deg2rad(B.lon)
    solar_hour_angle = np.deg2rad(solar_hour_angle)
    zenith_angle = np.arccos(np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(declination) * np.cos(solar_hour_angle))
    
    band_blue_Rrs = surface_to_remote_reflectance(band_blue,zenith_angle)
    band_green_Rrs = surface_to_remote_reflectance(band_green,zenith_angle)
    band_red_Rrs = surface_to_remote_reflectance(band_red,zenith_angle)
    band_nir_Rrs = surface_to_remote_reflectance(band_nir,zenith_angle)
    
    big_rrs_blue = band_blue/3.1415926
    big_rrs_green = band_green/3.1415926
    big_rrs_red = band_red/3.1415926
    
    w = big_rrs_green - 0.46*big_rrs_red - 0.54*big_rrs_blue
    
    rrs_vecblue = 1000*np.divide(big_rrs_blue,(1.7*big_rrs_blue  + 0.52))
    rrs_vecgreen = 1000*np.divide(big_rrs_green,(1.7*big_rrs_green  + 0.52))
    
    lrrs_vecblue = np.log(rrs_vecblue)
    lrrs_vecgreen = np.log(rrs_vecgreen)
    
    chla = np.power(10,-0.4909 + 191.659*w)#0.5
    m0 = 52.073*np.exp(0.957*chla)
    m1 = 50.156*np.exp(0.957*chla)
    
    # Calculate Dissolved Oxygen from regression equation
    secc = regr_secc.intercept_[0] + coefficients_secc[0] *band_blue.astype(float) + coefficients_secc[1]*band_green.astype(float) + coefficients_secc[2]*band_red.astype(float) + coefficients_secc[3]*band_nir.astype(float)
    dissolved_oxygen = regr.intercept_[0] + coefficients[0]*depth + coefficients[1] *band_blue.astype(float) + coefficients[2]*band_green.astype(float) + coefficients[3]*band_red.astype(float) + coefficients[4]*band_nir.astype(float) + coefficients[5]*secc.astype(float)
    dissolved_oxygen[dissolved_oxygen < 0] = 0
    dissolved_oxygen[dissolved_oxygen > 20] = 20
    
    # bathymetry
    a0 = -3.24
    a1 = 14.72
    a2 = -18.48
    bathymetry = m0 * np.divide(lrrs_vecblue,lrrs_vecgreen) - m1#a0 + a1*np.log(band_blue) + a2*np.log(band_green)
    #bathymetry = bathymetry/np.nanmax(bathymetry)
    # NDWI for masking
    ndwi = (band_green.astype(float) - band_nir.astype(float))/(band_green.astype(float) + band_nir.astype(float))
    dissolved_oxygen[ndwi < -0.4] = np.nan
    bathymetry[ndwi < -0.4] = np.nan
    bathymetry[bathymetry < 0] = 0
    #bathymetry[bathymetry > 20] = np.nan
    # check range values, excluding NaN
    #np.nanmin(dissolved_oxygen), np.nanmax(dissolved_oxygen)
    #np.nanmin(dissolved_oxygen), np.nanmax(dissolved_oxygen)
    
    # Set spatial characteristics of the output object to mirror the input
    kwargs = src.meta
    kwargs.update(
        dtype=rasterio.float32,
        count = 1)
    
    # Write band calculations to a new raster file
    with rasterio.open('dissolved_oxygen.tif', 'w', **kwargs) as dst:
        dst.write_band(1, dissolved_oxygen.astype(rasterio.float32))
    with rasterio.open('bathymetry.tif', 'w', **kwargs) as dst:
        dst.write_band(1, bathymetry.astype(rasterio.float32))
    
    
    # In[96]:
    
    
    
    
    """
    
    """
    
    class MidpointNormalize(colors.Normalize):
        """
        Normalise the colorbar so that diverging bars work their way either side from a prescribed midpoint value)
        e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
        Credit: Joe Kington, http://chris35wills.github.io/matplotlib_diverging_colorbar/
        """
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            # I'm ignoring masked values and all kinds of edge cases to make a
            # simple example...
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    
    
        
    # Set min/max values from range for image (excluding NAN)
    # set midpoint according to how NDVI is interpreted: https://earthobservatory.nasa.gov/Features/MeasuringVegetation/
    min= 5 #np.nanmin(dissolved_oxygen)
    max= 7 #np.nanmax(dissolved_oxygen)
    mid=6
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    
    # diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
    cmap = 'Set1'# plt.cm.RdYlGn 
    
    cax = ax.imshow(dissolved_oxygen, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid,vmin=min, vmax=max))
    
    ax.axis('off')
    title = "Oxigênio dissolvido (mg/L) em profundidade: " + str(depth) + " m"
    #ax.set_title(title, fontsize=18, fontweight='bold')
    
    #cbar = fig.colorbar(cax, orientation='vertical', shrink=0.65)
    
    #fig.savefig("do.png", dpi=400, bbox_inches='tight', pad_inches=0.3)
    fig.savefig("do.png", dpi=400, bbox_inches='tight', pad_inches=0, transparent=True)
    
    # Set min/max values from range for image (excluding NAN)
    # set midpoint according to how NDVI is interpreted: https://earthobservatory.nasa.gov/Features/MeasuringVegetation/
    
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    
    # diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
    cmap = plt.cm.RdYlGn 
    
    cax = ax.imshow(dissolved_oxygen, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid,vmin=min, vmax=max))
    
    ax.axis('off')
    title = "Oxigênio dissolvido (mg/L) em profundidade: " + str(depth) + " m"
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    cbar = fig.colorbar(cax, orientation='vertical', shrink=0.65)
    
    fig.savefig("do_legends.png", dpi=400, bbox_inches='tight', pad_inches=0.3)
    st.pyplot(fig)
    
    min= 0 #np.nanmin(bathymetry)
    max= 1 #np.nanmax(bathymetry)
    mid= (min + max)/2
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    
    # diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
    cmap = plt.cm.RdYlGn # plt.cm.RdYlGn 
    
    cax = ax.imshow(dissolved_oxygen, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid,vmin=min, vmax=max))
    
    ax.axis('off')
    title = "Profundidade máxima (m)"
    #ax.set_title(title, fontsize=18, fontweight='bold')
    
    #cbar = fig.colorbar(cax, orientation='vertical', shrink=0.65)
    
    #fig.savefig("do.png", dpi=400, bbox_inches='tight', pad_inches=0.3)
    fig.savefig("depth.png", dpi=400, bbox_inches='tight', pad_inches=0, transparent=True)
    
    # Set min/max values from range for image (excluding NAN)
    # set midpoint according to how NDVI is interpreted: https://earthobservatory.nasa.gov/Features/MeasuringVegetation/
    
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    
    # diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
    cmap = plt.cm.RdYlGn 
    
    cax = ax.imshow(dissolved_oxygen, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid,vmin=min, vmax=max))
    
    ax.axis('off')
    title = "Profundidade máxima (m)"
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    cbar = fig.colorbar(cax, orientation='vertical', shrink=0.65)
    
    fig.savefig("depth_legends.png", dpi=400, bbox_inches='tight', pad_inches=0.3)
    
    st.pyplot(fig)
    ## LC08 RGB Image
    in_path = 'dissolved_oxygen.tif'

    dst_crs = 'EPSG:4326'

    with rasterio.open(in_path) as src:

        img = src.read()

        src_crs = src.crs['init'].upper()
        min_lon, min_lat, max_lon, max_lat = src.bounds

    ## Conversion from UTM to WGS84 CRS
    bounds_orig = [[min_lat, min_lon], [max_lat, max_lon]]

    bounds_fin = []

    for item in bounds_orig:   
        #converting to lat/lon
        lat = item[0]
        lon = item[1]

        proj = Transformer.from_crs(int(src_crs.split(":")[1]), int(dst_crs.split(":")[1]), always_xy=True)

        lon_n, lat_n = proj.transform(lon, lat)

        bounds_fin.append([lat_n, lon_n])

    # Finding the centre latitude & longitude    
    centre_lon = bounds_fin[0][1] + (bounds_fin[1][1] - bounds_fin[0][1])/2
    centre_lat = bounds_fin[0][0] + (bounds_fin[1][0] - bounds_fin[0][0])/2
    m = folium.Map(location=[B.lat, B.lon],
                       tiles='Stamen Terrain', zoom_start = 17)

    tooltip = "Informações"
    for k in range(0,len(points)):
        src = rasterio.open("dissolved_oxygen.tif")
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xx, yy = transformer.transform(points[k].lon, points[k].lat)

        # get value from grid
        value = list(src.sample([(xx, yy)]))[0]
        info = points[k].name + " Oxigênio dissolvido: " + str(value) + " mg/L" 



        folium.Marker(
            [points[k].lat, points[k].lon], popup= info, tooltip=tooltip
        ).add_to(m)
        






    # Overlay raster (RGB) called img using add_child() function (opacity and bounding box set)
    m.add_child(folium.raster_layers.ImageOverlay("do.png", opacity=.8, 
                                     bounds = bounds_fin, transparent = True,name = "Dissolved Oxygen"))
    m.add_child(folium.raster_layers.ImageOverlay("depth.png", opacity=.8, 
                                     bounds = bounds_fin, transparent = True, name = "Bathymetry"))

    folium.TileLayer('Stamen Terrain', transparent = True).add_to(m)
    folium.TileLayer('openstreetmap', transparant = True).add_to(m)
    folium.TileLayer('Stamen Toner', transparent = True).add_to(m)
    folium.TileLayer('Stamen Water Color', transparent = True).add_to(m)
    folium.TileLayer('cartodbpositron', transparent = True).add_to(m)
    folium.TileLayer('cartodbdark_matter', transparent = True).add_to(m)
    folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True
       ).add_to(m)
    folium.LayerControl().add_to(m)
    # Display map 
    folium_static(m, width=1200, height=450)
    
    
    

depth = st.number_input('Escolha a profundidade', min_value = float(0.1), max_value = float(5),step = float(0.1))
image = option = st.selectbox(
    'Escolha a imagem',
    [onlyfiles[1],onlyfiles[len(onlyfiles)-1]])
image_path = os.path.join(mypath, image)
generate_tif(depth,image_path,image)


