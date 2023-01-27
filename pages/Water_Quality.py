# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 04:47:14 2023

@author: caion
"""

import os
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
from ipyleaflet import Map, basemaps

from os import listdir
from os.path import isfile, join

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

df = pandas.read_csv('PuroPeixe - PuroPeixe.csv')
df = df.applymap(lambda x: str(str(x).replace(',','.')))

column_headers = list(df.columns.values)
for k in range(0,len(column_headers)):
    if column_headers[k] == "Hora" or column_headers[k] == "Dia":
        continue;
    else:
        df[column_headers[k]] = pandas.to_numeric(df[column_headers[k]],errors = 'coerce')

date_aux = []
b_aux = []
s_aux = []
p_aux = []
z_aux = []
T_aux = []

for i in range(0,len(df)):
    
    hour = df.iloc[i]['Hora']
    dia = df.iloc[i]['Dia']
    datestr = dia + " " + hour
    dateTime = datetime.datetime.strptime(datestr, '%d/%m/%Y %H:%M')
    date_aux.append(dateTime)
    
    # Batelão
    cont = 3
    BA1 =df.iloc[i]['BA1']
    if np.isnan(BA1):
        cont = cont-1
    BA2 = df.iloc[i]['BA2']
    if np.isnan(BA2):
        cont = cont-1
    BA3 = df.iloc[i]['BA3']
    if np.isnan(BA3):
        cont = cont-1
    B_dep = df.iloc[i]['Profundidade de Secchi (m)']
    Bs = np.nansum(np.array([BA1,BA2,BA3]))/cont
    b_aux.append(Bs)
    # 6x6 
    cont = 3
    S1 = df.iloc[i]['6x61']
    if np.isnan(S1):
        cont = cont-1
    
    S2 = df.iloc[i]['6x62']
    if np.isnan(S2):
        cont = cont-1
    S3 = df.iloc[i]['6x63']
    if np.isnan(S3):
        cont = cont-1
    S_dep = df.iloc[i]['METROS.1']
    Ss = np.nansum(np.array([S1,S2,S3]))/cont
    s_aux.append(Ss)
    
    # Porto
    cont = 3
    P1 = df.iloc[i]['PORTO1']
    if np.isnan(P1):
        cont = cont-1
    P2 = df.iloc[i]['PORTO2']
    if np.isnan(P2):
        cont = cont-1
    P3 = df.iloc[i]['PORTO3']
    if np.isnan(P3):
        cont = cont-1
    P_dep = df.iloc[i]['METROS']
    Ps = np.nansum(np.array([P1, P2, P3]))/cont
    p_aux.append(Ps)
    
    # Zippy
    cont = 3
    Z1 = df.iloc[i]['ZIPPY1']
    if np.isnan(Z1):
        cont = cont-1
    Z2 = df.iloc[i]['ZIPPY2']
    if np.isnan(Z2):
        cont = cont-1
    Z3 = df.iloc[i]['ZIPPY3']
    if np.isnan(Z3):
        cont = cont-1
    Z_dep = df.iloc[i]['METROS.2']
    Zs = np.nansum(np.array([Z1, Z2, Z3]))/cont
    z_aux.append(Zs)
    
    # Temp
    T = df.iloc[i]['C°']
    T_aux.append(T)
    
df['dt'] = date_aux
df['B_avg'] = b_aux
df['S_avg'] = s_aux
df['P_avg'] = p_aux
df['Z_avg'] = z_aux
df['T'] = T_aux

fig, axes = plt.subplots(figsize = (20,5))
axes.set_title("Batelão")
df.plot(ax = axes, kind='scatter',x='dt',y='B_avg',color='red')
plt.show()
st.pyplot(fig)

fig, axes = plt.subplots(figsize = (20,5))
axes.set_title("6x6")
df.plot(ax = axes, kind='scatter',x='dt',y='S_avg',color='red')
plt.show()
st.pyplot(fig)

fig, axes = plt.subplots(figsize = (20,5))
axes.set_title("Porto")
df.plot(ax = axes, kind='scatter',x='dt',y='P_avg',color='red')
plt.show()
st.pyplot(fig)

fig, axes = plt.subplots(figsize = (20,5))
axes.set_title("Zippy")
df.plot(ax = axes, kind='scatter',x='dt',y='Z_avg',color='red')
plt.show()
st.pyplot(fig)


# List of all points

points = [B,Six6,P,Z]

# generate regression model
absolute_path = os.getcwd()
relative_path = "PuroPeixe/files/"
mypath = os.path.join(absolute_path, relative_path) 
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

cont = 1

DO = []
DEPTH = []
SECCHI = []
B2 = []
B4 = []
B6 = []
B8 = []


from pyproj import Transformer

for i in range(0,len(np.array(onlyfiles))):
    filename = mypath + onlyfiles[i]
    src = rasterio.open(filename)
    strAux = onlyfiles[i].split('/')[0]
    year = strAux[0:4]
    mon = strAux[4:6]
    day = strAux[6:8]
    H = strAux[9:11]
    M = strAux[11:13]
    fDate = day + "/" + mon + "/" + year + " " + H +":" + M
    dateTime = datetime.datetime.strptime(fDate, '%d/%m/%Y %H:%M')
    index = (pandas.to_datetime(df['dt'])-pandas.to_datetime(dateTime)).abs().idxmin()
    idx = df.loc[index,'dt']
    
    for k in range(0,len(points)):
        #rows, cols = rasterio.transform.rowcol(src.transform, points[k].lon, points[k].lat)
        #vals = src.sample((rows, cols))
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xx, yy = transformer.transform(points[k].lon, points[k].lat)

        # get value from grid
        value = list(src.sample([(xx, yy)]))[0]
        if k == 0:
            if ~np.isnan(df.loc[index,'BA1']):
                DO.append(df.loc[index,'BA1'])
                DEPTH.append(0.3)
                SECCHI.append(df.loc[index,'Profundidade de Secchi (m)'])
                B2.append(value[1])
                B4.append(value[3])
                B6.append(value[5])
                B8.append(value[7])
            if ~np.isnan(df.loc[index,'BA2']):
                DO.append(df.loc[index,'BA2'])
                DEPTH.append(1.5)
                SECCHI.append(df.loc[index,'Profundidade de Secchi (m)'])
                B2.append(value[1])
                B4.append(value[3])
                B6.append(value[5])
                B8.append(value[7])
            if ~np.isnan(df.loc[index,'BA3']):
                DO.append(df.loc[index,'BA3'])
                DEPTH.append(3.8)
                SECCHI.append(df.loc[index,'Profundidade de Secchi (m)'])
                B2.append(value[1])
                B4.append(value[3])
                B6.append(value[5])
                B8.append(value[7])
        elif k == 1:
            if ~np.isnan(df.loc[index,'6x61']):
                DO.append(df.loc[index,'6x61'])
                DEPTH.append(0.3)
                SECCHI.append(df.loc[index,'METROS.1'])
                B2.append(value[1])
                B4.append(value[3])
                B6.append(value[5])
                B8.append(value[7])
            if ~np.isnan(df.loc[index,'6x62']):
                DO.append(df.loc[index,'6x62'])
                DEPTH.append(1.5)
                SECCHI.append(df.loc[index,'METROS.1'])
                B2.append(value[1])
                B4.append(value[3])
                B6.append(value[5])
                B8.append(value[7])
            if ~np.isnan(df.loc[index,'6x63']):
                DO.append(df.loc[index,'6x63'])
                DEPTH.append(3.8)
                SECCHI.append(df.loc[index,'METROS.1'])
                B2.append(value[1])
                B4.append(value[3])
                B6.append(value[5])
                B8.append(value[7])
        elif k == 2:
            if ~np.isnan(df.loc[index,'PORTO1']):
                DO.append(df.loc[index,'PORTO1'])
                DEPTH.append(0.3)
                SECCHI.append(df.loc[index,'METROS'])
                B2.append(value[1])
                B4.append(value[3])
                B6.append(value[5])
                B8.append(value[7])
            if ~np.isnan(df.loc[index,'PORTO2']):
                DO.append(df.loc[index,'PORTO2'])
                DEPTH.append(1.5)
                SECCHI.append(df.loc[index,'METROS'])
                B2.append(value[1])
                B4.append(value[3])
                B6.append(value[5])
                B8.append(value[7])
            if ~np.isnan(df.loc[index,'PORTO3']):
                DO.append(df.loc[index,'PORTO3'])
                DEPTH.append(3.8)
                SECCHI.append(df.loc[index,'METROS'])
                B2.append(value[1])
                B4.append(value[3])
                B6.append(value[5])
                B8.append(value[7])
        elif k == 3:
            if ~np.isnan(df.loc[index,'ZIPPY1']):
                DO.append(df.loc[index,'ZIPPY1'])
                DEPTH.append(0.3)
                SECCHI.append(df.loc[index,'METROS.2'])
                B2.append(value[1])
                B4.append(value[3])
                B6.append(value[5])
                B8.append(value[7])
            if ~np.isnan(df.loc[index,'ZIPPY2']):
                DO.append(df.loc[index,'ZIPPY2'])
                DEPTH.append(1.5)
                SECCHI.append(df.loc[index,'METROS.2'])
                B2.append(value[1])
                B4.append(value[3])
                B6.append(value[5])
                B8.append(value[7])
            if ~np.isnan(df.loc[index,'ZIPPY3']):
                DO.append(df.loc[index,'ZIPPY3'])
                DEPTH.append(3.8)
                SECCHI.append(df.loc[index,'METROS.2'])
                B2.append(value[1])
                B4.append(value[3])
                B6.append(value[5])
                B8.append(value[7])
        
        
d = {'DO':DO,'depth': DEPTH, 'b2': B2,'b4': B4,'b6':B6,'b8':B8, 'secchi': SECCHI}
df2 = pandas.DataFrame(data=d)

X = df2[['depth','b2','b4','b6','b8','secchi']]
X_secchi = df2[['b2','b4','b6','b8']]
y = df2[['DO']]
y_secchi = df2[['secchi']]

from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X, y)
coefficients = pandas.concat([pandas.DataFrame(X.columns),pandas.DataFrame(np.transpose(regr.coef_))], axis = 1) 

regr_secc = linear_model.LinearRegression()
regr_secc.fit(X, y)
coefficients_secc = pandas.concat([pandas.DataFrame(X.columns),pandas.DataFrame(np.transpose(regr.coef_))], axis = 1)

# generate regression model
absolute_path = os.getcwd()
relative_path = "PuroPeixe/files/"
mypath = os.path.join(absolute_path, relative_path) 
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

B_pred_1 = [];
B_pred_2 = [];
B_pred_3 = [];
B_pred_secchi = [];
B_date_aux = [];
S_pred_1 = [];
S_pred_2 = [];
S_pred_3 = [];
S_pred_secchi = [];
S_date_aux = [];
P_pred_1 = [];
P_pred_2 = [];
P_pred_3 = [];
P_pred_secchi = [];
P_date_aux = [];
Z_pred_1 = [];
Z_pred_2 = [];
Z_pred_3 = [];
Z_pred_secchi = [];
Z_date_aux = [];
## Compare predictions
for i in range(0,len(np.array(onlyfiles))):
    filename = mypath + onlyfiles[i]
    src = rasterio.open(filename)
    band_red = src.read(6)
    
    band_nir = src.read(8)
    
    band_green = src.read(4)
        
    band_blue = src.read(2)
    strAux = onlyfiles[i].split('/')[0]
    year = strAux[0:4]
    mon = strAux[4:6]
    day = strAux[6:8]
    H = strAux[9:11]
    M = strAux[11:13]
    fDate = day + "/" + mon + "/" + year + " " + H +":" + M
    dateTime = datetime.datetime.strptime(fDate, '%d/%m/%Y %H:%M')
    index = (pandas.to_datetime(df['dt'])-pandas.to_datetime(dateTime)).abs().idxmin()
    idx = df.loc[index,'dt']
    
    
    
    
    for k in range(0,len(points)):
        #rows, cols = rasterio.transform.rowcol(src.transform, points[k].lon, points[k].lat)
        #vals = src.sample((rows, cols))
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xx, yy = transformer.transform(points[k].lon, points[k].lat)

        # get value from grid
        value = list(src.sample([(xx, yy)]))[0]
        band_blue = value[1]
        band_green = value[3]
        band_red = value[5]
        band_nir = value[7]
        
        secc = regr_secc.intercept_[0] + coefficients_secc.iat[0,1] *band_blue.astype(float) + coefficients_secc.iat[1,1]*band_green.astype(float) + coefficients_secc.iat[2,1]*band_red.astype(float) + coefficients_secc.iat[3,1]*band_nir.astype(float)
        depth1 = 0.3
        depth2 = 1.5
        depth3 = 3.8
        dissolved_oxygen_1 = regr.intercept_[0] + coefficients.iat[0,1]*depth1 + coefficients.iat[1,1] *band_blue.astype(float) + coefficients.iat[2,1]*band_green.astype(float) + coefficients.iat[3,1]*band_red.astype(float) + coefficients.iat[4,1]*band_nir.astype(float) + coefficients.iat[5,1]*secc.astype(float)
        dissolved_oxygen_2 = regr.intercept_[0] + coefficients.iat[0,1]*depth1 + coefficients.iat[1,1] *band_blue.astype(float) + coefficients.iat[2,1]*band_green.astype(float) + coefficients.iat[3,1]*band_red.astype(float) + coefficients.iat[4,1]*band_nir.astype(float) + coefficients.iat[5,1]*secc.astype(float)
        dissolved_oxygen_3 = regr.intercept_[0] + coefficients.iat[0,1]*depth1 + coefficients.iat[1,1] *band_blue.astype(float) + coefficients.iat[2,1]*band_green.astype(float) + coefficients.iat[3,1]*band_red.astype(float) + coefficients.iat[4,1]*band_nir.astype(float) + coefficients.iat[5,1]*secc.astype(float)
        if k == 0:
            B_pred_1.append(dissolved_oxygen_1)
            B_pred_2.append(dissolved_oxygen_2)
            B_pred_3.append(dissolved_oxygen_3)
            B_pred_secchi.append(secc)
            B_date_aux.append(dateTime)
        elif k == 1:
            S_pred_1.append(dissolved_oxygen_1)
            S_pred_2.append(dissolved_oxygen_2)
            S_pred_3.append(dissolved_oxygen_3)
            S_pred_secchi.append(secc)
            S_date_aux.append(dateTime)
        elif k == 2:
            P_pred_1.append(dissolved_oxygen_1)
            P_pred_2.append(dissolved_oxygen_2)
            P_pred_3.append(dissolved_oxygen_3)
            P_pred_secchi.append(secc)
            P_date_aux.append(dateTime)
        elif k == 3:
            Z_pred_1.append(dissolved_oxygen_1)
            Z_pred_2.append(dissolved_oxygen_2)
            Z_pred_3.append(dissolved_oxygen_3)
            Z_pred_secchi.append(secc)
            Z_date_aux.append(dateTime)
#st.write(B_pred_1)

def PlotFrom2DF(df1,df2, title, field1,field2):
    
    df1["smooth"] = np.polyval(np.polyfit(df["dt"].astype(int),df[field1], 5), df["dt"].astype(int))
    df2["smooth"] = np.polyval(np.polyfit(df["Aux_date"].astype(int),df[field2], 5), df["Aux_date"].astype(int))
    
    fig, axes = plt.subplots(figsize = (20,5))
    axes.set_title(title)
    #data1.plot()
    #data2.plot()
    df1.plot(ax = axes, kind='line',x='dt',y="smooth",color='red')
    df2.plot(ax = axes, kind='line',x='Aux_date',y="smooth",color='blue')
    plt.show()
    st.pyplot(fig)


dB = {'B_pred1': B_pred_1,'B_pred2': B_pred_2, 'B_pred3': B_pred_3, 'Aux_date' : B_date_aux}
dfB2 = pandas.DataFrame(data=dB)     

dS = {'S_pred1': S_pred_1,'S_pred2': S_pred_2, 'S_pred3': S_pred_3, 'Aux_date' : S_date_aux}
dfS2 = pandas.DataFrame(data=dS)     

dP = {'P_pred1': P_pred_1,'P_pred2': P_pred_2, 'P_pred3': P_pred_3, 'Aux_date' : P_date_aux}
dfP2 = pandas.DataFrame(data=dP)     

dZ = {'Z_pred1': Z_pred_1,'Z_pred2': Z_pred_2, 'Z_pred3': Z_pred_3, 'Aux_date' : Z_date_aux}
dfZ2 = pandas.DataFrame(data=dZ)     

PlotFrom2DF(df, dfB2, "Batelão - 0.3 m",'BA1','B_pred1')
PlotFrom2DF(df, dfB2, "Batelão - 1.5 m",'BA2','B_pred2')
PlotFrom2DF(df, dfB2, "Batelão - 3.8 m",'BA3','B_pred3')

PlotFrom2DF(df, dfS2, "6x6 - 0.3 m",'6x61','S_pred1')
PlotFrom2DF(df, dfS2, "6x6 - 1.5 m",'6x62','S_pred2')
PlotFrom2DF(df, dfS2, "6x6 - 3.8 m",'6x63','S_pred3')

PlotFrom2DF(df, dfB2, "Porto - 0.3 m",'PORTO1','P_pred1')
PlotFrom2DF(df, dfB2, "Porto - 1.5 m",'PORTO2','P_pred2')
PlotFrom2DF(df, dfB2, "Porto - 3.8 m",'PORTO3','P_pred3')

PlotFrom2DF(df, dfB2, "Zippy - 0.3 m",'ZIPPY1','Z_pred1')
PlotFrom2DF(df, dfB2, "Zippy - 1.5 m",'ZIPPY2','Z_pred2')
PlotFrom2DF(df, dfB2, "Zippy - 3.8 m",'ZIPPY3','Z_pred3')




## 
import statsmodels.api as sm
X1 = sm.add_constant(X)
result = sm.OLS(y, X1).fit()
#st.write('R2: ', result.rsquared)
coefs = coefficients.iat[1,1]
# Save model using dump function of pickle
pck_file = "regression_model.pkl"
absolute_path = os.getcwd()
relative_path = "PuroPeixe/files/"
mypath = os.path.join(absolute_path, relative_path) 
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
   
#@st.cache(allow_output_mutation=True)
def generate_tif(depth = 4, filename = ""):
    
    if filename == "":
        filename = mypath + onlyfiles[1]
    # List of filenames for tifs
    
    
    src = rasterio.open(filename)
    
    # Load red and NIR bands - note all PlanetScope 4-band images have band order BGRN
    with rasterio.open(filename) as src:
        band_red = src.read(6)
    
    with rasterio.open(filename) as src:
        band_nir = src.read(8)
    
    with rasterio.open(filename) as src:
        band_green = src.read(4)
        
    with rasterio.open(filename) as src:
        band_blue = src.read(2)
    
    # Allow division by zero
    np.seterr(divide='ignore', invalid='ignore')
    
    # Calculate Dissolved Oxygen from regression equation
    secc = regr_secc.intercept_[0] + coefficients_secc.iat[0,1] *band_blue.astype(float) + coefficients_secc.iat[1,1]*band_green.astype(float) + coefficients_secc.iat[2,1]*band_red.astype(float) + coefficients_secc.iat[3,1]*band_nir.astype(float)
    dissolved_oxygen = regr.intercept_[0] + coefficients.iat[0,1]*depth + coefficients.iat[1,1] *band_blue.astype(float) + coefficients.iat[2,1]*band_green.astype(float) + coefficients.iat[3,1]*band_red.astype(float) + coefficients.iat[4,1]*band_nir.astype(float) + coefficients.iat[5,1]*secc.astype(float)
    dissolved_oxygen[dissolved_oxygen < 0] = 0
    dissolved_oxygen[dissolved_oxygen > 20] = 20
    
    # bathymetry
    a0 = -3.24
    a1 = 14.72
    a2 = -18.48
    bathymetry = a0 + a1*np.log(band_blue/10000) + a2*np.log(band_green/10000)
    
    # NDWI for masking
    ndwi = (band_green.astype(float) - band_nir.astype(float))/(band_green.astype(float) + band_nir.astype(float))
    dissolved_oxygen[ndwi < -0.4] = np.nan
    bathymetry[ndwi < -0.4] = np.nan
    bathymetry[bathymetry < 0] = 0
    bathymetry[bathymetry > 20] = np.nan
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
    cmap = 'Set1'# plt.cm.RdYlGn 
    
    cax = ax.imshow(dissolved_oxygen, cmap=cmap, clim=(min, max), norm=MidpointNormalize(midpoint=mid,vmin=min, vmax=max))
    
    ax.axis('off')
    title = "Oxigênio dissolvido (mg/L) em profundidade: " + str(depth) + " m"
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    cbar = fig.colorbar(cax, orientation='vertical', shrink=0.65)
    
    fig.savefig("do_legends.png", dpi=400, bbox_inches='tight', pad_inches=0.3)
    st.pyplot(fig)
    
    min= np.nanmin(bathymetry)
    max= np.nanmax(bathymetry)
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
        index = len(df) - 1
        if k == 0:
            do_last = df.loc[index,'B_avg']
            depth_last = df.loc[index,'Profundidade de Secchi (m)']
        elif k == 1:
            do_last = df.loc[index,'S_avg']
            depth_last = df.loc[index,'METROS.1']
        elif k == 2:
            do_last = df.loc[index,'P_avg']
            depth_last =df.loc[index,'METROS']
        elif k == 3:
            do_last = df.loc[index,'Z_avg']
            depth_last = df.loc[index,'METROS.2']
        do_last = value
        info = points[k].name + " Oxigênio dissolvido: " + str(do_last) + " mg/L" 



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
generate_tif(depth,image_path)


