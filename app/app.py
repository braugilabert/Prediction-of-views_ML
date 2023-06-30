import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

with open('../models/modelo_RFR.pkl', 'rb') as a:
    best_model = pickle.load(a)

st.set_page_config(page_title="Views prediction :projector:", page_icon=":fire_extinguisher:",layout="wide",
     initial_sidebar_state="expanded")

df = pd.read_csv('../data/test.csv')
df = df.loc[df['Visualizaciones']<35000,:] # Quito los 2 outliers

process = pd.read_csv('../data/processed.csv')

def main():
    
    st.image("../Background DS.jpg")
    st.write("<div style='display:flex; flex-direction:column;'><h1 style='text-align: center;'>Processed dataset</h1>", unsafe_allow_html=True)
    st.dataframe(process.head())
    st.write("<div style='display:flex; flex-direction:column;'><h1 style='text-align: center;'>Gráfico predicts</h1>", unsafe_allow_html=True)
    st.image("../docs/predict menos 3000.png")
    st.write("<div style='display:flex; flex-direction:column;'><h1 style='text-align: center;'>Random Forest Regressor prediction</h1>", unsafe_allow_html=True)
    st.dataframe(df)
    st.success("Las visualizaciones del vídeo serían de: " + str(round(prediccion[0], 2)))
    
st.sidebar.header('Parámetros personalizados')
    
def user_input_parameters():
        clicks = st.sidebar.slider("Porcentaje de clics de las impresiones (%)",0,12)
        jcomp = st.sidebar.slider("Med Juego Compartido",0,10)
        horj = st.sidebar.slider("Med Horas vistas Juego",0,123)
        impj = st.sidebar.slider("Med Impresiones Juego",117,9400)
        clickip = st.sidebar.slider("Med porcentaje clicks Impresiones Juego",0,7)
        visuj = st.sidebar.slider("Med Visualizaciones Juego",36,3685)
        suscj = st.sidebar.slider("Med Suscriptores Juego",0,11)
        visa = st.sidebar.slider("Med Visualizaciones Año",28,1299)
        visas = st.sidebar.slider("Visualizaciones necesarias para suscribirse al canal",0,5477)
        visac = st.sidebar.slider("Visualizaciones necesarias para compartir el video",0,8049)
    
    
        data ={
        "Porcentaje de clics de las impresiones (%)":clicks,
        "Med Juego Compartido":jcomp,
        "Med Horas vistas Juego":horj,
        "Med Impresiones Juego":impj,
        "Med porcentaje clicks Impresiones Juego":clickip,
        "Med Visualizaciones Juego":visuj,
        "Med Suscriptores Juego":suscj,
        "Med Visualizaciones Año":visa,
        "Visualizaciones necesarias para suscribirse al canal":visas,
        "Visualizaciones necesarias para compartir el video":visac,
}

        features = pd.DataFrame(data, index=[0])
        return features
    
df = user_input_parameters()
prediccion = best_model.predict(df)

if __name__ == '__main__':
    main()
##python -m streamlit run main.py 