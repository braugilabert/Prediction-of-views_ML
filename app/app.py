import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import streamlit as st
from PIL import Image

with open('../models/modelo_RFR.pkl', 'rb') as a:
    best_model = pickle.load(a)

st.set_page_config(page_title="Views prediction :projector:", page_icon=":fire_extinguisher:",layout="wide",
     initial_sidebar_state="expanded")

df = pd.read_csv('../data/test.csv')
df = df.loc[df['Visualizaciones']<35000,:] # Quito los 2 outliers

def main():
    col1, col2 = st.columns((1,2))

    with col1:
        st.write('Random Forests Regressor prediction.')

    with col2:
        st.image("../Background DS.jpg")


st.sidebar.header('Parámetros personalizados')
    
def user_input_parameters():
        max_players = st.sidebar.slider("Porcentaje de clics de las impresiones (%)",0,12)
        bgg_rank = st.sidebar.slider("Med Juego Compartido",0,10)
        complejidad_juego = st.sidebar.slider("Med Horas vistas Juego",0,123)
        owned_users = st.sidebar.slider("Med Impresiones Juego",117,9400)
        mech_action = st.sidebar.slider("Med porcentaje clicks Impresiones Juego",0,7)
        min_age = st.sidebar.slider("Med Visualizaciones Juego",36,3685)
        mech_acting = st.sidebar.slider("Med Suscriptores Juego",0,11)
        mech_not_defined = st.sidebar.slider("Med Visualizaciones Año",28,1299)
        min_players = st.sidebar.slider("Visualizaciones necesarias para suscribirse al canal",0,5477)
        play_time = st.sidebar.slider("Visualizaciones necesarias para compartir el video",0,8049)
    
    
        data ={
        "Porcentaje de clics de las impresiones (%)":max_players,
        "Med Juego Compartido":bgg_rank,
        "Med Horas vistas Juego":complejidad_juego,
        "Med Impresiones Juego":owned_users,
        "Med porcentaje clicks Impresiones Juego":mech_action,
        "Med Visualizaciones Juego":min_age,
        "Med Suscriptores Juego":mech_acting,
        "Med Visualizaciones Año":mech_not_defined,
        "Visualizaciones necesarias para suscribirse al canal":min_players,
        "Visualizaciones necesarias para compartir el video":play_time,
}

        features = pd.DataFrame(data, index=[0])
        return features
    
df = user_input_parameters()
prediccion = best_model.predict(df)
st.success("Las visualizaciones serían de: " + str(round(prediccion[0], 2)))


if __name__ == '__main__':
    main()
##python -m streamlit run main.py 