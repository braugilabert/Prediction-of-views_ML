import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import streamlit as st
from PIL import Image

with open('Prediction_views_ML/models/modelo_RFR.pkl', 'rb') as a:
    best_model = pickle.load(a)

st.set_page_config(page_title="Views prediction", page_icon=":moneybag:",layout="wide",
     initial_sidebar_state="expanded")

df = pd.read_csv("Prediction_views_ML/data/streamlit.csv")


""" def main():
    col1, col2 = st.columns((1,2))

    with col1:
        st.write(' ')

    with col2:
        st.image("Prediction_views_ML/Background DS.jpg")

    st.sidebar.header('Parámetros personalizados')
    
    # Página prediccion
    def user_input_parameters():

        #Nivel experiencia
        Año = st.sidebar.slider('Año', 2020, 2023)
        Nivel_experiencia = st.sidebar.selectbox('Nivel de experiencia', ("EN", "MI", "SE", "EX"))
        if Nivel_experiencia == "EN":
            Nivel_experiencia = 1
        elif Nivel_experiencia == "MI":
            Nivel_experiencia = 2
        elif Nivel_experiencia == "SE":
            Nivel_experiencia = 3
        elif Nivel_experiencia == "EX":
            Nivel_experiencia = 4
        #Tipo contrato
        Tipo_contrato = st.sidebar.selectbox('Tipo de contrato', ('FT', 'CT', 'FL', 'PT'))
        if Tipo_contrato == "FT":
            Tipo_contrato = 2
        elif Tipo_contrato == "CT":
            Tipo_contrato = 0
        elif Tipo_contrato == "FL":
            Tipo_contrato = 1
        elif Tipo_contrato == "PT":
            Tipo_contrato = 3
        #Puesto trabajo
        Puesto_trabajo = st.sidebar.selectbox('Puesto de trabajo', ('scientist','Data Science','cloud',
    'Data Analyst','Data Analytics','Data Engineer','Data Strategist','Machine Learning','ML','AI','MLOps','Head'))
        if Puesto_trabajo == "scientist":
            Puesto_trabajo = 1
        elif Puesto_trabajo == "Data Science":
            Puesto_trabajo = 1
        elif Puesto_trabajo == "cloud":
            Puesto_trabajo = 2
        elif Puesto_trabajo == "Data Analyst":
            Puesto_trabajo = 3
        elif Puesto_trabajo == "Data Analytics":
            Puesto_trabajo = 3
        elif Puesto_trabajo == "Data Engineer":
            Puesto_trabajo = 3
        elif Puesto_trabajo == "Data Strategist":
            Puesto_trabajo = 3
        elif Puesto_trabajo == "Machine Learning":
            Puesto_trabajo = 4
        elif Puesto_trabajo == "ML":
            Puesto_trabajo = 4
        elif Puesto_trabajo == "AI":
            Puesto_trabajo = 4
        elif Puesto_trabajo == "MLOps":
            Puesto_trabajo = 4
        elif Puesto_trabajo == "Head":
            Puesto_trabajo = 5
        else:
            Puesto_trabajo = 6
        #Ratio remoto
        Ratio_remoto = st.sidebar.selectbox('Ratio Remoto', ('100','50','0',))
        if Ratio_remoto == "100":
            Ratio_remoto = 100
        elif Ratio_remoto == "50":
            Ratio_remoto = 50
        elif Ratio_remoto == "0":
            Ratio_remoto = 0
        #pais empleado
        Pais_residencia_trabajador = st.sidebar.selectbox('Pais trabajador', sorted(paises_trabajador.keys()))
        valor_asignado_trab = paises_trabajador[Pais_residencia_trabajador]
        #Pais empresa
        Pais_empresa = st.sidebar.selectbox('Pais empresa', sorted(paises_trabajador.keys()))
        valor_empresa = paises_empresa[Pais_empresa]
        #Tamaño empresa
        Tamaño_empresa = st.sidebar.selectbox('Tamaño empresa', ('S','M','L',))
        if Tamaño_empresa == "S":
            Tamaño_empresa = 0
        elif Tamaño_empresa == "M":
            Tamaño_empresa = 1
        elif Tamaño_empresa == "L":
             Tamaño_empresa = 2

        data = {'Nivel_experiencia' : Nivel_experiencia,
                'Tipo_contrato':Tipo_contrato,
                'Puesto_trabajo':Puesto_trabajo,
                'Ratio_remoto': Ratio_remoto,
                'Pais_residencia_trabajador': valor_asignado_trab,
                "Pais_empresa": valor_empresa,
                "Año" : Año,
                "Tamaño_empresa": Tamaño_empresa
               }
        
        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_parameters()
    prediccion = model_pretrained.predict(df)
    st.success("El salario estimado sería de: " + str(round(prediccion[0], 2)))

if __name__ == '__main__':
    main() """