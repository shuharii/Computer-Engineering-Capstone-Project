import numpy as np
import streamlit as st
import pandas as pd
from tensorflow import keras


st.title("W E L C O M E !")

st.title("Please select the concrete features you want to produce.!")

cement = st.sidebar.number_input('Cement:')
st.write(cement, ' kg/m^3 cement selected.')
cement = (cement - 135) / (540 - 135)

flyash = st.sidebar.number_input('Fly Ash')
st.write(flyash, ' kg/m^3 fly ash selected.')
flyash = flyash/200

water = st.sidebar.number_input('Water')
st.write(water, ' kg/m^3 water selected.')
water = (water - 140)/(228-140)

superplasticizer = st.sidebar.number_input('Superplasticizer')
st.write(superplasticizer, ' kg/m^3 superplasticizer selected.')
superplasticizer = (superplasticizer)/28

coarse_aggregate = st.sidebar.number_input('Coarse Aggregate')
st.write(coarse_aggregate, ' kg/m^3 coarse_aggregate selected.')
coarse_aggregate = (coarse_aggregate - 801)/(1125-801)

fine_aggregate = st.sidebar.number_input('Fine Aggregate')
st.write(fine_aggregate, ' kg/m^3 fine_aggregate selected.')
fine_aggregate = (fine_aggregate - 594)/(945-594)

age = st.sidebar.number_input('Age')
st.write(age, ' days selected.')
age = (age - 3)/(365-3)

a={'Cement':cement,'Fly Ash':flyash,'Water':water,'Superplasticizer':superplasticizer,'Coarse Aggregate':coarse_aggregate,'Fine_Aggregate':fine_aggregate,'Age':age}

X_manuel_test = pd.DataFrame(data=a, index=[0])

manuel_model = keras.models.load_model('my_model.h5')

if st.sidebar.button('Show Strength of Concrete'):
    ypred = manuel_model.predict(X_manuel_test)
    result = abs((float(ypred[0]) * (79.99 - 6.47))) + 6.47
    st.title('Strength of concrete produced : ')
    st.write('Compressive strength is: ',round(result,1), ' MPa')