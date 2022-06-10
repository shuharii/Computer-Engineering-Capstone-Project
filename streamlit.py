import numpy as np
import streamlit as st
import pandas as pd
#from tensorflow.keras.callbacks import EarlyStopping
#from sklearn.model_selection import train_test_split
from tensorflow import keras
#from tensorflow.keras import layers


st.title("W E L C O M E !")

st.title("Please select the concrete features you want to produce.!")

cement = st.sidebar.number_input('Cement:')
st.write(cement, ' kg/m^3 cement selected.')

flyash = st.sidebar.number_input('Fly Ash')
st.write(flyash, ' kg/m^3 fly ash selected.')

water = st.sidebar.number_input('Water')
st.write(water, ' kg/m^3 water selected.')

superplasticizer = st.sidebar.number_input('Superplasticizer')
st.write(superplasticizer, ' kg/m^3 superplasticizer selected.')

coarse_aggregate = st.sidebar.number_input('Coarse Aggregate')
st.write(coarse_aggregate, ' kg/m^3 coarse_aggregate selected.')

fine_aggregate = st.sidebar.number_input('Fine Aggregate')
st.write(fine_aggregate, ' kg/m^3 fine_aggregate selected.')

age = st.sidebar.number_input('Age')
st.write(fine_aggregate, ' days selected.')

a={'s':cement,'a':flyash,'k':water,'j':superplasticizer,'l':coarse_aggregate,'m':fine_aggregate,'n':age}

X_manuel_test = pd.DataFrame(data=a, index=[0])

manuel_model = keras.models.load_model('my_model.h5')

if st.sidebar.button('Show Strength of Concrete'):
    ypred = manuel_model.predict(X_manuel_test)
    st.title('Strength of concrete produced : ')
    st.title(ypred[0])