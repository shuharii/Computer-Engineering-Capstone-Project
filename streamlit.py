import numpy as np
import streamlit as st
import pandas as pd
#from tensorflow.keras.callbacks import EarlyStopping
#from sklearn.model_selection import train_test_split
from tensorflow import keras
#from tensorflow.keras import layers


st.title("W E L C O M E !")

st.title("Please select the concrete features you want to produce.!")

cement = st.sidebar.slider('Cement:', 0, 1300, 0)
st.write(cement, ' kg/m^3 cement selected.')

flyash = st.sidebar.slider('Fly Ash', 0, 1300, 0)
st.write(flyash, ' kg/m^3 fly ash selected.')

water = st.sidebar.slider('Water', 0, 1300, 0)
st.write(water, ' kg/m^3 water selected.')

superplasticizer = st.sidebar.slider('Superplasticizer', 0, 1300, 0)
st.write(superplasticizer, ' kg/m^3 superplasticizer selected.')

coarse_aggregate = st.sidebar.slider('Coarse Aggregate', 0, 1300, 0)
st.write(coarse_aggregate, ' kg/m^3 coarse_aggregate selected.')

fine_aggregate = st.sidebar.slider('Fine Aggregate', 0, 1300, 0)
st.write(fine_aggregate, ' kg/m^3 fine_aggregate selected.')

age = st.sidebar.slider('Age',1, 1300, 1)
st.write(fine_aggregate, ' days selected.')

X_manuel_test = pd.DataFrame(cement,flyash,water,superplasticizer,coarse_aggregate,fine_aggregate,age)

manuel_model = keras.models.load_model('my_model.h5')

if st.sidebar.button('Show House Price'):
    ypred = manuel_model.predict(X_manuel_test)
    st.title('Strength of concrete produced : ')
    st.title(np.round(ypred[0]))