import numpy as np
import streamlit as st
import pandas as pd
#from tensorflow.keras.callbacks import EarlyStopping
#from sklearn.model_selection import train_test_split
from tensorflow import keras
#from tensorflow.keras import layers


st.title("W E L C O M E !")

st.title("Please select the concrete features you want to produce.!")

url ='https://raw.githubusercontent.com/shuharii/ann-streamlit/main/concrete_data.csv'

df = pd.read_csv(url)
df = df[df['Blast Furnace Slag']==0.0]
df.drop(['Blast Furnace Slag'], axis=1, inplace=True)
X = df.iloc[:,:7] #Independent
y = df.iloc[:,7] #Dependent
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)

max_ = X.max(axis=0)
min_ = X.min(axis=0)


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

a={'Cement':cement,'Fly Ash':flyash,'Water':water,'Superplasticizer':superplasticizer,'Coarse Aggregate':coarse_aggregate,'Fine_Aggregate':fine_aggregate,'Age':age}

X_manuel_test = pd.DataFrame(data=a, index=[0])

X_train_scaled = (X_manuel_test - min_) / (max_ - min_)

manuel_model = keras.models.load_model('my_model.h5')

if st.sidebar.button('Show Strength of Concrete'):
    ypred = manuel_model.predict(X_train_scaled)
    st.title('Strength of concrete produced : ')
    st.title(ypred[0])