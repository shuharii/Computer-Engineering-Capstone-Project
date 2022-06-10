import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import time
import streamlit as st
import requests
#from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)
np.random.seed(42)


st.title("W E L C O M E !")

st.title("Please select the concrete features you want to produce.!")


concrete_data = 'https://raw.githubusercontent.com/shuharii/ann-streamlit/main/concrete_data.csv'
#result_data = 'https://raw.githubusercontent.com/shuharii/ann-streamlit/main/capstone_real_results.csv'
df = pd.read_csv(concrete_data)
df = df[df['Blast Furnace Slag']==0.0]
df.drop(['Blast Furnace Slag'], axis=1, inplace=True)
#Spliting into Independent and dependent variable
X = df.iloc[:,:7] #Independent
y = df.iloc[:,7] #Dependent
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)

#df_test = pd.read_csv('sample_data/capstone_real_results.csv') #veri dataframe olarak yuklendi.
#X_test = df_test.iloc[:,:7] #Independent
#y_test = df_test.iloc[:,7]  #Dependent

#scaler = StandardScaler()
#scaler.fit(X_train)

#X_train_scaled = scaler.transform(X_train)
#X_valid_scaled = scaler.transform(X_valid)
#X_test_scaled = scaler.transform(X_test)

# Scale to [0, 1]
max_ = X_train.max(axis=0)
min_ = X_train.min(axis=0)
X_train_scaled = (X_train - min_) / (max_ - min_)
X_valid_scaled = (X_valid - min_) / (max_ - min_)
#X_test_scaled = (X_test - min_) / (max_ - min_)

'''
stop_early = EarlyStopping(
    min_delta=0.001,              # minimium amount of change to count as an improvement
    patience=30,                  # how many epochs to wait before stopping
    restore_best_weights=True,
)

manuel_model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=512, activation='relu', input_shape=[7]),
    layers.Dropout(0.3),
    #layers.BatchNormalization(),
    layers.Dense(units=512, activation='relu'),
    layers.Dropout(0.3),
    #layers.BatchNormalization(),
    layers.Dense(units=512, activation='relu'),
    layers.Dropout(0.3),
    #layers.BatchNormalization(),
    # the linear output layer
    layers.Dense(units=1),
])

manuel_model.compile(
    #optimizer=keras.optimizers.Adam(learning_rate=0.01),
    optimizer='adam',
    loss="mse",
)

#history = manuel_model.fit(
#    X_train_scaled, y_train,
#    validation_data=(X_valid_scaled, y_valid),
#    batch_size=64,
#    epochs=250,
#    callbacks=[stop_early],
#    verbose=0,  # turn off training log
#)
'''

cement = st.sidebar.slider('Cement:', 0, 1300, 25)
st.write(cement, ' kg/m^3 cement selected.')

flyash = st.sidebar.slider('Fly Ash', 0, 1300, 25)
st.write(flyash, ' kg/m^3 fly ash selected.')

water = st.sidebar.slider('Water', 0, 1300, 25)
st.write(water, ' kg/m^3 water selected.')

superplasticizer = st.sidebar.slider('Superplasticizer', 0, 1300, 25)
st.write(superplasticizer, ' kg/m^3 superplasticizer selected.')

coarse_aggregate = st.sidebar.slider('Coarse Aggregate', 0, 1300, 25)
st.write(coarse_aggregate, ' kg/m^3 coarse_aggregate selected.')

fine_aggregate = st.sidebar.slider('Fine Aggregate', 0, 1300, 25)
st.write(fine_aggregate, ' kg/m^3 fine_aggregate selected.')

age = st.sidebar.slider('Age',1, 1300, 25)
st.write(fine_aggregate, ' days selected.')

X_manuel_test = pd.DataFrame(cement,flyash,water,superplasticizer,coarse_aggregate,fine_aggregate,age)

manuel_model = keras.models.load_model('saved_model')


if st.sidebar.button('Show House Price'):
    ypred = manuel_model.predict(X_manuel_test)
    st.title('Strength of concrete produced : ')
    st.title(np.round(ypred[0]))