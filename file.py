import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')

# Loads the Boston House Price Dataset
tunisia = pd.read_csv("./out.csv")
X = pd.DataFrame(tunisia, columns=['Area', 'room', 'bathroom', 'state', 'latt', 'long',
       'distance_to_capital', 'concierge', 'beach_view',
       'mountain_view', 'pool', 'air_conditioning',
       'central_heating'])
Y = pd.DataFrame(tunisia, columns=["price_tnd"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    Area = st.sidebar.slider('Area', float(X.Area.min()), float(X.Area.max()), float(X.Area.mean()))
    room = st.sidebar.slider('room', float(X.room.min()), float(X.room.max()), float(X.room.mean()))
    bathroom = st.sidebar.slider('bathroom', float(X.bathroom.min()), float(X.bathroom.max()), float(X.bathroom.mean()))
    state = st.sidebar.slider('state', float(X.state.min()), float(X.state.max()), float(X.state.mean()))
    latt = st.sidebar.slider('latt', float(X.latt.min()),float(X.latt.max()),float(X.latt.mean()))
    longi = st.sidebar.slider('long', float(X.long.min()), float(X.long.max()), float(X.long.mean()))
    distance_to_capital = st.sidebar.slider('distance_to_capital', float(X.distance_to_capital.min()),float(X.distance_to_capital.max()),float(X.distance_to_capital.mean()))
    concierge = st.sidebar.slider('concierge', float(X.concierge.min()),float(X.concierge.max()),float(X.concierge.mean()))
    beach_view = st.sidebar.slider('beach_view', float(X.beach_view.min()),float(X.beach_view.max()),float(X.beach_view.mean()))
    mountain_view = st.sidebar.slider('mountain_view', float(X.mountain_view.min()),float(X.mountain_view.max()),float(X.mountain_view.mean()))
    pool = st.sidebar.slider('pool', float(X.pool.min()), float(X.pool.max()), float(X.pool.mean()))
    air_conditioning = st.sidebar.slider('air_conditioning', float(X.air_conditioning.min()), float(X.air_conditioning.max()), float(X.air_conditioning.mean()))
    central_heating = st.sidebar.slider('central_heating', float(X.central_heating.min()), float(X.central_heating.max()), float(X.central_heating.mean()))
    data = {'Area': Area,
            'room': room,
            'bathroom': bathroom,
            'state': state,
            'latt': latt,
            'long': longi,
            'distance_to_capital': distance_to_capital,
            'concierge': concierge,
            'beach_view': beach_view,
            'mountain_view': mountain_view,
            'pool': pool,
            'air_conditioning': air_conditioning,
            'central_heating': central_heating}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y.values.ravel())
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of price')
st.write(prediction)
st.write('---')
