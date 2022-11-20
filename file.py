import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
st.write("""
# Tunisia House Price Prediction App
The main task of this project is to $predict$ the price of a housing in Tunsia. We will be using this dataset of House pricing in Tunisia, which originally contains more than 8000 rows and 25 features. It was preprocced (check this notebook) to fit a linear regression model that predicts a housing price using user input (sidebar).
""")
st.write('---')

# Loads the Tunisian House Price Dataset
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
    concierge = 0
    beach_view = 0
    mountain_view = 0
    pool = 0
    air_conditioning = 0
    central_heating = 0
    Area = st.sidebar.number_input('House Area(in m²)', min_value=20, max_value=2200, value=300, step=1)
    room = st.sidebar.select_slider('Number of rooms',  options=[i for i in range(1,43)],value=3)
    bathroom = st.sidebar.select_slider('Number of bathrooms', options=[i for i in range(1,15)], value=1)
    state = st.sidebar.select_slider('State of House(0 for now, 1 for normal, 2 for needs renovation', options=[i for i in range(0,3)], value=1)
    latt = st.sidebar.slider('latt', float(X.latt.min()),float(X.latt.max()),float(X.latt.mean()))
    longi = st.sidebar.slider('long', float(X.long.min()), float(X.long.max()), float(X.long.mean()))
    distance_to_capital = st.sidebar.slider('distance_to_capital', float(X.distance_to_capital.min()),float(X.distance_to_capital.max()),float(X.distance_to_capital.mean()))
    st.sidebar.write("## The following checkboxes represent categorial features")
    conciergecheck = st.sidebar.checkbox('Concierge', value=False)
    beach_viewcheck = st.sidebar.checkbox('Beach view', value=False)
    mountain_viewcheck = st.sidebar.checkbox('Mountain view', value=False)
    poolcheck = st.sidebar.checkbox('Pool', value=False)
    air_conditioningcheck = st.sidebar.checkbox('Air conditioning', value=False)
    central_heatingcheck = st.sidebar.checkbox('Central heating', value=False)
    if conciergecheck:
        concierge = 1
    if beach_viewcheck:
        beach_view = 1 
    if poolcheck:
        pool = 1
    if mountain_viewcheck:
        mountain_view = 1
    if air_conditioningcheck:
        air_conditioning = 1
    if central_heatingcheck:
        central_heating = 1
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
st.dataframe(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y.values.ravel())
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Price prediction')
st.write('#### ', round(prediction[0] , 3) ,'TND')
st.write('---')
st.header('Inferences')
st.write('### Linear regression')
st.write('Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting.')
st.latex(r'''
\begin{equation}
Y_i = \beta_0 + \beta_1 X_i + \epsilon_i
\end{equation} ''')
st.write('In our case we are forcasting $Y_i$ which is the feature `price_tnd`. While the rest of features `Area` `room` `bathroom` `latt` `long` `distance_to_capital` `concierge` `beach_view` `mountain_view` `pool` `air_conditioning` `central_heating`, represents $X_i$, the prediction variables.')
st.write('### Feature Importance')
st.write('#### Correlation')
st.write("Correlation is statistical technique which determines how one variables moves/changes in relation with the other variable. It gives us the idea about the degree of the relationship of the two variables. in this case it has been used to forecast our target variable using seaborn's heatmap.")
st.image("cor.png")
st.write('#### Shapley Additive Explanations')
st.write('It is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.')
st.latex(r'''
\operatorname{SHAP}_{\text {feature }}(x)=\sum_{\text {set:feature } \in \text { set }}\left[|\operatorname{set}| \times\left(\begin{array}{c}F \\ \mid \text { set } \mid\end{array}\right)\right]^{-1}\left[\operatorname{Predict}_{\text {set }}(x)-\operatorname{Predict}_{\text {set } \mid \text { feature }}(x)\right]
 ''')
st.image("shap.png")
