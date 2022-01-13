import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="Valor de venta en USD por pie cuadrado App")

@st.cache(allow_output_mutation=True)
def get_model():
  return load_model('modelo_precio_inmobiliario')

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['Label'][0]

model = get_model()

st.title("Valor de venta en USD por pie cuadrado App")

 
form = st.form("MEDV")
AGE = form.number_input('Age', min_value=1, max_value=100, value=25)
TAX = form.number_input('Tax', min_value=1, max_value=300, value=25)
RM = form.number_input('RM', min_value=1, max_value=300, value=25)
ZN = form.number_input('ZN', min_value=1, max_value=300, value=25)
INDUS = form.number_input('INDUS', min_value=1, max_value=300, value=25)
DIS = form.number_input('DIS', min_value=1, max_value=300, value=25)
BLACK = form.number_input('BLACK', min_value=1, max_value=300, value=25)
RAD = form.number_input('RAD', min_value=1, max_value=300, value=25)
CRIM = form.number_input('CRIM', min_value=1, max_value=300, value=25)
CHAS = form.number_input('CHAS', min_value=1, max_value=300, value=25)
LSTAT = form.number_input('LSTAT', min_value=1, max_value=300, value=25)
NOX = form.number_input('NOX', min_value=1, max_value=300, value=25)
PTRATIO = form.number_input('PTRATIO', min_value=1, max_value=300, value=25)

predict_button = form.form_submit_button('Predict')

input_dict = {'AGE': AGE,'TAX': TAX, 'RM': RM, 'ZN': ZN, 'INDUS':INDUS, 'DIS': DIS, 'BLACK': BLACK,
              'RAD': RAD,'CRIM': CRIM, 'CHAS': CHAS, 'LSTAT': LSTAT, 'NOX': NOX, 'PTRATIO': PTRATIO}
input_df = pd.DataFrame([input_dict])

if predict_button:
 out = predict(model, input_df)
 st.success('The predicted MEDV are ${:.2f}'.format(out))
