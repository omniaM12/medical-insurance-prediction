
import streamlit as st
import numpy as np
import pandas as pd
import joblib
 
poly_reg= joblib.load("poly_reg_model")
pfull_pipeline=joblib.load("pfull_pipeline")
medins=pd.read_csv("insurance.csv")
st.title("prediction of medical insurance charges")
st.write("""this is app predics the insurance charges""")
age=st.slider("age", float(medins["age"].min()),float(medins["age"].max()))                     
bmi=st.slider("bmi", float(medins["bmi"].min()),float(medins["bmi"].max()))
region= st.selectbox("region",("southeast","southwest","northwest", "northeast"))
children=st.selectbox("children",("1", "2", "3", "4","5"))
sex=st.selectbox("sex",("male","female"))
smoker=st.selectbox("smoker",("yes","no"))
#dict
user_data={"children":children, "age":age,"bmi":bmi, "region":region,"sex":sex,"smoker":smoker}
inspara=pd.DataFrame(user_data, index=[0])
inspara_ready=pfull_pipeline.transform(inspara)
#predict_model
medins_predictions=poly_reg.predict(inspara_ready)
#display
st.markdown("""# $ {} """.format(medins_predictions))
 
