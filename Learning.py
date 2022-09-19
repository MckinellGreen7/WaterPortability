import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.metrics import plot_roc_curve

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Welcome to my awesome data science project!")
    st.text("In this project we will find out whether the water is potable or not")

with dataset:
    st.header("Water Potability Dataset")
    st.text("I found this data set on Kaggle")
    
    df=pd.read_csv(r"C:\Users\hp\Desktop\Water-Potability\water-potability11.csv")
    st.write(df.head())
    
    st.subheader("Potability of Water")
    a=df['Potability'].value_counts()
    st.bar_chart(a)


with features:
    st.header("The features I created")
    st.markdown('* **first feature:** pH of the water')
    
with model_training:
    st.header("Time to Train our model")
    st.text("Here we get to choose the hyperparameters for our model")
    
    sel_col,disp_col = st.columns(2)
    
    ph=sel_col.slider('pH',min_value=0,max_value=14,value=7)
    hardness=sel_col.selectbox("Hardness",options=[50,100,150,200,250],index=0)
    solids=sel_col.slider('Solids',min_value=10000,max_value=25000,value=15000)
    chloroamines=sel_col.slider('Chloroamines',min_value=1,max_value=14,value=7,step=1)
    sulfate=sel_col.slider('Sulfate',min_value=50,max_value=400,value=200,step=50)
    conductivity=sel_col.slider('Conductivity',min_value=100,max_value=500,value=300,step=50)
    organic_carbon=sel_col.slider('Organic Carbon',min_value=5,max_value=25,value=14,step=2)
    trihalomethanes=sel_col.slider('Trihalomethanes',min_value=20,max_value=100,value=50,step=5)
    turbidity=sel_col.slider('Turbidity',min_value=1,max_value=7,value=4,step=1)    
    #n_estimators=sel_col.slider('How many trees should be there', min_value=10,max_value=1000,value=200,step=10)
    arr=[[ph,hardness,solids,chloroamines,sulfate,conductivity,organic_carbon,trihalomethanes,turbidity]]
    #nput_feature = sel_col.text_input("Which feature should be used as the input feature","pH")
    
    clf=RandomForestClassifier(max_depth=None,n_estimators=310,min_samples_leaf=5,min_samples_split=7)
    X=df.drop('Potability',axis=1)
    y=df['Potability']
    clf.fit(X,y)
    
    y_pred=clf.predict(arr)
    if y_pred==1:
        st.write("""
                 ### Water is Potable
                 """)
    else:
        st.write("### Water is not Potable")