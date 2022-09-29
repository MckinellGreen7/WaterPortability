import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.metrics import plot_roc_curve
from streamlit_lottie import st_lottie
import requests
import PIL.Image
st.set_page_config(page_title="Hydr8", page_icon=":tada:", layout= "wide")

from PIL import Image
image = Image.open("C:\\Users\\Lenovo\\Desktop\\Streamlit Project\\hydr8.png")
image2 = Image.open("C:\\Users\\Lenovo\\Desktop\\Streamlit Project\\Data\\22hydr8.png")
header = st.container()
from PIL import Image
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.write('##')
        original_title = '<p style=" text-align: left; background: linear-gradient(to right, #a8c0ff, #3f2b96);color:transparent;background-clip:text;-webkit-background-clip: text; font-weight: Bold; font-size: 70px;" class = "heading">HYDR8</p>'
        st.markdown(original_title, unsafe_allow_html=True)
    with right_column:
        st.image(image2, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()
lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_q5qeoo3q.json")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("C:\\Users\\Lenovo\\Desktop\\Streamlit Project\\css\\code.css")



    #original_script = '<p style=" text-align: center; font-family:Lato; font-size: 18px; font-weight: Bold; border-width:3px; border-style:solid; border-color:#FF0000; padding: 1em;">   In this project we will find out whether the water is potable or not.</p>'
    #st.markdown(original_script, unsafe_allow_html=True)

with st.container():
    st.subheader("#")
    st.title("Introducing Water Quality prediction to everyone")
    st.write("Hello amigo. Ever wondered the water which you are drinking is fresh or not\n No worries cause here we come to lend you a helping hand with our ML model - 'Hydr8'")
with st.container():
    st.write("...")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What is Hydr8?")
        st.write("##")
        st.write(
        """
        “Hydr8” is a machine learning model project that works to predict if water content to be predicted is portable or not. Data on parameters like pH, Total Dissolvable Solid value, Total Organic Carbon value, sulphate content value, chloramines content value, conductivity and hardness of water are considered to train and test the model. Our model “Hydr8”, so trained and tested results with a maximum efficiency of nearly seventy percent. “Hydr8” also helps visualize data in the most efficient way in the form of graphs.
        """
        )
    with right_column:
            st_lottie(lottie_coding, height = 500, quality="high", key="AI powered predictor")





dataset = st.container()
features = st.container()
inputs = st.container()
time = st.container()





    






#@st.cache(for running it for one time if the file name is same then its not going to run again)
#def get_data():
#  taxi_data=pd.read_csv

# For customization u can use css:
# =============================================================================
# st.markdown(
#     """
#     <style>
#     .main {
#     background-color:#F5F5F5;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True)
# ============================================================================
    
#st.markdown("![Alt Text](https://www.edureka.co/blog/wp-content/uploads/2018/08/Insurance-Leadspace-Aniamted.gif)")
    

st.write("###")
st.write("###")
st.write("###")

st.write("###")
st.write("###")
st.write("###")
with dataset:
    st.header("Water Potability Dataset")
    st.text("I found this data set on Kaggle")
    
    df=pd.read_csv(r"C:\Users\Lenovo\Desktop\Streamlit Project\water-potability11.csv")
    st.write(df.head())
    
    st.subheader("Potability of Water")
    a=df['Potability'].value_counts()
    st.bar_chart(a,height=500)
    
    
    
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()
lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_5tl1xxnz.json")



st.write("###")
st.write("###")
st.write("###")
with st.container():
    with left_column:
        st.header("The features on which the model is trained")
        st.markdown('* pH of Water')
        st.markdown('* Hardness of Water')
        st.markdown('* Solids in water')
        st.markdown('* Chloroamines in water')
        st.markdown('* Sulfate in water')
        st.markdown('* Conductivity of water')
        st.markdown('* Organic Carbon in water')
        st.markdown('* Trihalomethanes in water')
        st.markdown('* Turbidity in water')
        st.markdown('* Potability of water')
        
    with right_column:
            st_lottie(lottie_coding, height = 600, quality="high", key="AI powered")
    
    
    

    
with inputs:
    st.header("Time to take input from the user")
    st.text("Enter the inputs according to the parameters of the water taken")
    
    sel_col,disp_col = st.columns(2)
    
    ph=sel_col.slider('pH',min_value=0,max_value=14,value=7)
    hardness=sel_col.selectbox("Hardness",options=[50,100,150,200,250],index=0)
    solids=sel_col.slider('Solids',min_value=10000,max_value=25000,value=15000)
    chloroamines=sel_col.slider('Chloroamines',min_value=1,max_value=14,value=7,step=1)
    sulfate=sel_col.slider('Sulfate',min_value=50,max_value=400,value=200)
    conductivity=sel_col.slider('Conductivity',min_value=100,max_value=500,value=300)
    organic_carbon=sel_col.slider('Organic Carbon',min_value=5,max_value=25,value=14)
    trihalomethanes=sel_col.slider('Trihalomethanes',min_value=20,max_value=100,value=50)
    turbidity=sel_col.slider('Turbidity',min_value=1,max_value=7,value=4)    
    #n_estimators=sel_col.slider('How many trees should be there', min_value=10,max_value=1000,value=200,step=10)
    arr=[[ph,hardness,solids,chloroamines,sulfate,conductivity,organic_carbon,trihalomethanes,turbidity]]
    #input_feature = sel_col.text_input("Which feature should be used as the input feature","pH")
    
    clf=RandomForestClassifier(max_depth=None,n_estimators=310,min_samples_leaf=5,min_samples_split=7)
    X=df.drop('Potability',axis=1)
    y=df['Potability']
    clf.fit(X,y)
    
    st.write("###")
    y_pred=clf.predict(arr)
    
    if y_pred==1:
        import time
        with st.spinner(text='Calculating..'):
            time.sleep(1)
            st.success('### Water is Potable')
        #st.write("""""")
    else:
        import time
        with st.spinner(text='Calculating..'):
            time.sleep(1)
            st.error('###  Water is not Potable')
           
        #st.write("### Water is not Potable")
        
# =============================================================================
# do_sleep = True
# import hydralit_components as hc
# with hc.HyLoader('Now  Calculating ',hc.Loaders.pulse_bars,):
#     time.sleep(loader_delay)
# =============================================================================
    

with st.container():
    st.write("###")
    st.write("###")
    st.write("###")
    st.write("###")
    st.write("###")
    st.header("Want to collaborate with our projects in Future!\nThen get in Touch with us ")
    st.write("##")
    contact_form = '''
    <form action="https://formsubmit.co/shivamanik593@gmail.com" method="POST">
        <input type ="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder = "Your name" required>
        <input type="email" name="email" placeholder = "Your mail_id" required>
        <textarea name = "message" placeholder="Your message" required></textarea>
        <button type="submit">Send</button>
    </form>
    '''
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    
    
    
    
    

    
    
    
