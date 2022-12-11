import streamlit as st
import requests
import PIL.Image
from streamlit_lottie import st_lottie
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#img_1 = PIL.Image.open(r"C:\Users\DELL\Python\img1.png")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()
lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_q5qeoo3q.json")

st.set_page_config(page_title="Hydr8", page_icon=":tada:", layout= "wide")

with st.container():
    st.subheader("By DSI")
    st.title("Introducing Water Quality prediction to everyone")
    st.write("Hello amigo. Ever wondered the water which you are drinking is fresh or not\n No worries cause here we come to lend you a helping hand with our ML model - 'Hydr8Fresh'")
with st.container():
    st.write("...")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What is Hydr8Fresh?")
        st.write("##")
        st.write(
        """
        “Hydr8fresh” is a machine learning model project that works to predict if water content to be predicted is portable or not. Data on parameters like pH, Total Dissolvable Solid value, Total Organic Carbon value, sulphate content value, chloramines content value, conductivity and hardness of water are considered to train and test the model. Our model “Hydr8fresh”, so trained and tested results with a maximum efficiency of nearly seventy percent. “Hydr8fresh” also helps visualize data in the most efficient way in the form of graphs.
        """
        )
    with right_column:
        st_lottie(lottie_coding, height = 600, key="AI powered predictor")

with st.container():
    st.write("...")
    st.header("Previous Project")
    st.write("##")
    image_column, text_column = st.columns((1,2))
    #with image_column:
        #st.image(img_1  )
    with text_column:
        st.header("WOW - Water On Web")
        st.write("##")
        st.write(
        """
        We see a lot of water wastage in some developed parts of the country. So, the website developed by our team can help reduce the water wastage issue in the country. Our project WoW goes with the slogan “DO THE EARTH A FAVOR, BE A WATER SAVER!!!”
        """
        )
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css(r"C:\Users\hp\Desktop\Water-Potability\Untitled-1.css")
with st.container():
    st.write("...")
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

#Clustering_for_multiple_input
def pred():
    p1 = int(input())
    p2 = int(input())
    input = [[0,1],[1,1],[3,2],[4,3],[5,4],[4,5],[0,6],[5,7],[9,8]]
    output = [0,0,1,1,0,0,1,1,0]
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(input,output)
    a = neigh.predict([[p1,p2]])
    return a

with st.container():
    st.title("DATA SET")
    st.write("Data found at kaggle.com")
    st.write("[Go to Kaggle >](https://www.kaggle.com/datasets/adityakadiwal/water-potability?select=water_potability.csv)")
    df=pd.read_csv(r"C:\Users\hp\Desktop\Water-Potability\water-potability11.csv")
    st.write(df.head(51)) #st.write(df.head()) --> prints only first 5 lines of the data.

    g1,g2 = st.columns(2)
    with g1:
        st.subheader("Sulfate content")
        sulfate = pd.DataFrame(df["Sulfate"]) #pd.DataFrame(df["Hardness"].value_counts())
        st.bar_chart(sulfate.head(50)) #for first 500 data
    with g2:
        st.subheader("Chloramine Content")
        chloramines = pd.DataFrame(df["Chloramines"])
        st.bar_chart(chloramines.head(50))

with st.container():
    a,b = st.columns(2)
    a.subheader("Input")
    p1 = a.slider("input 1", min_value = 0, value = 3, max_value = 15, step = 1)  #value=000 implies slider is shown at 000 in the beginning
    p2 = a.slider("input 2", min_value = 0, value = 2, max_value = 15, step = 1)
    nen = a.slider("KNN neighbors value", min_value = 1, value = 3, max_value = 10, step = 1)
    input = [[0,1],[1,1],[3,2],[4,3],[5,4],[4,5],[0,6],[5,7],[9,8],[12,14]]
    output = [0,0,1,1,0,0,1,1,0,2]
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=nen)
    neigh.fit(input,output)
    predict = neigh.predict([[p1,p2]])
    b.subheader("Prediction")
    if predict==1:
        b.write("Portable")
    else:
        b.write("Not Portable")