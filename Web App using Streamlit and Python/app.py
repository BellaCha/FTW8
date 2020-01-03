import pandas as pd
import streamlit as st
import joblib
import numpy as np 
#import seaborn as sns

#DATA
Data_URL='data/advertising_regression.csv'

@st.cache
def load_data(nrows):
    data=pd.read_csv(Data_URL,nrows=nrows)
    return data

data=load_data(1000)


def main():
    data =load_data(1000)
    page =st.sidebar.selectbox("Choose a page",["Prediction", "Data Exploration", "Data Visualization"])
    if page == "Prediction":
                               
        # TITLE OF YOUR WEB APPLICATION
        st.title('Advertising Sales Prediction')

        # DESCREIBE YOUR WEB APP
        st.write('We demonstate how we can predict advertising sales based on ad expenditure') 
        st.subheader('Given the following Advertising Costs')

       #Show Histogram (sales)
          # CHECK MIN AND MAX ON YOUR DATA PROFILLING/ANALYSIS FOR REFERENCE
       #a. TV SIDEBAR
        TV=st.slider('TV Advertising Cost',0,300,150)  # MIN (O), MAX(300), Default(150)

       #b. RADIO SIDEBAR
        radio=st.slider('Radio Advertising Cost',0,50,25)

       #c. NEWSPAPER SIDEBAR
        newspaper=st.slider('Newspaper Advertising Cost',0,120,25)  

        #load Saved Machine Learning Model
        saved_model=joblib.load('advertising_model.sav')

        #Predict sales using Variable/features
        predicted_sales=saved_model.predict([[TV,radio,newspaper]])[0]

        #print predictions
        st.success(f"Predicted sales is {predicted_sales} dollars.")


        st.subheader('Sales Ad Cost Distribution')
        hist_values=np.histogram(data.sales,bins=30,range=(0,30))[0]
   
          #Show bar chart
        st.bar_chart(hist_values)
    
    if page == "Data Exploration":
        st.title('Explore the Advertising Data-Set')
        st.subheader('Show Data Set')
        if st.checkbox('Full Data Set'):
             data
        if st.checkbox('Selected Data Set'):
        

#Select ramdom rows
           selected_indices = st.multiselect('Select rows 0-199 :', data.index)
           selected_rows = data.loc[selected_indices]
           st.write('### Selected Data Set', selected_rows)        


 #SHOW SUMMARY

        st.subheader('Show Data Information')
        if st.button("Data Shape"):
           st.write (data.shape)
        if st.button("Data Columns"):
           st.write (data.columns)
        if st.button("Data Types"):
           st.write(data.dtypes)
        if st.button("Data Describe"):
           st.write(data.describe())   




    if page == "Data Visualization":
        
    ###CREATE HISTOGRAM/BAR CHART
        st.subheader("Data Visualization")
       

    #Show Histogram (RADIO)
        st.subheader('Radio Ad Cost Distribution')
        hist_values=np.histogram(data.radio,bins=50,range=(0,50))[0]
   
    #Show bar chart
        st.bar_chart(hist_values)

    #Show Histogram (TV)
        st.subheader('TV Ad Cost Distribution')
        hist_values=np.histogram(data.TV,bins=300,range=(0,300))[0]
   
    #Show bar chart
        st.bar_chart(hist_values)

    #Show Histogram (NEWSPAPER)
        st.subheader('NewsPaper Ad Cost Distribution')
        hist_values=np.histogram(data.newspaper,bins=120,range=(0,120))[0]
   
    #Show bar chart
        st.bar_chart(hist_values)
        
main()





