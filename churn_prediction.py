import streamlit as st
import numpy as np
import pandas as pd
import pickle

# menampilkan judul
st.title('Churn Predictor')
st.write('This website can be used to predict churn rate ABC Bank Customer')

# menambahkan sidebar
st.sidebar.header("Please input customer's features")

# membuat customer input
def create_user_input():
    # numerical
    credit_score=st.sidebar.number_input('credit_score',min_value=350,max_value=850,value=500)
    age=st.sidebar.slider('age',min_value=18,max_value=92,value=30)
    tenure=st.sidebar.slider('tenure',min_value=0,max_value=10,value=5)
    balance=st.sidebar.number_input('balance',min_value=0.0,max_value=250898.0,value=100000.0)
    products_number=st.sidebar.radio('products_number', [1,2,3,4])
    credit_card=st.sidebar.radio('credit_card',[0,1])
    active_member=st.sidebar.radio('active_member',[0,1])
    estimated_salary=st.sidebar.number_input('estimated_salary',min_value=11.58,max_value=199992.48,value=100000.0)

    # categorical
    gender=st.sidebar.radio('gender',['Female','Male'])
    country=st.sidebar.radio('country',['Germany','Spain','France'])

    # create data frame
    user_df=pd.DataFrame([
        {
            'credit_score':credit_score,
            'country':country,
            'gender':gender,
            'age':age,
            'tenure':tenure,
            'balance':balance,
            'products_number':products_number,
            'credit_card':credit_card,
            'active_member':active_member,
            'estimated_salary':estimated_salary
        }

    ])
    return user_df

# define customer data
data_customer=create_user_input()

# create 2 containers
col1,col2=st.columns(2)

# left
with col1:
    st.subheader('Customer Feature')
    st.write(data_customer.T.rename(columns={0:'Feature Name',1:'Value'}))

# load model
with open('best_model.sav','rb') as f:
    model_loaded=pickle.load(f)

# predict to customer data
target=model_loaded.predict(data_customer)  # target 1 atau 0
probability=model_loaded.predict_proba(data_customer)[0]

# menampilkan hasil prediksi
# right
with col2:
    st.subheader('Prediction Result')
    if target == 1:
        st.write("This customer will CHURN")
    else:
        st.write("This customer will NOT CHURN")
    # display probability
    st.write(f"Probability of Churn: {probability[1]:.2f}")




