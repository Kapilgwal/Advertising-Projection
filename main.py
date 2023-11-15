import streamlit as st
import pickle
import numpy as np
import re

pipe = pickle.load(open('unemployment.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Advertising Projection")

s1 = st.number_input("Enter the spending on the Television")
s2 = st.number_input("Enter the spending on the Radio")
s3 = st.number_input("Enter the spending on the Newspaper")

if st.button("Prediction of Sales"):
    query = np.array([s1,s2,s3],dtype=object)
    query = query.reshape(1,3)

    text = str(pipe.predict(query)[0])
    st.title("The prediction of sales after the sales is " + text)