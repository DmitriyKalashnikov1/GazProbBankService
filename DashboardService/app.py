import numpy as np
import requests
import json
import pandas as pd
import streamlit as st

df = pd.read_csv('../data/dataset.csv')

st.write("## There could have been a cool dashboard here, but we didn't have time to make it, so for now, take a look at our dataset from https://www.banki.ru.")
st.write("## It took three days to parse it.")

st.dataframe(data=df)

st.write("## Test form for topics and sentiments detection")
text = st.text_area("TestText")

submit = st.button("Submit")

if submit:
    st.write("wait...")
    jsonTest = json.dumps([{
        "id": 1,
        "text": text
    }])
    resp = requests.post("http://0.0.0.0:8000/predict/", data=jsonTest, headers={"Content-type": "application/json"})
    st.write(f"## Status code: {resp.status_code}")
    if resp.status_code == 200:
        answ = json.loads(resp.text)[0]
        st.write(f"## Topics: {answ['topics']}")
        st.write(f"## Sentiments: {answ['sentiments']}")

        trunatedSentiments = np.trim_zeros(answ['sentiments'])

        sentDf = pd.DataFrame(trunatedSentiments, index=[f for f in answ['topics']])

        st.bar_chart(data=sentDf)
    else:
        st.write("## Ups! Something wrong with backend!")


