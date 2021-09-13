import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
#App prediksi iris flower pakai streamlit

apliaksi ini untuk memprediksi tiper bunga iris

""")

st.sidebar.header('input parameter yang dimasukkan user') #untuk panel sidebar

#define input yang dimasukkan oleh user
def user_input_features():
    sepal_length = st.sidebar.slider('sepal_length', 3.0, 8.0, 5.5) #ukuran slider yang dimunculkan pada frontend
    sepal_width = st.sidebar.slider('sepal_width', 1.5, 4.0, 7.5)
    petal_length = st.sidebar.slider('petal_length', 4.0, 8.5, 3.5)
    petal_width = st.sidebar.slider('petal_width', 3.0, 8.0, 5.5)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[1]) #menjelaskan data frames yang dimasukkan
    return features
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)