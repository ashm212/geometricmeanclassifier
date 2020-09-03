# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 22:38:16 2020

@author: ashu0
"""

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type based on where the geometric mean of the flower's features lands!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.number_input('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.number_input('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.number_input('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.number_input('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')

iris = datasets.load_iris()

import scipy
from scipy import stats
print(df.iloc[:,0:4])
x=scipy.stats.gmean(df.iloc[:,0:4],axis=1)
df['geometric mean'] = x

conditions = [
    (df['geometric mean'] <= 2.33),
    (df['geometric mean'] > 2.33) & (df['geometric mean'] <= 3.47),
    (df['geometric mean'] > 3.47) & (df['geometric mean'] <= 100),
    ]

labels =['setosa','versicolor','virginica']

df['prediction'] = np.select(conditions, labels)
col_list = ['geometric mean', 'prediction']

df=df[col_list]
st.write(df)

