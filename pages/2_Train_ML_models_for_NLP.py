import streamlit as st
import pandas as pd

import scripts.linear_classifiers_functions

st.title("Train ML models for NLP")
st.subheader("""
             Classification models
             - Data must have a column 'text' and a column 'label'.
             """)

st.subheader("Training data")
train_file = st.file_uploader("Upload your training data", type=("csv"))


st.subheader("Testing data")
test_file = st.file_uploader("Upload your testing data", type=("csv"))

if train_file is not None:
    model_name = st.sidebar.selectbox(
    'Select Model',
    ('RFC','KNN'))
    
    if model_name in ['RFC','KNN']:
        embedding_method = st.sidebar.selectbox(
        'Select Embedding method (default: TFIDF)',
        ('OHE','TF', 'TFIDF')).lower()

