import streamlit as st
import pandas as pd

import scripts.linear_classifiers_functions

st.title("Train ML models for NLP")
st.subheader("""
             Classification models
             - Data must have a column 'text' and a column 'label'.
             """)

st.subheader("Training data")
data = st.file_uploader("Upload your training data", type=("csv"))


st.subheader("Testing data")
data = st.file_uploader("Upload your testing data", type=("csv"))

