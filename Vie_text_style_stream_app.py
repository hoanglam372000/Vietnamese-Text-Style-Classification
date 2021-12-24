#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import model_prediction
import streamlit as st

desc = "Vietnamese Text Style Classification!"

st.title('Vietnamese Text Style Classification')
st.write(desc)

user_input = st.text_input('Vietnamse Text (input a paragraph)')


if st.button('Check Text Style'):
    check_text_style = model_prediction.predict(trainer, [str(user_input)]).loc[0,'label']
    st.write(check_text_style)

