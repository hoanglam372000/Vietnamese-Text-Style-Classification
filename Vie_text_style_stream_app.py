#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import model_prediction
import streamlit as st

desc = "Vietnamese Text Style Classification!"

st.title('Vietnamese Text Style Classification')
st.write(desc)

user_input = st.text_area('Vietnamse Text (input a paragraph)')


if st.button('Check Text Style'):
    main_style, probs_refer =  model_prediction.predict(model_prediction.trainer, [str(user_input)])
    st.markdown(f'Phong cách ngôn ngữ chính của văn bản: **{main_style}**')
    st.write('Bảng xác suất các phong cách ngôn ngữ cho văn bản trên: ')
    st.dataframe(probs_refer)

