#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import model_prediction
import streamlit as st
st.markdown("<h1 style='text-align: center; font-size:30px'>VIETNAM NATIONAL UNIVERSITY, HO CHI MINH CITY</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;font-size:30px'>UNIVERSITY OF SCIENCE</h1>", unsafe_allow_html=True)
#st.markdown('<div align="center">**VIETNAM NATIONAL UNIVERSITY, HO CHI MINH CITY**</div>')
#st.markdown('<div align="center">**UNIVERSITY OF SCIENCE**</div>')
#st.image(model_prediction.image)
#desc = st.markdown('**Author:** *Lam Thai Hoang, Tuan Le Ngoc, Hien Pham Thi Hoai, Huy Nguyen Tien, Son Nguyen Truong*')
st.markdown("<h1 style='text-align: center;font-size:40px'>Vietnamese Text Style Classification Tool</h1>", unsafe_allow_html=True)
#st.title('Vietnamese Text Style Classification Tool')
st.markdown('**Author:** *Lam Thai Hoang, Tuan Le Ngoc, Hien Pham Thi Hoai, Huy Nguyen Tien, Son Nguyen Truong*')
#st.write(desc)

user_input = st.text_area('Vietnamse Text (input a paragraph)')


if st.button('Check Text Style'):
    main_style, probs_refer =  model_prediction.predict(model_prediction.trainer, [str(user_input)])
    st.markdown(f'Phong cách ngôn ngữ chính của văn bản: **{main_style}**')
    st.write('Bảng xác suất các phong cách ngôn ngữ cho văn bản trên: ')
    st.dataframe(probs_refer)

