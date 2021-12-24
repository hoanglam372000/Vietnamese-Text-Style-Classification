#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModel
from datasets import Dataset
from datasets import load_metric
import re 
from underthesea import word_tokenize
import wget



#"""# Hyperparameters"""

# setup dataset & tokenizer
num_labels = 6
max_seq_len = 256
overwrite_output_dir = True


#"""# Load Model + tokenizer"""

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
#model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels= num_labels)



def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length = max_seq_len)

import gdown
#wget.download('https://drive.google.com/uc?export=download&id=1-6LAadf5ccHnjVFVNHI-AXoul1WRebYH')
# url_json = "https://drive.google.com/uc?export=download&id=1-6LAadf5ccHnjVFVNHI-AXoul1WRebYH"
# output_json = "config.json"
# gdown.download(url_json, output_json,quiet=False)

# url_bin = "https://drive.google.com/uc?export=download&id=1-852eJpSBLwY6SrFj-RV0qhohX8A0S6t"
# output_bin = "pytorch_model.bin"
# gdown.download(url_bin, output_bin,quiet=False)

#wget.download('https://drive.google.com/uc?export=download&id=1-852eJpSBLwY6SrFj-RV0qhohX8A0S6t')
import shutil
# os.mkdir('final_model_6C')

url_zip = "https://drive.google.com/uc?export=download&id=1zXQySyx7gWqhOMN3_f9SnSi6PGbLr682"
output_zip = "final_model_6C.zip"
gdown.download(url_zip, output_zip,quiet=False)

shutil.unpack_archive('final_model_6C.zip')
# shutil.move('config.json', './final_model_6C')
# shutil.move('pytorch_model.bin','./final_model_6C')
#
model_load = AutoModelForSequenceClassification.from_pretrained("final_model_6C")

#"""## Trainer for model"""

# arguments for Trainer
inf_batch_size = 64

test_args = TrainingArguments(
    output_dir = "tmp_trainer",
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = inf_batch_size    
)

trainer = Trainer(
    model=model_load, 
    args = test_args                       
)

#"""## Predict"""

from scipy.special import softmax

def preprocess(txt):
    def clean_html(raw_html):
        CLEANR = re.compile('<.*?>')    
        cleantext = re.sub(CLEANR, '', raw_html)
        return cleantext

    def remove_special_chars(txt):
        regex = r"[^,.!\"\'(...);\w\s]"
        return re.sub(regex, '', txt)

    def normalize_unicode(txt):
        def loaddicchar():
            dic = {}
            char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
                '|')
            charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
                '|')
            for i in range(len(char1252)):
                dic[char1252[i]] = charutf8[i]
            return dic
        
        
        dicchar = loaddicchar()
        

        return re.sub(r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
                    lambda x: dicchar[x.group()], txt)

    def tokenize(txt):
        return word_tokenize(txt, format="text")

    temp = txt
    temp = clean_html(temp)
    temp = remove_special_chars(temp)
    temp = normalize_unicode(temp)
    temp = temp.strip()

    temp = tokenize(temp)
    return temp

def predict(trainer, text):
    def create_dataset_inference(text):
        df = pd.DataFrame({'text': text})
        df['text'] = df['text'].astype(str)
        df['text'] = df['text'].apply(preprocess)

        dataset = {'text': df['text']}
        dataset = Dataset.from_dict(dataset)
        return dataset

    dataset = create_dataset_inference(text)
    dataset = dataset.map(tokenize_function, batched=True)
    # predict
    preds = np.array(trainer.predict(dataset)[0])

    # convert to label
    # label = np.argmax(preds, axis = 1)
    prob = softmax(preds, axis = 1).reshape(-1)
    pred_df = pd.DataFrame({'label': [0, 1, 2, 3, 4, 5], 'probability': prob}).reset_index(drop = True)
    pred_df = pred_df.sort_values(by = 'probability', ascending = False)

    pred_df['probability']= pred_df['probability'].apply(lambda x: x * 100)
    pred_df = pred_df.round({'probability': 2})

    # remove 0s
    pred_df['probability'] = pred_df['probability'].astype(str)
    pred_df['probability'] = pred_df['probability'].str.rstrip('0')   

    
    # pred_df = pd.DataFrame({'text': text, 'label': label, 'probability': prob})
    pred_df['label'] = pred_df['label'].replace([0,1,2,3,4,5], ['Phong cách báo chí','Phong cách sinh hoạt hằng ngày','Phong cách nghệ thuật', 'Phong cách khoa học', 'Phong cách hành chính','Phong cách chính luận'])
    final_df = pred_df.reset_index(drop = True)
    return final_df.loc[0,label], final_df

