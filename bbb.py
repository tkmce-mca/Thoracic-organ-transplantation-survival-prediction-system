# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:52:02 2020

@author: Aruna
"""


import nltk,re,pprint
from nltk import word_tokenize
f=open('docu.txt').read()
tokens=word_tokenize(f)
words=[w.lower() for w in tokens ]
vocab=sorted(set(words))
#print(vocab)
wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in vocab]
print(vocab)
print("gg")

