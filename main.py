#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# !pip install pysbd
# !pip install tqdm
# !pip install ftfy

import re
import pysbd
from nltk import ngrams
import numpy as np
from tqdm import tqdm
import pandas as pd
import ftfy

def get_ngrams(text, n=3): 
    return list(ngrams(text, 3)) # ngrams with n=3 divides into trigrams , ie 3 words in each sequence to for,m meaningful sentences

def get_jaccard_simm(ngrams1, ngrams2):
    ngrams1, ngrams2 = set(ngrams1), set(ngrams2)
    return len(ngrams1.intersection(ngrams2))/len(ngrams1.union(ngrams2))

seg = pysbd.Segmenter(language="en", clean=False) # Segmenting into sentences that are meaningful. Specfying english language out of the 22 languages available

# summ = "/home/mann/All Data/1/PRS Committee Reports/Activities and Functioning of India Trade Promotion Organisation/Activities and Functioning of India Trade Promotion Organisation_summary.txt"
# text = "/home/mann/All Data/1/PRS Committee Reports/Activities and Functioning of India Trade Promotion Organisation/raw_Activities and Functioning of India Trade Promotion Organisation.txt"

# INPUT SUMMARY FILE
summ = "/home/mann/Annotation/Niti Ayog Reports/NITI-Aayog-_26-World-Bank-Energy-Water-Agriculture-Nexus/NITI-Aayog-_26-World-Bank-Energy-Water-Agriculture-Nexus_summ.txt"
# INPUT TEXT FILE
text = "/home/mann/Annotation/Niti Ayog Reports/NITI-Aayog-_26-World-Bank-Energy-Water-Agriculture-Nexus/raw_NITI-Aayog-_26-World-Bank-Energy-Water-Agriculture-Nexus.txt"

with open(summ, 'r') as summ:
    summ = ftfy.fix_text(summ.read()) 
    
with open(text, 'r', encoding='latin-1') as text: # encoding='latin-1' is used to read the file with special characters
    text = ftfy.fix_text(text.read())
    
summ = summ.replace("\n", ' ').lower() # Replacing new line with space and converting to lower case
text = text.replace("\n", ' ').lower()

summ = re.sub(r'\s{2,}', ' ', summ, flags=re.I | re.M)  # Removing extra spaces (sub used for substitution) of \s{2,} (2 or more spaces) with ' ' (single space). re.I is for ignoring case and re.M is for multiline
text = re.sub(r'\s{2,}', ' ', text, flags=re.I | re.M)

summ = seg.segment(summ)    # Segmenting into sentences
text = seg.segment(text)

summ_ngrams = list(map(get_ngrams, summ))  
text_ngrams = list(map(get_ngrams, text))

# sim = np.zeros((len(text_ngrams), len(summ_ngrams)))
sim = np.zeros((len(summ_ngrams), len(text_ngrams)))  # sim is an empty similarity matrix of size (no. of sentences in summary, no. of sentences in text)
for i in tqdm(range(len(summ_ngrams))):   
    for j in range(len(text_ngrams)):
        sim[i, j] = get_jaccard_simm(summ_ngrams[i], text_ngrams[j]) # similarity between each sentence of summary and text is calculated and stored in sim matrix
        
# sim = sim.T

top_n = 5
# args = np.argmax(sim, axis=-1)
args = np.argsort(sim).T[::-1].T  #return the indexes of the sorted sim array in descending order ie finding the most similar candidates for each summary trigram
args = args.T[:top_n].T  

text = np.array(text)

data = []
for i in range(args.shape[0]): # args.shape[0] is the no. of sentences in summary
    summ_ = summ[i] 
    data.append([summ_] + text[args[i]].tolist()) # appending the summary sentence and the top_n candidates to data
    
data = pd.DataFrame(data, columns=["Summary"] + [f"candidate{i}" for i in range(top_n)])

save_name = "output_bertscore_" + "PUT_YOUR_NAME_HERE" + ".csv"

data.to_csv(save_name, index=False)
