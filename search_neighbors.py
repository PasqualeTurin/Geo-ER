import argparse
import os
from os import listdir
from os.path import isfile, join
from io import open
import torch
import sys
from torch import nn, optim
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer
import numpy as np
from collections import Counter
from torch.autograd import Variable
import random
import time
import math
import json
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from ast import literal_eval
from difflib import SequenceMatcher
from functions import compute_dist, similar, jaccard_similarity

city = 'sin'

NEIGHBORHOOD_RADIUS = 1000
HIDDEN_SIZE = 768

language_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

csv_path_osm = 'data_csv/'+city+'/osm_'+city+'.csv'
osm_dataset = pd.read_csv(csv_path_osm, index_col=0).fillna(' ')
csv_path_yelp = 'data_csv/'+city+'/yelp_'+city+'.csv'
yelp_dataset = pd.read_csv(csv_path_yelp, index_col=0).fillna(' ')

dataset = osm_dataset.append(yelp_dataset, ignore_index=True)

train_path = 'train_valid_test/'+city+'/train.txt'
valid_path = 'train_valid_test/'+city+'/valid.txt'
test_path = 'train_valid_test/'+city+'/test.txt'

train_path_out = 'neighborhood_train_valid_test/'+city+'/n_train.txt'
valid_path_out = 'neighborhood_train_valid_test/'+city+'/n_valid.txt'
test_path_out = 'neighborhood_train_valid_test/'+city+'/n_test.txt'

for path in [train_path, valid_path, test_path]:

    entries = []

    if path == train_path:
        out_path = train_path_out
        print('Preparing train neighborhood data...')
    elif path == valid_path:
        out_path = valid_path_out
        print('Preparing valid neighborhood data...')
    else:
        out_path = test_path_out
        print('Preparing test neighborhood data...')


    with open(path, 'r') as f:
    
        for line in f:
        
            e1 = line.split('\t')[0].lower()
            e2 = line.split('\t')[1].lower()
            
            name1 = []
            name2 = []
            
            words = e1.split()
            
            for i,word in enumerate(words):
                if words[i-1] == 'val' and words[i-2] == 'name':
                    j=i
                    while not (words[j]=='col' and words[j+1]=='latitude'):
                        name1.append(words[j])
                        j+=1
              
              
                if words[i-1] == 'val' and words[i-2] == 'latitude':
                    lat1 = word
                if words[i-1] == 'val' and words[i-2] == 'longitude':
                    long1 = word
              
            words = e2.split()
                        
            for i,word in enumerate(words):
                if words[i-1] == 'val' and words[i-2] == 'name':
                    j=i
                    while not (words[j]=='col' and words[j+1]=='latitude'):
                        name2.append(words[j])
                        j+=1
                        
                if words[i-1] == 'val' and words[i-2] == 'latitude':
                    lat2 = word
                if words[i-1] == 'val' and words[i-2] == 'longitude':
                    long2 = word
            
            name1 = ' '.join(name1)
            name2 = ' '.join(name2)
            
            neighborhood1 = []
            neighborhood2 = []
            
            distances1 = []
            distances2 = []
            
            
            x = tokenizer.tokenize('[CLS] ' + name1 + ' [SEP]')
            x = tokenizer.convert_tokens_to_ids(x)
            x = torch.tensor(x).view(1,-1)
            x = language_model(x)[0][:,0,:].view(-1).detach().numpy()
            neighborhood1.append(x)
            
            
            
            x = tokenizer.tokenize('[CLS] ' + name2 + ' [SEP]')
            x = tokenizer.convert_tokens_to_ids(x)
            x = torch.tensor(x).view(1,-1)
            x = language_model(x)[0][:,0,:].view(-1).detach().numpy()
            neighborhood2.append(x)
            
            
            for i in range(dataset.shape[0]):
                row = dataset.iloc[i]
                
                dist = compute_dist(lat1, long1, str(row['latitude']), str(row['longitude']))
                
                try:
                    dist = int(dist)
                except ValueError:
                    continue
                
                if (jaccard_similarity(name1.split(), row['name'].lower().split()) > 0.4 or similar(name1, row['name'].lower()) > 0.6) and dist < NEIGHBORHOOD_RADIUS:
                    
                    if (name1 == row['name'].lower() and str(row['latitude']) == lat1 and str(row['longitude']) == long1) or (name2 == row['name'].lower() and str(row['latitude']) == lat2 and str(row['longitude']) == long2):
                        continue
                
                
                    x = tokenizer.tokenize('[CLS] ' + row['name'].lower() + ' [SEP]')
                    x = tokenizer.convert_tokens_to_ids(x)
                    x = torch.tensor(x).view(1,-1)
                    x = language_model(x)[0][:,0,:].view(-1).detach().numpy()
                    neighborhood1.append(x)
                    distances1.append(dist)
                  
                  
                dist = compute_dist(lat2, long2, str(row['latitude']), str(row['longitude']))
                
                try:
                    dist = int(dist)
                except ValueError:
                    continue
                
                if (jaccard_similarity(name2.split(), row['name'].lower().split()) > 0.4 or similar(name2, row['name'].lower()) > 0.6) and dist < NEIGHBORHOOD_RADIUS:
                    
                    if (name1 == row['name'].lower() and str(row['latitude']) == lat1 and str(row['longitude']) == long1) or (name2 == row['name'].lower() and str(row['latitude']) == lat2 and str(row['longitude']) == long2):
                        continue
                
                    x = tokenizer.tokenize('[CLS] ' + row['name'].lower() + ' [SEP]')
                    x = tokenizer.convert_tokens_to_ids(x)
                    x = torch.tensor(x).view(1,-1)
                    x = language_model(x)[0][:,0,:].view(-1).detach().numpy()
                    neighborhood2.append(x)
                    distances2.append(dist)
                    
                    
            
            if len(neighborhood1) < 2:
                neighborhood1.append(np.zeros(HIDDEN_SIZE))
                distances1.append(NEIGHBORHOOD_RADIUS)
            
            if len(neighborhood2) < 2:
                neighborhood2.append(np.zeros(HIDDEN_SIZE))
                distances2.append(NEIGHBORHOOD_RADIUS)
            
            
            
            #print(len(neighborhood1))
            #print(len(neighborhood1[0]))
            
            #time.sleep(10)
            
            
            entry = [neighborhood1, distances1, neighborhood2, distances2]
            entries.append(entry)
            
            
    
    with open(out_path, 'wb') as f:
        pickle.dump(entries, f)
