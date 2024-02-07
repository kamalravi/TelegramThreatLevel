#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries needed
import pandas as pd
from collections import OrderedDict

import glob
from natsort import natsorted
import json
import tldextract
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import random
random.seed(42)


# In[2]:


# lib_subs = ['Socialism_101', 'progressive', 'socialism', 'obama', 'occupywallstreet', 'neoliberal', 'democrats']
# con_subs = ['askaconservative', 'NolibsWatch', 'Romney', 'neoconNWO', 'Republican', 'Conservative']
# res_subs = ['far_right', 'DylannRoofInnocent', 'alllivesmatter', 'pol', 'Physical_Removal', 'nazi', 'WhiteRights', 'ZOG', 
#             'NationalSocialism', 'paleoconservative', '911truth', 'tea_party', 'HBD', 'CringeAnarchy', 'uncensorednews',
#             'ChapoTrapHouse', 'new_right', 'The_Donald']
# after removing None json files
lib_subs = ['progressive', 'socialism', 'obama', 'occupywallstreet', 'neoliberal', 'democrats']
con_subs = ['askaconservative', 'NolibsWatch', 'Romney', 'neoconNWO', 'Republican', 'Conservative']
res_subs = ['DylannRoofInnocent', 'alllivesmatter', 'pol', 'Physical_Removal', 'nazi', 'WhiteRights', 'ZOG', 
            'NationalSocialism', 'paleoconservative', '911truth', 'tea_party', 'HBD', 'CringeAnarchy', 'uncensorednews',
            'new_right', 'The_Donald']


# In[3]:


subs_dir = "/home/ravi/PROJECTS_DATA/RedditNewsDataCombined_filterURLdomains_TextAll_FilterNaN_FilterDup_FilterURLsInText_WC_Filter250WC/"

def mergeSubreddits(subs):
    dfs = []    
    for sub in subs:
        df = pd.read_json(subs_dir+'subreddit_'+sub+'.json')
        # To replace NoneType with empty string
        df['author_flair_text'] = [str(ii or '') for ii in df['author_flair_text'].tolist()]
        # print(df)
        df = df.sort_index(axis=1)
        # print(df)
        # break
        dfs.append(df)

    mergeDF = pd.concat(dfs, ignore_index=True)
    
    return mergeDF.sample(frac=1, random_state=42, ignore_index=True)


# In[4]:


libDF = mergeSubreddits(lib_subs)
libDF['label']=0
conDF = mergeSubreddits(con_subs)
conDF['label']=1
resDF = mergeSubreddits(res_subs)
resDF['label']=2


# In[5]:


# libDF
print("writing to json")
libDF.to_json("/home/ravi/PROJECTS_DATA/DataModelsResults/Data/libDF.json", orient="records", default_handler = str)

# In[6]:


# conDF
print("writing to json")
conDF.to_json("/home/ravi/PROJECTS_DATA/DataModelsResults/Data/conDF.json", orient="records", default_handler = str)

# In[7]:


# resDF
print("writing to json")
resDF.to_json("/home/ravi/PROJECTS_DATA/DataModelsResults/Data/resDF.json", orient="records", default_handler = str)

# In[8]:


print(libDF.shape)
print(conDF.shape)
print(resDF.shape)


# In[9]:


AllClassesDF = pd.concat([libDF, conDF, resDF], ignore_index=True) 
AllClassesDF = AllClassesDF.sample(frac=1, random_state=42, ignore_index=True)


# In[10]:


# AllClassesDF
print(AllClassesDF.shape)


# In[11]:

print("writing to json")
AllClassesDF.to_json("/home/ravi/PROJECTS_DATA/DataModelsResults/Data/AllClassesDF.json", orient="records", default_handler = str)


# In[ ]:




