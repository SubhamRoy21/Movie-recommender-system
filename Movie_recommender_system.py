#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.express as plt


# In[2]:


credits = pd.read_csv(r"C:\Users\roysu\Downloads\tmdb_5000_credits.csv")
movies = pd.read_csv(r"C:\Users\roysu\Downloads\tmdb_5000_movies.csv")


# In[3]:


movies = movies.merge(credits,on='title')


# In[4]:


#genre
#id
#keywords
#title
#overview
#cast
#crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[5]:


movies.head()


# In[6]:


movies.isnull().sum()


# In[7]:


movies.dropna(inplace = True)


# In[8]:


movies.duplicated().sum()


# In[9]:


movies.iloc[0].genres


# In[10]:


import ast


# In[11]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[12]:


movies['genres']=movies['genres'].apply(convert)


# In[13]:


movies.head(1)


# In[14]:


movies['keywords']=movies['keywords'].apply(convert)


# In[15]:


movies.head()


# In[16]:


def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[17]:


movies['cast']=movies['cast'].apply(convert3)


# In[18]:


def fetch(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
             L.append(i['name'])
             break
       
    return L


# In[19]:


movies['crew']=movies['crew'].apply(fetch)


# In[20]:


movies.head()


# In[21]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[22]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[23]:


movies.head()


# In[24]:


movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[25]:


new_df = movies[['movie_id','title','tags']]


# In[26]:


new_df


# In[27]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[28]:


new_df.head()


# In[29]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[41]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[42]:


cv.get_feature_names_out()


# In[33]:


get_ipython().system('pip install nltk')


# In[34]:


import nltk


# In[35]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[36]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[37]:


ps.stem('acting')


# In[39]:


new_df['tags']=new_df['tags'].apply(stem)


# In[43]:


from sklearn.metrics.pairwise import cosine_similarity


# In[49]:


similarity=cosine_similarity(vectors)


# In[56]:


def recommend(movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[61]:


recommend("Pirates of the Caribbean: At World's End")


# In[53]:


new_df.iloc[1216].title


# In[64]:


import pickle
pickle.dump(new_df,open('movies.pkl','wb'))


# In[65]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[67]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))

