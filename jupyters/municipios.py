#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


# In[2]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# # Dataset

# In[3]:


df_municipios_2015 = pd.read_csv('../../data/bcggammachallenge/municipios/municipios20150101.csv')
df_municipios_2016 = pd.read_csv('../../data/bcggammachallenge/municipios/municipios20160101.csv')
df_municipios_2017 = pd.read_csv('../../data/bcggammachallenge/municipios/municipios20170101.csv')


# In[4]:


print(df_municipios_2015.shape)
print(df_municipios_2016.shape)
print(df_municipios_2017.shape)


# In[5]:


df = pd.concat([df_municipios_2015, df_municipios_2016, df_municipios_2017])


# In[6]:


df.shape


# In[16]:


df.head()


# In[8]:


df['regiao'].value_counts()


# In[23]:


df['cod_municipio'].unique().size


# Não há repetição de nenhum município

# In[27]:


df.columns


# In[24]:


columns = [
    'regiao',
    'unidade_federativa',
    'municipio',
    'num_escolas',
    'num_escolas_em_atividade',
    'num_professores',
    'num_estudantes',
    'num_funcionarios'    
]

df[columns].head()


# In[25]:


df[columns].describe()

