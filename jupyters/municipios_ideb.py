#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


# In[18]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# # Dataset

# In[19]:


df_municipios_2015 = pd.read_csv('../../data/bcggammachallenge/municipios/municipios20150101.csv')
df_municipios_2016 = pd.read_csv('../../data/bcggammachallenge/municipios/municipios20160101.csv')
df_municipios_2017 = pd.read_csv('../../data/bcggammachallenge/municipios/municipios20170101.csv')


# In[20]:


df = pd.concat([df_municipios_2015, df_municipios_2016, df_municipios_2017])


# In[21]:


df.head()


# In[22]:


df_ideb_ini = pd.read_csv('../../data/bcggammachallenge/ideb/ideb_municipios_anosiniciais2005_2017.csv',sep = ',',encoding='latin-1')


# In[23]:


df_ideb_ini.columns


# In[24]:


df_ideb_ini[['Cod_Municipio_Completo', 'Ideb2017']].head()


# In[25]:


df_ideb_ini = df_ideb_ini.rename(columns={'Cod_Municipio_Completo': 'cod_municipio'})

df_ideb_ini_2015 = df_ideb_ini.copy()
df_ideb_ini_2017 = df_ideb_ini.copy()


# In[26]:


df_ideb_ini_2015 = df_ideb_ini_2015[['cod_municipio', 'Ideb2015']]
df_ideb_ini_2017 = df_ideb_ini_2017[['cod_municipio', 'Ideb2017']]


# In[27]:


df_ideb_ini_2015.head()


# In[28]:


df_ideb_ini_2017.head()


# In[32]:


df_ideb_ini_2015['cod_municipio'] = df_ideb_ini_2015.cod_municipio.astype(float)
df_ideb_ini_2017['cod_municipio'] = df_ideb_ini_2017.cod_municipio.astype(float)


# In[33]:


df_result_2015 = pd.merge(df_municipios_2015, df_ideb_ini_2015, how='inner', on='cod_municipio')
df_result_2017 = pd.merge(df_municipios_2017, df_ideb_ini_2017, how='inner', on='cod_municipio')


# In[36]:


df_result_2015 = df_result_2015.rename(columns={'Ideb2015': 'ideb'})
df_result_2017 = df_result_2017.rename(columns={'Ideb2017': 'ideb'})


# In[52]:


df_result_2015.sort_values(by=['ideb'], ascending=False).head(8)


# In[59]:


df_result_2017.sort_values(by=['ideb'], ascending=False).head(8)


# In[57]:


print(df_result_2015[df_result_2015['ideb'] != '-']['ideb'].max())
print(df_result_2015[df_result_2015['ideb'] != '-']['ideb'].min())


# In[58]:


print(df_result_2017[df_result_2017['ideb'] != '-']['ideb'].max())
print(df_result_2017[df_result_2017['ideb'] != '-']['ideb'].min())


# ## Correlações linear entre todas as variáveis numéricas com o Ideb

# In[131]:


df_result_2015['ideb'] = df_result_2015['ideb'].replace('-',0)
df_result_2017['ideb'] = df_result_2017['ideb'].replace('-',0)


# In[132]:


df_result_2015['ideb'] = pd.to_numeric(df_result_2015['ideb'])
df_result_2017['ideb'] = pd.to_numeric(df_result_2017['ideb'])


# In[138]:


def calculate_pearson(df):
    correlations = {}
    numerical_features = df.select_dtypes(exclude = ["object"]).columns
    numerical_features = numerical_features.drop("cod_municipio")
    for i in numerical_features:
        corr = stats.pearsonr(df[i], df['ideb'])[0]
        correlations[i] = corr
    df_corr = pd.DataFrame(list(correlations.items()), columns=['feature', 'correlation_with_ideb'])        
    df_corr = df_corr.dropna()
    
    return df_corr


# In[141]:


df_corr_2015 = calculate_pearson(df_result_2015)
df_corr_2017 = calculate_pearson(df_result_2017)


# In[140]:


df_corr_2015.sort_values(by=['correlation_with_ideb'], ascending=False)


# In[142]:


df_corr_2017.sort_values(by=['correlation_with_ideb'], ascending=False)


# In[ ]:




