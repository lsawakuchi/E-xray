#!/usr/bin/env python
# coding: utf-8

# - três variaveis (Regioes, municipios, escolas), 
# - cada variavel foi separado em iniciais e finais
# - separei por ano, cada dado
# 
# Alguns anos, os valore de Ibed estavam com '-', retirei os valores.
# 
# Os dataframes estão no item SEPARAR POR DADOS
# Plotei todos os gráficos, olha a primeira coluna que é o grafico do Ibed em relação as variaveis existente.
# 
# Retirei um outlier que tinha taxa de aprovação igual a zero.

# In[1]:


import pandas as pd
import chardet
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
import numpy as np


# In[2]:


#!pip install chardet


# In[3]:


#!pip install --upgrade pip


# In[4]:


regioes_anosfinais2005_2017 = pd.read_csv('../../data/bcggammachallenge/ideb/ideb_uf_regioes_anosfinais2005_2017.csv',sep = ',',encoding='latin-1')
regioes_anosiniciais2005_2017 = pd.read_csv('../../data/bcggammachallenge/ideb/ideb_uf_regioes_anosiniciais2005_2017.csv',sep = ',',encoding='latin-1')

municipios_anosfinais2005_2017 = pd.read_csv('../../data/bcggammachallenge/ideb/ideb_municipios_anosfinais2005_2017.csv',sep = ',',encoding='latin-1')
municipios_anosiniciais2005_2017 = pd.read_csv('../../data/bcggammachallenge/ideb/ideb_municipios_anosiniciais2005_2017.csv',sep = ',',encoding='latin-1')

escolas_anosiniciais2005_2017 = pd.read_csv('../../data/bcggammachallenge/ideb/ideb_escolas_anosiniciais2005_2017.csv',sep = ',',encoding='latin-1')
escolas_anosfinais2005_2017 = pd.read_csv('../../data/bcggammachallenge/ideb/ideb_escolas_anosfinais2005_2017.csv',sep = ',',encoding='latin-1')


# In[5]:


municipios_anosiniciais2005_2017.head()


# In[6]:


municipios_anosfinais2005_2017.head()


# In[7]:


print(regioes_anosfinais2005_2017.shape)
print(regioes_anosiniciais2005_2017.shape)
print(municipios_anosfinais2005_2017.shape)
print(municipios_anosiniciais2005_2017.shape)
print(escolas_anosiniciais2005_2017.shape)
print(escolas_anosfinais2005_2017.shape)


# # Missing Datas

# In[8]:


import seaborn as sns


# In[9]:


total = regioes_anosfinais2005_2017.isnull().sum().sort_values(ascending=False)
percent = (regioes_anosfinais2005_2017.isnull().sum()/regioes_anosfinais2005_2017.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[10]:


total = regioes_anosiniciais2005_2017.isnull().sum().sort_values(ascending=False)
percent = (regioes_anosiniciais2005_2017.isnull().sum()/regioes_anosiniciais2005_2017.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[11]:


total = municipios_anosfinais2005_2017.isnull().sum().sort_values(ascending=False)
percent = (municipios_anosfinais2005_2017.isnull().sum()/municipios_anosfinais2005_2017.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[12]:


total = municipios_anosiniciais2005_2017.isnull().sum().sort_values(ascending=False)
percent = (municipios_anosiniciais2005_2017.isnull().sum()/municipios_anosiniciais2005_2017.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[13]:


total = escolas_anosiniciais2005_2017.isnull().sum().sort_values(ascending=False)
percent = (escolas_anosiniciais2005_2017.isnull().sum()/escolas_anosiniciais2005_2017.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[14]:


total = escolas_anosfinais2005_2017.isnull().sum().sort_values(ascending=False)
percent = (escolas_anosfinais2005_2017.isnull().sum()/escolas_anosfinais2005_2017.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# # Separar os dados

# In[15]:


dados2005_inic = pd.DataFrame(regioes_anosiniciais2005_2017.iloc[:,0:2])
dados2005_inic = pd.concat([dados2005_inic,regioes_anosiniciais2005_2017.filter(like='2005')], axis=1)
dados2005_inic = dados2005_inic.rename(columns={"TaxaAprovacao2005_1ao5ano": "TaxaAprovacao_1ao5ano",
                                     "TaxaAprovacao2005_1ano": "TaxaAprovacao_1ano",
                                     "TaxaAprovacao2005_2ano": "TaxaAprovacao_2ano",
                                     "TaxaAprovacao2005_3ano": "TaxaAprovacao_3ano",
                                     "TaxaAprovacao2005_4ano": "TaxaAprovacao_4ano",
                                     "TaxaAprovacao2005_5ano": "TaxaAprovacao_5ano",
                                     "IndicadorRendimento_2005": "IndicadorRendimento_inicial",
                                     "NotaProvaBrasil_MT_2005": "NotaProvaBrasil_MT_inicial",
                                     "NotaProvaBrasil_LP_2005": "NotaProvaBrasil_LP_inicial",
                                     "NotaProvaBrasil_NotaMedia_2005": "NotaProvaBrasil_NotaMedia_inicial",
                                     "Ideb2005": "Ideb",
                                     "ProjecaoIdeb2005": "ProjecaoIdeb_inicial"})
dados2005_inic.head()


# In[16]:


#scatterplot
categorical_features = dados2005_inic.select_dtypes(include = ["object"]).columns
numerical_features = dados2005_inic.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2005_inic_num = dados2005_inic[numerical_features]
dados2005_inic_cat = dados2005_inic[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2005_inic[cols], size = 3.5)
plt.show();


# In[17]:


dados2005_fim = pd.DataFrame(regioes_anosfinais2005_2017.iloc[:,0:2])
dados2005_fim = pd.concat([dados2005_fim,regioes_anosfinais2005_2017.filter(like='2005')], axis=1)
dados2005_fim = dados2005_fim.rename(columns={"TaxaAprovacao2005_6ao9ano": "TaxaAprovacao_6ao9ano",
                                     "TaxaAprovacao2005_6ano": "TaxaAprovacao_6ano",
                                     "TaxaAprovacao2005_7ano": "TaxaAprovacao_7ano",
                                     "TaxaAprovacao2005_8ano": "TaxaAprovacao_8ano",
                                     "TaxaAprovacao2005_9ano": "TaxaAprovacao_9ano",
                                     "IndicadorRendimento_2005": "IndicadorRendimento_finais",
                                     "NotaProvaBrasil_MT_2005": "NotaProvaBrasil_MT_finais",
                                     "NotaProvaBrasil_LP_2005": "NotaProvaBrasil_LP_finais",
                                     "NotaProvaBrasil_NotaMedia_2005": "NotaProvaBrasil_NotaMedia_finais",
                                     "Ideb2005": "Ideb",
                                     "ProjecaoIdeb2005": "ProjecaoIdeb_finais"})
dados2005_fim.head()


# In[18]:


#scatterplot
categorical_features = dados2005_fim.select_dtypes(include = ["object"]).columns
numerical_features = dados2005_fim.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2005_fim_num = dados2005_fim[numerical_features]
dados2005_fim_cat = dados2005_fim[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2005_fim[cols], size = 3.5)
plt.show();


# # Ano 2007

# In[19]:


dados2007_inic = pd.DataFrame(regioes_anosiniciais2005_2017.iloc[:,0:2])
dados2007_inic = pd.concat([dados2007_inic,regioes_anosiniciais2005_2017.filter(like='2007')], axis=1)
dados2007_inic = dados2007_inic.rename(columns={"TaxaAprovacao2007_1ao5ano": "TaxaAprovacao_1ao5ano",
                                     "TaxaAprovacao2007_1ano": "TaxaAprovacao_1ano",
                                     "TaxaAprovacao2007_2ano": "TaxaAprovacao_2ano",
                                     "TaxaAprovacao2007_3ano": "TaxaAprovacao_3ano",
                                     "TaxaAprovacao2007_4ano": "TaxaAprovacao_4ano",
                                     "TaxaAprovacao2007_5ano": "TaxaAprovacao_5ano",
                                     "IndicadorRendimento_2007": "IndicadorRendimento_inicial",
                                     "NotaProvaBrasil_MT_2007": "NotaProvaBrasil_MT_inicial",
                                     "NotaProvaBrasil_LP_2007": "NotaProvaBrasil_LP_inicial",
                                     "NotaProvaBrasil_NotaMedia_2007": "NotaProvaBrasil_NotaMedia_inicial",
                                     "Ideb2007": "Ideb",
                                     "ProjecaoIdeb2007": "ProjecaoIdeb_inicial"})
dados2007_inic.head()


# In[20]:


#scatterplot
categorical_features = dados2007_inic.select_dtypes(include = ["object"]).columns
numerical_features = dados2007_inic.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2007_inic_num = dados2007_inic[numerical_features]
dados2007_inic_cat = dados2007_inic[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2007_inic[cols], size = 3.5)
plt.show();


# In[21]:


dados2007_fim = pd.DataFrame(regioes_anosfinais2005_2017.iloc[:,0:2])
dados2007_fim = pd.concat([dados2007_fim,regioes_anosfinais2005_2017.filter(like='2007')], axis=1)
dados2007_fim = dados2007_fim.rename(columns={"TaxaAprovacao2007_6ao9ano": "TaxaAprovacao_6ao9ano",
                                     "TaxaAprovacao2007_6ano": "TaxaAprovacao_6ano",
                                     "TaxaAprovacao2007_7ano": "TaxaAprovacao_7ano",
                                     "TaxaAprovacao2007_8ano": "TaxaAprovacao_8ano",
                                     "TaxaAprovacao2007_9ano": "TaxaAprovacao_9ano",
                                     "IndicadorRendimento_2007": "IndicadorRendimento_finais",
                                     "NotaProvaBrasil_MT_2007": "NotaProvaBrasil_MT_finais",
                                     "NotaProvaBrasil_LP_2007": "NotaProvaBrasil_LP_finais",
                                     "NotaProvaBrasil_NotaMedia_2007": "NotaProvaBrasil_NotaMedia_finais",
                                     "Ideb2007": "Ideb",
                                     "ProjecaoIdeb2007": "ProjecaoIdeb_finais"})
dados2007_fim.head()


# In[22]:


#scatterplot
categorical_features = dados2007_fim.select_dtypes(include = ["object"]).columns
numerical_features = dados2007_fim.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2007_fim_num = dados2007_fim[numerical_features]
dados2007_fim_cat = dados2007_fim[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2007_fim[cols], size = 3.5)
plt.show();


# # Ano 2009

# In[23]:


dados2009_inic = pd.DataFrame(regioes_anosiniciais2005_2017.iloc[:,0:2])
dados2009_inic = pd.concat([dados2009_inic,regioes_anosiniciais2005_2017.filter(like='2009')], axis=1)
dados2009_inic = dados2009_inic.rename(columns={"TaxaAprovacao2009_1ao5ano": "TaxaAprovacao_1ao5ano",
                                     "TaxaAprovacao2009_1ano": "TaxaAprovacao_1ano",
                                     "TaxaAprovacao2009_2ano": "TaxaAprovacao_2ano",
                                     "TaxaAprovacao2009_3ano": "TaxaAprovacao_3ano",
                                     "TaxaAprovacao2009_4ano": "TaxaAprovacao_4ano",
                                     "TaxaAprovacao2009_5ano": "TaxaAprovacao_5ano",
                                     "IndicadorRendimento_2009": "IndicadorRendimento_inicial",
                                     "NotaProvaBrasil_MT_2009": "NotaProvaBrasil_MT_inicial",
                                     "NotaProvaBrasil_LP_2009": "NotaProvaBrasil_LP_inicial",
                                     "NotaProvaBrasil_NotaMedia_2009": "NotaProvaBrasil_NotaMedia_inicial",
                                     "Ideb2009": "Ideb",
                                     "ProjecaoIdeb2009": "ProjecaoIdeb_inicial"})
dados2009_inic.head()


# In[24]:


dados2009_inic = dados2009_inic[dados2009_inic['Ideb']!= '-']
dados2009_inic = dados2009_inic.reset_index()
dados2009_inic['Ideb'] = pd.to_numeric(dados2009_inic['Ideb'])


# In[25]:


# Retirar outlier, TaxaAprovacao_1ano = 0
dados2009_inic = dados2009_inic.drop(dados2009_inic[dados2009_inic['TaxaAprovacao_1ano']==0].index[0])


# In[26]:


#scatterplot
categorical_features = dados2009_inic.select_dtypes(include = ["object"]).columns
numerical_features = dados2009_inic.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2009_inic_num = dados2009_inic[numerical_features]
dados2009_inic_cat = dados2009_inic[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2009_inic[cols], size = 3.5)
plt.show();


# In[27]:


dados2009_fim = pd.DataFrame(regioes_anosfinais2005_2017.iloc[:,0:2])
dados2009_fim = pd.concat([dados2009_fim,regioes_anosfinais2005_2017.filter(like='2009')], axis=1)
dados2009_fim = dados2009_fim.rename(columns={"TaxaAprovacao2009_6ao9ano": "TaxaAprovacao_6ao9ano",
                                     "TaxaAprovacao2009_6ano": "TaxaAprovacao_6ano",
                                     "TaxaAprovacao2009_7ano": "TaxaAprovacao_7ano",
                                     "TaxaAprovacao2009_8ano": "TaxaAprovacao_8ano",
                                     "TaxaAprovacao2009_9ano": "TaxaAprovacao_9ano",
                                     "IndicadorRendimento_2009": "IndicadorRendimento_finais",
                                     "NotaProvaBrasil_MT_2009": "NotaProvaBrasil_MT_finais",
                                     "NotaProvaBrasil_LP_2009": "NotaProvaBrasil_LP_finais",
                                     "NotaProvaBrasil_NotaMedia_2009": "NotaProvaBrasil_NotaMedia_finais",
                                     "Ideb2009": "Ideb",
                                     "ProjecaoIdeb2009": "ProjecaoIdeb_finais"})
dados2009_fim.head()


# In[28]:


dados2009_fim = dados2009_fim[dados2009_fim['Ideb']!= '-']
dados2009_fim = dados2009_fim.reset_index()
dados2009_fim['Ideb'] = pd.to_numeric(dados2009_fim['Ideb'])


# In[29]:


#scatterplot
categorical_features = dados2009_fim.select_dtypes(include = ["object"]).columns
numerical_features = dados2009_fim.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2009_fim_num = dados2009_fim[numerical_features]
dados2009_fim_cat = dados2009_fim[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2009_fim[cols], size = 3.5)
plt.show();


# # Ano 2011

# In[30]:


dados2011_inic = pd.DataFrame(regioes_anosiniciais2005_2017.iloc[:,0:2])
dados2011_inic = pd.concat([dados2011_inic,regioes_anosiniciais2005_2017.filter(like='2011')], axis=1)
dados2011_inic = dados2011_inic.rename(columns={"TaxaAprovacao2011_1ao5ano": "TaxaAprovacao_1ao5ano",
                                     "TaxaAprovacao2011_1ano": "TaxaAprovacao_1ano",
                                     "TaxaAprovacao2011_2ano": "TaxaAprovacao_2ano",
                                     "TaxaAprovacao2011_3ano": "TaxaAprovacao_3ano",
                                     "TaxaAprovacao2011_4ano": "TaxaAprovacao_4ano",
                                     "TaxaAprovacao2011_5ano": "TaxaAprovacao_5ano",
                                     "IndicadorRendimento_2011": "IndicadorRendimento_inicial",
                                     "NotaProvaBrasil_MT_2011": "NotaProvaBrasil_MT_inicial",
                                     "NotaProvaBrasil_LP_2011": "NotaProvaBrasil_LP_inicial",
                                     "NotaProvaBrasil_NotaMedia_2011": "NotaProvaBrasil_NotaMedia_inicial",
                                     "Ideb2011": "Ideb",
                                     "ProjecaoIdeb2011": "ProjecaoIdeb_inicial"})
dados2011_inic.head()


# In[31]:


#scatterplot
categorical_features = dados2011_inic.select_dtypes(include = ["object"]).columns
numerical_features = dados2011_inic.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2011_inic_num = dados2011_inic[numerical_features]
dados2011_inic_cat = dados2011_inic[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2011_inic[cols], size = 3.5)
plt.show();


# In[32]:


dados2011_fim = pd.DataFrame(regioes_anosfinais2005_2017.iloc[:,0:2])
dados2011_fim = pd.concat([dados2011_fim,regioes_anosfinais2005_2017.filter(like='2011')], axis=1)
dados2011_fim = dados2011_fim.rename(columns={"TaxaAprovacao2011_6ao9ano": "TaxaAprovacao_6ao9ano",
                                     "TaxaAprovacao2011_6ano": "TaxaAprovacao_6ano",
                                     "TaxaAprovacao2011_7ano": "TaxaAprovacao_7ano",
                                     "TaxaAprovacao2011_8ano": "TaxaAprovacao_8ano",
                                     "TaxaAprovacao2011_9ano": "TaxaAprovacao_9ano",
                                     "IndicadorRendimento_2011": "IndicadorRendimento_finais",
                                     "NotaProvaBrasil_MT_2011": "NotaProvaBrasil_MT_finais",
                                     "NotaProvaBrasil_LP_2011": "NotaProvaBrasil_LP_finais",
                                     "NotaProvaBrasil_NotaMedia_2011": "NotaProvaBrasil_NotaMedia_finais",
                                     "Ideb2011": "Ideb",
                                     "ProjecaoIdeb2011": "ProjecaoIdeb_finais"})
dados2011_fim.head()


# In[33]:


#scatterplot
categorical_features = dados2011_fim.select_dtypes(include = ["object"]).columns
numerical_features = dados2011_fim.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2011_fim_num = dados2011_fim[numerical_features]
dados2011_fim_cat = dados2011_fim[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2011_fim[cols], size = 3.5)
plt.show();


# # Ano 2013

# In[34]:


dados2013_inic = pd.DataFrame(regioes_anosiniciais2005_2017.iloc[:,0:2])
dados2013_inic = pd.concat([dados2013_inic,regioes_anosiniciais2005_2017.filter(like='2013')], axis=1)
dados2013_inic = dados2013_inic.rename(columns={"TaxaAprovacao2013_1ao5ano": "TaxaAprovacao_1ao5ano",
                                     "TaxaAprovacao2013_1ano": "TaxaAprovacao_1ano",
                                     "TaxaAprovacao2013_2ano": "TaxaAprovacao_2ano",
                                     "TaxaAprovacao2013_3ano": "TaxaAprovacao_3ano",
                                     "TaxaAprovacao2013_4ano": "TaxaAprovacao_4ano",
                                     "TaxaAprovacao2013_5ano": "TaxaAprovacao_5ano",
                                     "IndicadorRendimento_2013": "IndicadorRendimento_inicial",
                                     "NotaProvaBrasil_MT_2013": "NotaProvaBrasil_MT_inicial",
                                     "NotaProvaBrasil_LP_2013": "NotaProvaBrasil_LP_inicial",
                                     "NotaProvaBrasil_NotaMedia_2013": "NotaProvaBrasil_NotaMedia_inicial",
                                     "Ideb2013": "Ideb",
                                     "ProjecaoIdeb2013": "ProjecaoIdeb_inicial"})
dados2013_inic.head()


# In[35]:


#scatterplot
categorical_features = dados2013_inic.select_dtypes(include = ["object"]).columns
numerical_features = dados2013_inic.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2013_inic_num = dados2013_inic[numerical_features]
dados2013_inic_cat = dados2013_inic[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2013_inic[cols], size = 3.5)
plt.show();


# In[36]:


dados2013_fim = pd.DataFrame(regioes_anosfinais2005_2017.iloc[:,0:2])
dados2013_fim = pd.concat([dados2013_fim,regioes_anosfinais2005_2017.filter(like='2013')], axis=1)
dados2013_fim = dados2013_fim.rename(columns={"TaxaAprovacao2013_6ao9ano": "TaxaAprovacao_6ao9ano",
                                     "TaxaAprovacao2013_6ano": "TaxaAprovacao_6ano",
                                     "TaxaAprovacao2013_7ano": "TaxaAprovacao_7ano",
                                     "TaxaAprovacao2013_8ano": "TaxaAprovacao_8ano",
                                     "TaxaAprovacao2013_9ano": "TaxaAprovacao_9ano",
                                     "IndicadorRendimento_2013": "IndicadorRendimento_finais",
                                     "NotaProvaBrasil_MT_2013": "NotaProvaBrasil_MT_finais",
                                     "NotaProvaBrasil_LP_2013": "NotaProvaBrasil_LP_finais",
                                     "NotaProvaBrasil_NotaMedia_2013": "NotaProvaBrasil_NotaMedia_finais",
                                     "Ideb2013": "Ideb",
                                     "ProjecaoIdeb2013": "ProjecaoIdeb_finais"})
dados2013_fim.head()


# In[37]:


#scatterplot
categorical_features = dados2013_fim.select_dtypes(include = ["object"]).columns
numerical_features = dados2013_fim.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2013_fim_num = dados2013_fim[numerical_features]
dados2013_fim_cat = dados2013_fim[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2013_fim[cols], size = 3.5)
plt.show();


# # Ano 2015

# In[38]:


dados2015_inic = pd.DataFrame(regioes_anosiniciais2005_2017.iloc[:,0:2])
dados2015_inic = pd.concat([dados2015_inic,regioes_anosiniciais2005_2017.filter(like='2015')], axis=1)
dados2015_inic = dados2015_inic.rename(columns={"TaxaAprovacao2015_1ao5ano": "TaxaAprovacao_1ao5ano",
                                     "TaxaAprovacao2015_1ano": "TaxaAprovacao_1ano",
                                     "TaxaAprovacao2015_2ano": "TaxaAprovacao_2ano",
                                     "TaxaAprovacao2015_3ano": "TaxaAprovacao_3ano",
                                     "TaxaAprovacao2015_4ano": "TaxaAprovacao_4ano",
                                     "TaxaAprovacao2015_5ano": "TaxaAprovacao_5ano",
                                     "IndicadorRendimento_2015": "IndicadorRendimento_inicial",
                                     "NotaProvaBrasil_MT_2015": "NotaProvaBrasil_MT_inicial",
                                     "NotaProvaBrasil_LP_2015": "NotaProvaBrasil_LP_inicial",
                                     "NotaProvaBrasil_NotaMedia_2015": "NotaProvaBrasil_NotaMedia_inicial",
                                     "Ideb2015": "Ideb",
                                     "ProjecaoIdeb2015": "ProjecaoIdeb_inicial"})
dados2015_inic.head()


# In[39]:


#scatterplot
categorical_features = dados2015_inic.select_dtypes(include = ["object"]).columns
numerical_features = dados2015_inic.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2015_inic_num = dados2015_inic[numerical_features]
dados2015_inic_cat = dados2015_inic[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2015_inic[cols], size = 3.5)
plt.show();


# In[40]:


dados2015_fim = pd.DataFrame(regioes_anosfinais2005_2017.iloc[:,0:2])
dados2015_fim = pd.concat([dados2015_fim,regioes_anosfinais2005_2017.filter(like='2015')], axis=1)
dados2015_fim = dados2015_fim.rename(columns={"TaxaAprovacao2015_6ao9ano": "TaxaAprovacao_6ao9ano",
                                     "TaxaAprovacao2015_6ano": "TaxaAprovacao_6ano",
                                     "TaxaAprovacao2015_7ano": "TaxaAprovacao_7ano",
                                     "TaxaAprovacao2015_8ano": "TaxaAprovacao_8ano",
                                     "TaxaAprovacao2015_9ano": "TaxaAprovacao_9ano",
                                     "IndicadorRendimento_2015": "IndicadorRendimento_finais",
                                     "NotaProvaBrasil_MT_2015": "NotaProvaBrasil_MT_finais",
                                     "NotaProvaBrasil_LP_2015": "NotaProvaBrasil_LP_finais",
                                     "NotaProvaBrasil_NotaMedia_2015": "NotaProvaBrasil_NotaMedia_finais",
                                     "Ideb2015": "Ideb",
                                     "ProjecaoIdeb2015": "ProjecaoIdeb_finais"})
dados2015_fim.head()


# In[41]:


#scatterplot
categorical_features = dados2015_fim.select_dtypes(include = ["object"]).columns
numerical_features = dados2015_fim.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2015_fim_num = dados2015_fim[numerical_features]
dados2015_fim_cat = dados2015_fim[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2015_fim[cols], size = 3.5)
plt.show();


# # Ano 2017

# In[42]:


dados2017_inic = pd.DataFrame(regioes_anosiniciais2005_2017.iloc[:,0:2])
dados2017_inic = pd.concat([dados2017_inic,regioes_anosiniciais2005_2017.filter(like='2017')], axis=1)
dados2017_inic = dados2017_inic.rename(columns={"TaxaAprovacao2017_1ao5ano": "TaxaAprovacao_1ao5ano",
                                     "TaxaAprovacao2017_1ano": "TaxaAprovacao_1ano",
                                     "TaxaAprovacao2017_2ano": "TaxaAprovacao_2ano",
                                     "TaxaAprovacao2017_3ano": "TaxaAprovacao_3ano",
                                     "TaxaAprovacao2017_4ano": "TaxaAprovacao_4ano",
                                     "TaxaAprovacao2017_5ano": "TaxaAprovacao_5ano",
                                     "IndicadorRendimento_2017": "IndicadorRendimento_inicial",
                                     "NotaProvaBrasil_MT_2017": "NotaProvaBrasil_MT_inicial",
                                     "NotaProvaBrasil_LP_2017": "NotaProvaBrasil_LP_inicial",
                                     "NotaProvaBrasil_NotaMedia_2017": "NotaProvaBrasil_NotaMedia_inicial",
                                     "Ideb2017": "Ideb",
                                     "ProjecaoIdeb2017": "ProjecaoIdeb_inicial"})
dados2017_inic.head()


# In[43]:


dados2017_inic = dados2017_inic[dados2017_inic['Ideb']!= '-']
dados2017_inic = dados2017_inic.reset_index()
dados2017_inic['Ideb'] = pd.to_numeric(dados2017_inic['Ideb'])


# In[44]:


#scatterplot
categorical_features = dados2017_inic.select_dtypes(include = ["object"]).columns
numerical_features = dados2017_inic.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2017_inic_num = dados2017_inic[numerical_features]
dados2017_inic_cat = dados2017_inic[categorical_features]
sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2017_inic[cols], size = 3.5)
plt.show();


# In[45]:


dados2017_fim = pd.DataFrame(regioes_anosfinais2005_2017.iloc[:,0:2])
dados2017_fim = pd.concat([dados2017_fim,regioes_anosfinais2005_2017.filter(like='2017')], axis=1)
dados2017_fim = dados2017_fim.rename(columns={"TaxaAprovacao2017_6ao9ano": "TaxaAprovacao_6ao9ano",
                                     "TaxaAprovacao2017_6ano": "TaxaAprovacao_6ano",
                                     "TaxaAprovacao2017_7ano": "TaxaAprovacao_7ano",
                                     "TaxaAprovacao2017_8ano": "TaxaAprovacao_8ano",
                                     "TaxaAprovacao2017_9ano": "TaxaAprovacao_9ano",
                                     "IndicadorRendimento_2017": "IndicadorRendimento_finais",
                                     "NotaProvaBrasil_MT_2017": "NotaProvaBrasil_MT_finais",
                                     "NotaProvaBrasil_LP_2017": "NotaProvaBrasil_LP_finais",
                                     "NotaProvaBrasil_NotaMedia_2017": "NotaProvaBrasil_NotaMedia_finais",
                                     "Ideb2017": "Ideb",
                                     "ProjecaoIdeb2017": "ProjecaoIdeb_finais"})
dados2017_fim.head()


# In[46]:


#scatterplot
categorical_features = dados2017_fim.select_dtypes(include = ["object"]).columns
numerical_features = dados2017_fim.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("Ideb")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dados2017_fim_num = dados2017_fim[numerical_features]
dados2017_fim_cat = dados2017_fim[categorical_features]

sns.set()
cols = ['Ideb']
cols.extend(numerical_features[1:5].tolist())
sns.pairplot(dados2017_fim[cols], size = 3.5)
plt.show();


# In[ ]:





# In[47]:


#box plot overallqual/saleprice
var = categorical_features[0]
data = pd.concat([dados2017_fim['Ideb'], dados2017_fim_cat[var]], axis=1)
f, ax = plt.subplots(figsize=(10, 10))

fig = sns.boxplot(x="Ideb", y=var, data=data)
fig.axis(ymin=0, ymax=10);


# In[48]:


#box plot overallqual/saleprice
var = categorical_features[1]
data = pd.concat([dados2017_fim['Ideb'], dados2017_fim_cat[var]], axis=1)
f, ax = plt.subplots(figsize=(10, 10))

fig = sns.boxplot(x=var, y="Ideb", data=data)
fig.axis(ymin=0, ymax=10);


# In[49]:


data


# In[ ]:





# In[50]:


numerical_features = dados2017_fim.select_dtypes(exclude = ["object"]).columns
dados2017_fim_num = dados2017_fim[numerical_features]
corrmat = dados2017_fim_num.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmin=-1,vmax=1, square=True,center=0)


# In[51]:


k = 7 #number of variables for heatmap

cols = corrmat.nlargest(k, 'Ideb')['Ideb'].index
cm = np.corrcoef(dados2017_fim_num[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# # Salvar

# In[52]:


path = '../../data/bases_william'

dados2005_inic.to_csv(path + '/ano_inicial/dados2005_inic.csv')
dados2005_fim.to_csv(path + '/ano_final/dados2005_fim.csv')

dados2007_inic.to_csv(path + '/ano_inicial/dados2007_inic.csv')
dados2007_fim.to_csv(path + '/ano_final/dados2007_fim.csv')

dados2009_inic.to_csv(path + '/ano_inicial/dados2009_inic.csv')
dados2009_fim.to_csv(path + '/ano_final/dados2009_fim.csv')

dados2011_inic.to_csv(path + '/ano_inicial/dados2011_inic.csv')
dados2011_fim.to_csv(path + '/ano_final/dados2011_fim.csv')

dados2013_inic.to_csv(path + '/ano_inicial/dados2013_inic.csv')
dados2013_fim.to_csv(path + '/ano_final/dados2013_fim.csv')

dados2015_inic.to_csv(path + '/ano_inicial/dados2015_inic.csv')
dados2015_fim.to_csv(path + '/ano_final/dados2015_fim.csv')

dados2017_inic.to_csv(path + '/ano_inicial/dados2017_inic.csv')
dados2017_fim.to_csv(path + '/ano_final/dados2017_fim.csv')

