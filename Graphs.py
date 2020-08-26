#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[22]:


train_dir='C:/Users/admin/Project1/trainexamples.json'
valid_dir='C:/Users/admin/Project1/validexamples.json'
test_dir='C:/Users/admin/Project1/testexamples.json'


# In[23]:


df_train= pd.read_json(train_dir, orient='index')
df_valid= pd.read_json(valid_dir, orient='index')
df_test= pd.read_json(test_dir, orient='index')


# In[29]:


df_train['instrument_family'].value_counts().reindex(np.arange(0,11, 1)).plot(kind='bar', figsize=(5,3))
plt.title("Instrument Family Distribution for Training Set")
plt.xlabel("Instrument Family")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("C:/Users/admin/Project1/trainexamples.png")
plt.show()


# In[30]:


df_valid['instrument_family'].value_counts().reindex(np.arange(0,11, 1)).plot(kind ='bar', figsize=(5,3))
plt.title("Instrument Family Distribution for Validation Set")
plt.xlabel("Instrument Family")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("C:/Users/admin/Project1/validexamples.png")
plt.show()


# In[20]:


df_test['instrument_family'].value_counts().reindex(np.arange(0,11, 1)).plot(kind ='bar', figsize=(5,3))
plt.title("Instrument Family Distribution for Test Set")
plt.xlabel("Instrument Family")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("C:/Users/admin/Project1/testexamples.png")
plt.show()


# In[ ]:




