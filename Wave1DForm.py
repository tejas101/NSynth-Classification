#!/usr/bin/env python
# coding: utf-8

# In[28]:


import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
rate, data = wav.read('C:/Users/admin/Project1/Waves/bass_electronic_018-022-100.wav')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data)
plt.title('Sample Bass 1D Audio Form')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()


# In[27]:


rate,data = wav.read('C:/Users/admin/Project1/Waves/brass_acoustic_006-025-025.wav')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data)
plt.title('Sample Brass 1D Audio Form')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()


# In[13]:


rate,data = wav.read('C:/Users/admin/Project1/Waves/flute_acoustic_002-067-025.wav')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data)
plt.title('Sample Flute 1D Audio Form')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()


# In[16]:

rate,data = wav.read('C:/Users/admin/Project1/Waves/guitar_acoustic_010-021-127.wav')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data)
plt.title('Sample Guitar 1D Audio Form')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()


# In[17]:

rate,data = wav.read('C:/Users/admin/Project1/Waves/keyboard_acoustic_004-022-050.wav')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data)
plt.title('Sample Keyboard 1D Audio Form')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()


# In[19]:


rate,data = wav.read('C:/Users/admin/Project1/Waves/mallet_acoustic_047-065-025.wav')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data)
plt.title('Sample Mallet 1D Audio Form')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()


# In[21]:

rate,data = wav.read('C:/Users/admin/Project1/Waves/organ_electronic_001-036-050.wav')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data)
plt.title('Sample Organ 1D Audio Form')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()


# In[22]:


rate, data = wav.read('C:/Users/admin/Project1/Waves/reed_acoustic_011-035-127.wav')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data)
plt.title('Sample Reed 1D Audio Form')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()


# In[24]:

rate,data = wav.read('C:/Users/admin/Project1/Waves/string_acoustic_012-025-127.wav')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data)
plt.title('Sample String 1D Audio Form')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()


# In[26]:

rate,data = wav.read('C:/Users/admin/Project1/Waves/vocal_acoustic_000-050-025.wav')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(data)
plt.title('Sample Vocal 1D Audio Form')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()





