#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle


# In[2]:


df=pd.read_csv('penguins.csv')
df


# In[3]:


df.dropna(inplace=True)
df


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


df=df.drop(columns=['island','species'])
df['sex']=df['sex'].astype('category')
df['sex']=df['sex'].cat.codes


# In[7]:


for x in ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']:
    df[x]=(df[x]-df[x].min())/(df[x].max()-df[x].min())


# In[8]:


y = df['sex']
x = df.drop(columns=['sex'])


# In[9]:


np.random.seed(1200)
ratio = 0.20
total_rows = df.shape[0]
test_size = int(total_rows*ratio)


# In[10]:


x_test = x[0:test_size]
x_train = x[test_size:]
y_test = y[0:test_size]
y_train = y[test_size:]


# In[11]:


x_train.shape


# In[12]:


x_test.shape


# In[13]:


y_train.shape


# In[14]:


y_test.shape


# In[24]:


class LogitRegression:
   
    def __init__(self,learning_rate,iterations):
        self.learning_rate=learning_rate
        self.iterations=iterations
    
    def sigmoid(self,z):
        S=1/(1+np.exp(-z))
        return S
    
    def cost(self,x,y):
        c=np.dot(x,self.weights) 
        h=self.sigmoid(c)
        return np.mean(-y*np.log(h)-(1-y)*np.log(1-h))
    
    def gradient_descent(self,x,y):
        z=np.dot(x,self.weights)+self.bias
        predict=self.sigmoid(z)
        d=predict-y
        d_weights=(1/len(y))*np.dot(x.transpose(),d)
        d_bias=(1/len(y))*np.sum(d)
        return d_weights,d_bias
    
    def fit(self,x,y):
        self.loss=[]
        self.weights=np.random.uniform(0,1,x.shape[1])
        self.bias=0
        for i in range(0,self.iterations):
            dw,db=self.gradient_descent(x, y) 
            self.bias-=self.learning_rate*db
            self.weights-=self.learning_rate*dw
            self.loss.append(self.cost(x,y))     
        return self.weights,self.loss
    
    def predict(self,x):
        z=np.dot(x,self.weights)+self.bias
        h=1/(1+np.exp(-z))
        y_pred=np.where(h >= 0.5, 1, 0)
        return y_pred


# In[25]:


model=LogitRegression(1e-3,100000)
weights,loss=model.fit(x_train, y_train)


# In[28]:


y_pred=model.predict(x_test)
accuracy=np.mean(y_pred == y_test)
print("Accuracy:", accuracy)


# In[27]:


with open('penguins_model.pkl', 'wb') as f:
    pickle.dump(weights, f)

