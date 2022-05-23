#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


data=pd.read_csv('TravelInsurancePrediction.csv')
data.head()


# In[3]:


data.drop(columns=["Unnamed: 0"], inplace=True)


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data["TravelInsurance"] = data["TravelInsurance"].map({0: "Not Purchased", 1: "Purchased"})


# In[8]:


sns.histplot(data,x="Age",hue="TravelInsurance",multiple="stack")
plt.title("Factors Affecting Purchase of Travel Insurance: Age")


# In[9]:


sns.histplot(data,x="Employment Type",hue="TravelInsurance",multiple="stack")
plt.title("Factors Affecting Purchase of Travel Insurance: Employment Type")


# In[10]:


sns.histplot(data,x="AnnualIncome",hue="TravelInsurance",multiple="stack")
plt.title("Factors Affecting Purchase of Travel Insurance: Annual Income")


# In[11]:


data["GraduateOrNot"] = data["GraduateOrNot"].map({"No": 0, "Yes": 1})
data["FrequentFlyer"] = data["FrequentFlyer"].map({"No": 0, "Yes": 1})
data["EverTravelledAbroad"] = data["EverTravelledAbroad"].map({"No": 0, "Yes": 1})


# In[12]:


X=np.array(data[["Age", "GraduateOrNot","AnnualIncome", 
                 "FamilyMembers", "ChronicDiseases", 
                 "FrequentFlyer", "EverTravelledAbroad"]])
y = np.array(data[["TravelInsurance"]])


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[16]:


y_pred= dtc.predict(X_test)


# In[17]:


from sklearn import metrics


# In[18]:


metrics.accuracy_score(y_test,y_pred)


# In[19]:


cm = metrics.confusion_matrix(y_test,y_pred);cm


# In[20]:


from sklearn.ensemble import RandomForestClassifier


# In[21]:


rf=RandomForestClassifier(n_estimators=300,random_state=0)


# In[22]:


rf.fit(X_train,y_train)


# In[23]:


y_pred=rf.predict(X_test)


# In[24]:


metrics.accuracy_score(y_test,y_pred)


# In[25]:


cm = metrics.confusion_matrix(y_test,y_pred);cm

