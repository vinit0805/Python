#!/usr/bin/env python
# coding: utf-8

# # HOUSE PRICE PREDICTION IN INDIA

# ## Load necessary libraries 

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Load Data  

# In[2]:


test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")


# In[3]:


train_df.head()


# In[4]:


train_df.shape


# ### Checking for missing values 

# In[5]:


train_df.isnull().sum()


# In[6]:


train_df.info()


# #### As we see the column of the address contains two parts. Let's divide this value and add a column of cities in the dataset.

# In[7]:


train_df['ADDRESS_PART1'] = train_df['ADDRESS'].apply(lambda x: x.split(',')[0].strip())
train_df['CITY'] = train_df['ADDRESS'].apply(lambda x: x.split(',')[1].strip())


# In[8]:


train_df.head()


# In[9]:


len(train_df['CITY'].unique()) 


# ### Check the correlation of columns 

# In[10]:


train_df.corr()


# In[11]:


plt.figure(figsize = (10,8))
sns.heatmap(train_df.corr())


# ###  max value of correlation between price and square of flat

# In[12]:


# Visualize the outliers by box plot
plt.figure(figsize= (10,8))
sns.boxplot(y = 'SQUARE_FT',data = train_df)


# In[13]:


def get_outliers(df, column_name):
    
    IQR = df[column_name].quantile(0.75) - df[column_name].quantile(0.25)
    lower_sq_limit = df[column_name].quantile(0.25) - (IQR * 1.5)
    upper_sq_limit = df[column_name].quantile(0.75) + (IQR * 1.5)
    outliers = np.where(df[column_name] > upper_sq_limit, True,
    np.where(df[column_name] < lower_sq_limit, True, False))
    return outliers


# In[14]:


sqr_ft_outliers = get_outliers(train_df, 'SQUARE_FT')
df_without_outliers = train_df.loc[~(sqr_ft_outliers),]
print(train_df.shape,df_without_outliers.shape)
        


# In[15]:


print("{} rows was been deleted".format(
    train_df.shape[0] - df_without_outliers.shape[0]))


# In[16]:


plt.figure(figsize=(10,8))
sns.boxplot(y='SQUARE_FT', data=df_without_outliers)


# ### Check in outliers in target column 

# In[17]:


plt.figure(figsize=(10,8))
sns.boxplot(y='TARGET(PRICE_IN_LACS)', data=df_without_outliers)


# In[18]:


price_outliers = get_outliers(df_without_outliers, 'TARGET(PRICE_IN_LACS)')
len(price_outliers)


# In[19]:


new_df = df_without_outliers.iloc[~(price_outliers),]
print(train_df.shape, new_df.shape)


# In[20]:


plt.figure(figsize=(10,8))
sns.boxplot(y='TARGET(PRICE_IN_LACS)',data = new_df)


# In[21]:


new_df.head()


# In[22]:


print("{} rows was been deleted".format(train_df.shape[0] - new_df.shape[0]))


# In[23]:


new_df.index = np.arange(new_df.shape[0])
new_df.index


# ### let's look to the city column 

# In[24]:


head_values = new_df['CITY'].value_counts().head(20).index.to_list()
head_city = new_df[new_df['CITY'].isin(head_values)]
plt.figure(figsize = (10,8))
sns.boxplot(y = 'TARGET(PRICE_IN_LACS)', x = 'CITY', data= head_city)
plt.xticks(rotation=45)


# ### We can see that CITY is an important feature for predicting model 

# In[25]:


new_df.nunique()


# In[26]:


new_df = pd.concat([new_df,pd.get_dummies(new_df['POSTED_BY'])],axis = 1)


# In[27]:


new_df = pd.concat([new_df,pd.get_dummies(new_df['BHK_OR_RK'])],axis = 1)


# In[28]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[29]:


le = LabelEncoder()
le.fit(new_df['CITY'])
new_df['NEW_CITY'] = le.transform(new_df['CITY'])


# In[30]:


le.fit(new_df['ADDRESS_PART1'])
new_df['NEW_ADDRESS_PART1'] = le.transform(new_df['ADDRESS_PART1'])


# In[31]:


new_df.head()


# In[32]:


new_df.drop(['POSTED_BY','BHK_OR_RK','ADDRESS','CITY','ADDRESS_PART1'],axis = 1 , inplace= True)


# In[33]:


temp =  new_df[['SQUARE_FT','LONGITUDE','LATITUDE','TARGET(PRICE_IN_LACS)']]
scaler = StandardScaler()
scaler.fit(temp)
temp_scaled = scaler.transform(temp)
temp_scaled = pd.DataFrame(temp_scaled,columns = temp.columns)
temp_scaled


# In[34]:


new_df.drop(['SQUARE_FT','LONGITUDE', 'LATITUDE', 'TARGET(PRICE_IN_LACS)'],axis = 1)


# # Final DATA

# In[35]:


new_df


# In[36]:


X = new_df.loc[:, new_df.columns != 'TARGET(PRICE_IN_LACS)']
y = new_df['TARGET(PRICE_IN_LACS)']


# In[37]:


from sklearn.model_selection import train_test_split, cross_val_score


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


# In[39]:


X_train.shape,X_test.shape


# In[40]:


from sklearn.ensemble import GradientBoostingRegressor


# In[41]:


gbr = GradientBoostingRegressor(max_depth=9, n_estimators=154, max_features=6)


# In[42]:


gbr.fit(X_train, y_train)


# In[43]:


gbr.score(X_test, y_test)


# ## Ploting the difference of real and predicted data

# In[48]:


fig, ax = plt.subplots(figsize=(30, 10))
ax.plot(y_test.to_list()[:150], 
        label='First 150 values', color='red', linewidth=3)
ax.plot(gbr.predict(X_test)[:150], 
        label='Predicted first 150 values', 
        linestyle='dashed', linewidth=3)
ax.legend(prop={"size":20})


# # THANKYOU
