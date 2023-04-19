#!/usr/bin/env python
# coding: utf-8

# # House Price Forecasting Through a Multiple Linear Regression 

# The purpose of this project is to predict a house price of unit area based on several house parameters. In this case Multiple Linear Regression is the best option to create an accurate model.

# ### Importing libraries

# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading the data set

# In[37]:


data = pd.read_csv(r"C:\Users\SilkRIT\Desktop\разное\ML_stepik\multilinear regression\Real_estate.csv")


# ### Data preparation

# Let's have a look at the data. Here we can see the dataframe with variables which column name starts with "X" and target column "Y house price of unit area".

# In[38]:


data.describe()


# In[39]:


data.head()


# Checking whether data has null values or not.

# In[40]:


pd.isnull(data).any()


# In[41]:


data.info()


# Analysing the distribution of price of unit area. It's likely to be normal with skewness.

# In[42]:


plt.figure(figsize =(10,6))
plt.hist(data['Y house price of unit area'], bins = 50, ec= 'black', color = '#2196f3')
plt.xlabel('Price')
plt.ylabel('№ of houses')
plt.show()


# Next I want to see the distribution of variable Distance to the nearest MRT station.

# In[43]:


plt.figure(figsize =(10,6))
plt.hist(data['X3 distance to the nearest MRT station'], bins = 50, ec= 'black', color = '#2196f3')
plt.xlabel('Distance to the nearest MRT station')
plt.ylabel('№ of houses')
plt.show()


# And now we may calculate the correlation coefficient of variables. One option is to do it manualy one by one.

# In[44]:


data['Y house price of unit area'].corr(data['X3 distance to the nearest MRT station'])


# In[45]:


data['Y house price of unit area'].corr(data['X4 number of convenience stores'])


# The other option is to use pandas tool corr which provides every correlation between variables.

# In[46]:


data.corr()


# I will build the heatmap to understand correlations better.

# In[47]:


mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True
mask


# On this heatmap we may see how variables influence on target value. The greater the absolute value of the coefficient, the greater the influence of the variable on the desired value. Also variables may be influenced by each other, for example longitude and distance to the nearest MRT station. It may lead to wrong model setup.  

# In[48]:


plt.figure(figsize =(16,10))
sns.heatmap(data.corr(), mask= mask, annot = True, annot_kws= {"size" : 14})
sns.set_style('white')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()


# Now I will build the scatter plot with correlation of distance to the nearest MRT station and price and then create the trend line.

# In[49]:


x3_y_corr = round(data['X3 distance to the nearest MRT station'].corr(data['Y house price of unit area']),2)

plt.figure(figsize = (9,6))
plt.scatter(x= data['X3 distance to the nearest MRT station'],
            y = data['Y house price of unit area'], alpha = 0.6, s=80, color ='skyblue')

plt.title(f'X3 vs Price (Correlation {x3_y_corr})', fontsize = 14)
plt.xlabel('distance to the nearest MRT station', fontsize = 14)
plt.ylabel('house price of unit area', fontsize = 14)

plt.show()


# This way I can see the negative relationship between two variables.

# In[50]:


sns.lmplot(x = 'X3 distance to the nearest MRT station', y ='Y house price of unit area',data = data)
plt.show()


# ### Data modeling

# First, we need to split the data on train and test samples 

# In[51]:


prices = data['Y house price of unit area']
features = data.drop(['Y house price of unit area','No'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state=10)


# Then I create the Linear Regression model, fit it and check the r-squared. Test data r-squared is 0.59, let's try to increase it.

# In[52]:


regr = LinearRegression()
regr.fit(X_train, y_train)

print('Training data r-squared:', regr.score(X_train, y_train))
print('Test data r-squared:', regr.score(X_test, y_test))

print('Intercept:', regr.intercept_)
pd.DataFrame(data= regr.coef_, index = X_train.columns, columns = ['coef'])


# We may start with checking the skewness of prices and then take logarithm of them.

# In[53]:


data['Y house price of unit area'].skew()


# In[54]:


data['Y house price of unit area'].min


# In[55]:


y_log = np.log(data['Y house price of unit area'])
y_log.tail()


# In[56]:


y_log.skew()


# In[57]:


sns.distplot(y_log)
plt.title(f'Log price with skew: {y_log.skew()}')
plt.show()


# Let's see what happend to r-squared of price logs. Test data r-squared increased to 0.71.

# In[58]:


prices = np.log(data['Y house price of unit area'])
features = data.drop(['Y house price of unit area','No'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state=10)

regr = LinearRegression()
regr.fit(X_train, y_train)

print('Training data r-squared:', regr.score(X_train, y_train))
print('Test data r-squared:', regr.score(X_test, y_test))

print('Intercept:', regr.intercept_)
pd.DataFrame(data= regr.coef_, index = X_train.columns, columns = ['coef'])


# Next step is to evaluate the P-value of each variable. P-value higher than 0.05 indicates than variable insignificant in our model. In our case it's longitude with 0.559 P-value.

# In[59]:


X_incl_const = sm.add_constant(X_train)

model = sm.OLS(y_train,X_incl_const)
results = model.fit()

pd.DataFrame({'coef' : results.params, 'p-value' : round(results.pvalues, 3)})


# To make sure that dropping longitude will not change our model we also need to check BIC value.
# BIC (Bayesian Information Criteria) estimates the likelihood of a model to predict. There is no explicitly 'good' BIC value. BIC values need to be compared. The best model for the data is the one with the lowest BIC value.
# So let's compare BIC and r-squared with and without longitude.

# In[60]:


X_incl_const = sm.add_constant(X_train)

model = sm.OLS(y_train,X_incl_const)
results = model.fit()

org_coef = pd.DataFrame({'coef' : results.params, 'p-value' : round(results.pvalues, 3)})

print('BIC is:', results.bic)
print('r-squared is:', results.rsquared)


# In[61]:


X_incl_const = sm.add_constant(X_train)
X_incl_const = X_incl_const.drop(['X6 longitude'], axis = 1)

model = sm.OLS(y_train,X_incl_const)
results = model.fit()

reduced_coef = pd.DataFrame({'coef' : results.params, 'p-value' : round(results.pvalues, 3)})
print('BIC is:', results.bic)
print('r-squared is:', results.rsquared)


# Now we see that creating model without longitude variable doesn't lead to r-squared reduction, but reduces the BIC.

# In[62]:


frames = [org_coef, reduced_coef]
pd.concat(frames,axis = 1)


# Then I plot the residuals of model. The main idea is to make sure that there is no tendency in correlation between residuals and predicted values. As you can see in our model residuals don't depend on predicted prices.

# In[63]:


prices = np.log(data['Y house price of unit area'])
features = data.drop(['Y house price of unit area','No', 'X6 longitude'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state=10)
X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train,X_incl_const)
results = model.fit()

plt.scatter(x= results.fittedvalues, y = results.resid, c = 'navy', alpha = 0.6)

plt.xlabel('Predicted log prices $\hat y _i$', fontsize = 14)
plt.ylabel('Residuals', fontsize = 14)
plt.title('Residuals vs Predicted log prices', fontsize = 17)

plt.show()


# Distribution of residuals is normal with zero mean value.

# In[64]:


resid_mean = round(results.resid.mean(),3)
resid_skew = round(results.resid.skew(),3)

sns.distplot(results.resid, color = 'navy')
plt.title(f'Log price model:residuals Skew ({resid_skew}) Mean ({resid_mean})')

plt.show()


# Before giving the answer about predicted price per unit area it is more correctly to evaluate confidence interval (+/- 2 std).
# For example, if our model predictes 30$ price per unit area the interval will be:

# In[65]:


reduced_log_mse = round(results.mse_resid, 3)

print('Evaluating interval for predicted 30$ price per unit area value with 95% accuracy:')

print(f'2 standart diviations in logs', 2*np.sqrt(reduced_log_mse))

upper_bound = np.log(30) + 2*np.sqrt(reduced_log_mse)
print('The upper bound in normal prices is $', np.e**upper_bound)

lower_bound = np.log(30) - 2*np.sqrt(reduced_log_mse)
print('The upper bound in normal prices is $', np.e**lower_bound)


# In[ ]:




