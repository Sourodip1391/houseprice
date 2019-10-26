mport pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


# In[28]:


# Import melbourne housing price
house=pd.read_csv('Melbourne_housing.csv')
house


# In[3]:


# To check the dimension of the dataset
house.shape


# In[ ]:


sns.distplot(house['Price'], fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(house['Price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Price distribution')
plt.ticklabel_format(style='plain')

fig = plt.figure()
res = stats.probplot(house['Price'], plot=plt)
plt.show()

print("Skewness: %f" % house['Price'].skew())
print("Kurtosis: %f" % house['Price'].kurt())


# As we can see the distribution is not normal and also there is some skewness and there is long tail to the right side.

# In[29]:


# Check for missing values
sns.set_style("whitegrid")
missing = house.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# In[129]:


# Filling missing values of Price with median since the distribution is not normal
house['Price']= house['Price'].fillna(house['Price'].median())
house['Distance']= house['Distance'].fillna(house['Distance'].median())
house['Lattitude']=house['Lattitude'].fillna(0)
house['Longtitude']=house['Longtitude'].fillna(0)
house['Bathroom']=house['Bathroom'].fillna(house['Bathroom'].median())
house['Bedroom2']=house['Bedroom2'].fillna(0)
house['Car']=house['Car'].fillna(0)
house['Landsize']=house['Landsize'].fillna(house['Landsize'].median())
house['YearBuilt']=house['YearBuilt'].fillna(0)
house['BuildingArea']= house['BuildingArea'].fillna(house['BuildingArea'].median())
house


# In[6]:


# Checking the categorical variables 
house.select_dtypes(include=['object']).columns


# In[20]:


house.select_dtypes(include=['int64','float64']).columns


# In[58]:


# Suplots of features v price
sns.set_style('darkgrid')
f, axes = plt.subplots(2,2, figsize = (16,16))

# Plot [0,0]
sns.boxplot(data = house, x = 'Type', y = 'Price', ax = axes[0,0])
axes[0,0].set_xlabel('Type')
axes[0,0].set_ylabel('Price')
axes[0,0].set_title('Type v Price')

# Plot [1,0]
sns.boxplot(x = 'Regionname', y = 'Price', data = house, ax = axes[0,1])
axes[1,0].set_xlabel('Regionname')
#axes[1,0].set_ylabel('Price')
axes[1,0].set_title('Region Name v Price')


# In[10]:


# Top 10 suburbs in Melbourne
Suburb=house['Suburb'].value_counts()
Suburb.head(10)


# In[11]:


house[house.columns[1:]].corr()['Price'][:]


# In[51]:


#select only the data we are interested in
attributes= ['Price', 'Distance', 'Bathroom', 'Rooms', 'Car', 'Landsize', 'BuildingArea','Lattitude', 'Longtitude', 'Propertycount']
h= house[attributes]

#whitegrid
sns.set_style('whitegrid')
#compute correlation matrix...
corr_matrix=h.corr(method='spearman')
#...and show it with a heatmap
#first define the dimension
plt.figure(figsize=(20,15))

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, vmax=1, vmin =-1, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[59]:


# exploratory data analysis of the numerical data
sns.set_style('darkgrid')
f, axes = plt.subplots(3,2, figsize = (20,30))

# Plot [0,0]
axes[0,0].scatter(x = 'Rooms', y = 'Price', data = house, edgecolor = 'b')
axes[0,0].set_xlabel('Rooms')
axes[0,0].set_ylabel('Price')
axes[0,0].set_title('Rooms v Price')

# Plot [0,1]
axes[0,1].scatter(x = 'Distance', y = 'Price', data = house, edgecolor = 'b')
axes[0,1].set_xlabel('Distance')
# axes[0,1].set_ylabel('Price')
axes[0,1].set_title('Distance v Price')

# Plot [1,0]
axes[1,0].scatter(x = 'Bathroom', y = 'Price', data = house, edgecolor = 'b')
axes[1,0].set_xlabel('Bathroom')
axes[1,0].set_ylabel('Price')
axes[1,0].set_title('Bathroom v Price')

# Plot [1,1]
axes[1,1].scatter(x = 'Car', y = 'Price', data = house, edgecolor = 'b')
axes[1,0].set_xlabel('Car')
axes[1,1].set_ylabel('Price')
axes[1,1].set_title('Car v Price')

# Plot [2,0]
axes[2,0].scatter(x = 'Landsize', y = 'Price', data = house, edgecolor = 'b')
axes[2,0].set_xlabel('Landsize')
axes[2,0].set_ylabel('Price')
axes[2,0].set_title('Landsize v  Price')

# Plot [2,1]
axes[2,1].scatter(x = 'BuildingArea', y = 'Price', data = house, edgecolor = 'b')
axes[2,1].set_xlabel('BuildingArea')
axes[2,1].set_ylabel('BuildingArea')
axes[2,1].set_title('BuildingArea v Price')


plt.show()


# In[130]:


# Split
# Create features variable

features = house[['Rooms','Bathroom','Car','BuildingArea','Price']]
features
# Create target variable
X=features.drop(['Price'],axis=1).values
y=features['Price'].values
# Train, test, split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state= 0)


# In[131]:


# Normalizing the variables
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[132]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[133]:


model=LinearRegression()
model.fit(X_train,y_train)
print(model)


# In[134]:


y_pred=model.predict(X_test)
print(y_pred)


# In[135]:


from sklearn.metrics import mean_squared_error
predictions = model.predict(X_train)
MSE = mean_squared_error(y_train, predictions)
RMSE = np.sqrt(MSE)
print('RMSE of the model is', round(RMSE,2))


# In[136]:


import pickle
import requests
import json


# In[137]:


pickle.dump(model, open('model.pkl','wb'))


# In[142]:


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,2.5,2,79]]))
