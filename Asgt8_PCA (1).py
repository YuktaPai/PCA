#!/usr/bin/env python
# coding: utf-8

# Perform Principal component analysis and perform clustering using first 
# 3 principal component scores 
# (both heirarchial and k mean clustering(scree plot or elbow curve) and obtain 
# optimum number of clusters 
# and check whether we have obtained same number of clusters with the original data 
# (class column we have ignored at the begining who shows it has 3 clusters)df
# 

# In[2]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


# In[3]:


columns = ['class','alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
    'proanthocyanins', 'color_intensity', 'hue',
    'dilution_of_wines', 'proline']
wine = pd.read_csv('C:/Users/17pol/Downloads/wine.csv', names = columns, header = 0 )
wine.head()


# In[4]:


wine.describe().transpose()


# In[5]:


wine.info()


# In[6]:


wine.isna().sum()


# In[8]:


#checking unique classes or predefined clusters present in data
wine['class'].nunique()


# In[9]:


wine['class'].value_counts()


# In[11]:


#Plotting class counts
wine['class'].value_counts().plot.bar(color = 'purple')
plt.legend()
plt.xlabel('class')


# ### # Using the standard scaler method to get the values converted into integers

# In[16]:


#considering only numerical data
wine.data = wine.iloc[:,1:].values
from sklearn.preprocessing import StandardScaler

#Normalizing values
Wine_normal = scale(wine.data)
Wine_normal.shape


# ### PCA

# In[18]:


'''We use PCA to reduce dimensionality of data for better processing and lesser machine load requirement. 
These give better results as more and more data can be accomodated with better importance allocation to different features'''

pca = PCA()
pca_values = pca.fit_transform(Wine_normal)


# In[19]:


pca_values


# In[20]:


# Creating a dataframe featuring the three Principal components that we acquired through PCA.
df = pd.DataFrame(pca_values)
df.head()


# In[21]:


pca = PCA(n_components=3)
pca_values = pca.fit_transform(Wine_normal)


# In[22]:


#amount of variance that each pca explains is
var = pca.explained_variance_ratio_
var


# In[23]:


#cumulative variance
cumv = np.cumsum(np.round(var, decimals=4)*100)
cumv
#we get just 66.53% data with first 3 components


# In[24]:


pca.components_


# In[27]:


#variance plot for PCA components obtained
plt.plot(cumv, color='Red')


# In[28]:


#plot between pca components
x =pca_values[:,0:1]
y =pca_values[:,1:2]
z =pca_values[:,2:3]
plt.scatter(x,y)
plt.scatter(y,z)


# In[30]:


# Visualizing the results of the 3D PCA.
ax = plt.figure(figsize=(10,10)).gca(projection='3d')
plt.title('3D Principal Component Analysis (PCA)')
ax.scatter(
    xs=x, 
    ys=y, 
    zs=z, 
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()


# In[42]:


final_df = df.iloc[:,:3]
final_df = final_df.rename({0:'PCA1', 1:'PCA2', 2:'PCA3'}, axis = 1)
final_df.head()


# In[45]:


final_df = pd.concat([final_df,wine['class']], axis = 1)


# In[46]:


final_df.head()
final_df.info()


# In[47]:


import seaborn as sns
sns.scatterplot(data = final_df, x = 'PCA1', y = 'PCA2', hue = 'class')


# ## Performing Heirarchical Clustering

# In[48]:


#performing clustering using PCA1 and PCA2
#importing heirarchical clustering libraries

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[51]:


#  Normalizing Dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
pca_df_normal = scaler.fit_transform(final_df.iloc[:,:3])
print(pca_df_normal)


# In[52]:


# Creating clusters
from sklearn.cluster import AgglomerativeClustering
H_clusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
H_clusters


# In[53]:


y=pd.DataFrame(H_clusters.fit_predict(pca_df_normal),columns=['clustersid_H'])
y['clustersid_H'].value_counts()


# ### Conclusively 3 clusters formed by heirarchical clustering and KMeans, as present in original data 

# In[54]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


# In[55]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(pca_df_normal)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[56]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(pca_df_normal)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[58]:


#Build Cluster algorithm

KM_clusters = KMeans(3, random_state=42)
KM_clusters.fit(pca_df_normal)


# In[59]:


y=pd.DataFrame(KM_clusters.fit_predict(pca_df_normal),columns=['clusterid_Kmeans'])
y['clusterid_Kmeans'].value_counts()


# ### Preparing Actual Vs. Predicted Clusering Data

# In[66]:


wine_class = wine['class']
wine_class = pd.Series(wine_class)


# In[68]:


clustersid_HC = H_clusters.labels_
clustersid_HC = pd.Series(clustersid_HC)


# In[69]:


clusterid_Kmeans = KM_clusters.labels_
clusterid_Kmeans = pd.Series(clusterid_Kmeans)


# In[70]:


pred_df = pd.concat([wine_class, clustersid_HC, clusterid_Kmeans],axis = 1)
pred_df


# In[71]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




