#!/usr/bin/env python
# coding: utf-8

# In[130]:


import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


# In[131]:


def distance(dt1, dt2):
    return math.sqrt(np.sum((dt1 - dt2) ** 2))


# In[132]:


def fillDistanceMatrix(distanceMatrix, X):
    for i in range(len(X)):
        j = i
        while j < len(X):
            distanceMatrix[i][j] = distance(X[i], X[j])
            distanceMatrix[j][i] = distanceMatrix[i][j]
            j += 1


# In[133]:


def find(parent,  i):
    if(parent[i] == i):
        return i
    else:
        parent = find(parent , parent[i])
        parent[i] = parent
        return parent


# In[134]:


def unionSet(parent,  x , z):
    xParent = find(parent ,x)
    zParent = find(parent ,z)
    parent[zParent] = xParent


# In[135]:


def appendChild(closedCluster, child):
    for ele in child[closedCluster[1]]:
        child[closedCluster[0]].append(ele)


# In[136]:


def findMin(distanceMatrix, parent):
    minn = float('inf')
    index = [-1, -1]
    for i in range(len(distanceMatrix)):
        j = i + 1
        while j < len(distanceMatrix):
            if minn > distanceMatrix[i][j] and parent[i] == i and parent[j] == j:
                minn = distanceMatrix[i][j]
                index = [i, j]
            j += 1
    return index


# In[ ]:


def linkageSingle(child, first, second, dataset):
    minn = float('inf')
    for child1 in child[first]:
        for child2 in child[second]:
            minn = min(minn, distance(dataset[child1], dataset[child2]))
    return minn


# In[137]:


def computeMatrix(distanceMatrix, parent, child, NearestClustere, dataset):
    for i in range(len(distanceMatrix)):
        if parent[i] == i:
            distanceMatrix[i][NearestClustere[0]] = linkageSingle(child, NearestClustere[0], i, dataset)
            distanceMatrix[NearestClustere[0]][i] = distanceMatrix[i][NearestClustere[0]]


# In[142]:


if __name__ == "__main__" : 
  
    df = pd.read_csv("cancer.csv", usecols = ["radius_mean", "texture_mean",  "perimeter_mean",	"area_mean",	
        "smoothness_mean",	"compactness_mean",	"concavity_mean",	"concave points_mean",	"symmetry_mean",	
        "fractal_dimension_mean",	"radius_se",	"texture_se",	"perimeter_se",	"area_se",	"smoothness_se",	
        "compactness_se",	"concavity_se",	"concave points_se",	"symmetry_se",	"fractal_dimension_se",	
        "radius_worst",	"texture_worst",	"perimeter_worst",	"area_worst",	"smoothness_worst",	"compactness_worst",	
        "concavity_worst",	"concave points_worst",	"symmetry_worst",	"fractal_dimension_worst" ])

    X = df.iloc[:, ].values
    
    X = X.astype(float)
    
    k  = 2 
    X = np.array(X)
    distanceMatrix = np.zeros((len(X), len(X)))
    
    
    # initially Ecah data point is a cluster in itself , so each data point is parent of itself  
    parent = [0]*len(X)
    for i in range(len(X)):
        parent[i] = i    
        
    child = [[i] for i in range(len(X))]
    
    fillDistanceMatrix(distanceMatrix, X)
    
    clusters = len(X)
    while clusters > k:
        print("clusters = ", clusters)
        NearestCluster = findMin(distanceMatrix, parent)
        parent[NearestCluster[1]] = NearestCluster[0]
        appendChild(NearestCluster, child)
        computeMatrix(distanceMatrix, parent, child, NearestCluster, X)
        clusters -= 1
    
    getCluster = []
    for i in range(len(X)):
        if parent[i] == i:
            getCluster.append(child[i])
    
    labels = [0]*len(X)
    for i in range(len(getCluster[0])):
        labels[getCluster[0][i]] = 1
    
    colors = ["g.", "r. ", "c." , "b.", "k.", "o."]

    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 7)


    print("Number of points in Cluster 1 and 2 are : ", len(getCluster[0]) ," & ", len(X) - len(getCluster[0])  , "respectively")    
    plt.xlabel("Radius_Mean")
    plt.ylabel("Texture_Mean")
    plt.show()

   
    


# In[ ]:




