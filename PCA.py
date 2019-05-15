# In this i will implement PCA, to kow how pca works refer net
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')
# print(df.head())

labels = df['Species'] # taking species column as labels
X = df.drop(['Id', 'Species'], axis=1)
# we have separated our labels and features

# Step 1 of PCA, ie we center it around origin
X_std = StandardScaler().fit_transform(X)

pca = PCA(n_components=2) # initially we take all features so that we could find out all those who are useful
X_transform = pca.fit_transform(X_std)
# next we will find out how much variance each component have
# explained_variance_ is the attribute which will allow us to see that

# print(pca.explained_variance_)#we see that first two have the highest variance,so we'll change n_components value to 2

# print(X_transform)
# in order to plot it in graph i am seperating the values by using inverse zip
pca1 = list(zip(*X_transform))[0]
# print(pca1)
pca2 = list(zip(*X_transform))[1]

color_dict = {}
color_dict['Iris-setosa'] = 'green'
color_dict['Iris-versicolor'] = 'red'
color_dict['Iris-virginica'] = 'blue'

i=0
for label in labels:
    plt.scatter(pca1[i], pca2[i], color=color_dict[label])
    i=i+1

plt.show()