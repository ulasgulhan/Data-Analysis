import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('Data/teleCust1000t.csv')
print(df.head().to_string())


# In the KNN algorithm, geometric calculations are made to compute the distance from each point to another point. For this, we will take each value in our dataset. Essentially, I am creating a matrix here.

# When we examine the values, we see that while one value in the income column is 944, the values in the age column or in the region column are either 1 or 0. These values have different magnitudes as a scaler. Therefore, we need to normalize the values in our dataset.


# Normalization Process


X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
        'employ', 'retire', 'gender', 'reside']].values

y = df['custcat'].values

# The fields to be normalized have been created above. Now let's perform the normalization process using the sklearn preprocessing module.

X = preprocessing.StandardScaler().fit_transform(X)

# Let's split the dataset.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# I am training my KNN model to look at the nearest 4 neighbors for each point in our training dataset, i.e., for each feature. Here, we arbitrarily chose 4 neighbors without any specific logic.

neighbor = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)
y_result = neighbor.predict(X_test)

# Accuracy evaluation
# When creating multiple classes or labels in models, it is important to validate each subclass. Here, we can do this using the Jaccard Similarity Index, which is heavily used in the field of statistics. We will use the accuracy_score() function found in the sklearn module to find the similarity and diversity between 'y_train' and 'neighbor.predict(X_train)', and 'y_test' and 'neighbor.predict(X_test)'. The result returned by this function is successful if it approaches 1 and unsuccessful if it approaches 0.

print(f'Train Set Accuracy: {metrics.accuracy_score(y_train, neighbor.predict(X_train))}')
print(f'Test Set Accuracy: {metrics.accuracy_score(y_test, neighbor.predict(X_test))}')

# Let's examine the question of how many nearest neighbors we should look at to find the best 4 classes in the KNN algorithm. Above, we arbitrarily chose 4. Now let's retrain our model by looking at nearest neighbors ranging from 1 to 10.

k_neigh = int(input('Please type K number: '))
array_length = k_neigh - 1
jsi_acc = np.zeros(array_length)
std_acc = np.zeros(array_length)

for k in range(1, k_neigh):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    jsi_acc[k-1] = metrics.accuracy_score(y_test, y_pred)
    std_acc[k-1] = np.std(y_pred == y_test) / np.sqrt(y_pred.shape[0])

print(f'JSI Score: {jsi_acc}')
print(f'JSI Score: {std_acc}')

plt.plot(range(1, k_neigh), jsi_acc, c='g')
plt.fill_between(range(1, k_neigh), jsi_acc - 1 * std_acc + 1 * std_acc, alpha=0.10)
plt.legend(('accuracy', 'std'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbor')
plt.grid(True)
plt.tight_layout()
plt.show()

print(f'The best accuracy was with: {jsi_acc.max()}, with k={jsi_acc.argmax()}')
