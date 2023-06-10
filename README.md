# Regression
import pandas as pd\\
import numpy as np\\
import matplotlib.pyplot as plt
dataset = pd.read_csv("kddcup99.csv")
x = dataset.iloc[:, [0, 18]].values
y = dataset.iloc[:, 19].values
class_one = 0
class_two = 1
for i in y:
    if i ==0:
         class_one+=1
           else:
             class_two+=1
values = np.array([class_one, class_two])
label = ["Attack", "No attack"]
plt.pie(values, labels = label)
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 0)\\

from sklearn.preprocessing import StandardScaler
sc_ x = StandardScaler()
xtrain = sc_x.fit_transform(X_train)
xtest = sc\textunderscore x.transform(X_test)
print (xtrain[0:10, :])

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, y_train)
y_pred = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print ("Confusion Matrix : \n", cm)

from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from matplotlib.colors import ListedColormap
X_set, y_set = xtest, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1,stop = X_set[:, 0].max() + 1, step = 0.01),np.arange(start = X_set[:, 1].min() - 1,stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmapListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y\textunderscore set)):
    plt.scatter(X_set[y_set == j, 0] X_set[y_set == j, 1],c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier')
plt.legend()
plt.show()
