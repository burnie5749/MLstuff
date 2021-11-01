import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def nearest_mean_classifier(training_data, training_labels,
                            testing_data, testing_labels):
    # class labels 1,2...c
    c = np.max(training_labels).astype('int')  # number of classes
    n = len(training_data[0])  # number of features

    m = np.empty((c, n))  # array to hold the means

    for i in range(c):
        m[i] = np.mean(training_data[training_labels == i + 1], axis=0)
        assigned_labels = np.empty(len(testing_labels))
        for i in range(len(testing_labels)):
            x = testing_data[i]  # object i from the testing data
            di = np.sum((m - x) ** 2, axis=1)  # distance to means
            assigned_labels[i] = np.argmin(di) + 1
        e = np.mean(testing_labels != assigned_labels)
        return e, assigned_labels


df = pd.read_csv("data_banknote_authentication.txt", header=None)  # this is the data set Z
# print(df.info())
# convert to a numpy array
npdf = df.to_numpy()  # numpy datafile
# print(npdf.info())
# split into data and labels
data = npdf[:, :-1]  # features of the banknotes
labels = npdf[:, -1]  # labels of the  banknotes

n = len(data[0])  # No of features
N = len(data)  # No of objects

for i in range(len(labels)):
    labels[i] += 1  # encode classes and 1 and 2
c = np.max(labels).astype('int')  # No of classes

print(f"No of classes: {c} \nNo of features: {n} \nNo of Objects: {N}")
plt.figure(figsize=(10, 8))
k = 1
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, k)
        plt.scatter(data[:, i], data[:, j], c=labels, s=0.5)
        plt.axis('Off')
        plt.axis('Equal')
        titleString = f"{i + 1}v{j + 1}"
        plt.title(titleString, size=8)
        k += 1
plt.show()
ResubError, _ = nearest_mean_classifier(data, labels, data, labels)  # calculate the resub using nearest_mean_classifier
print(f"Resubstitution error: {ResubError}")

# split the data using numpy for the holdout method

np.random.shuffle(data)
np.random.shuffle(labels)
splitPercent = int(0.5 * len(npdf))
trd, tsd, trl, tsl = data[:splitPercent], data[splitPercent:], labels[:splitPercent], labels[:splitPercent]
# print(f"training data: {len(trd)}") # used for checking the split of data
holdoutError, _ = nearest_mean_classifier(trd, trl, tsd, tsl)  # calculate the holdout using nearest_mean_classifier
print(f"Holout Error: {holdoutError}")
