import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
# mpl.use('Agg')


# load mini training data and labels
mini_train = np.load('knn_minitrain.npy')
mini_train_label = np.load('knn_minitrain_label.npy')

# randomly generate test data
np.random.seed(0)
mini_test = np.random.randint(20, size=20)
mini_test = mini_test.reshape(10, 2)


# Define knn classifier
def kNNClassify(newInput, dataSet, labels, k):
    result = []
    for new_xy in newInput:
        dist_dict = {}
        knn_labels = []
        for old_id, old_xy in enumerate(dataSet):
            dist = euclideanDistance2D(new_xy, old_xy)
            dist_dict[old_id] = dist
        k_pairs = sorted(dist_dict.items(), key=lambda item: item[1])[:k]
        for old_knn_id, knn_dist in k_pairs:
            knn_labels.append(labels[old_knn_id])
        pred = max(knn_labels, key=knn_labels.count)
        result.append(pred)
    return result


def euclideanDistance2D(pointA, pointB):
    x1, y1 = pointA
    x2, y2 = pointB
    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return dist


outputlabels = kNNClassify(mini_test, mini_train, mini_train_label, 5)

print('random test points are:', mini_test)
print('knn classfied labels for test:', outputlabels)

# plot train data and classfied test data
train_x = mini_train[:, 0]
train_y = mini_train[:, 1]
fig = plt.figure()
plt.scatter(train_x[np.where(mini_train_label == 0)], train_y[np.where(mini_train_label == 0)], color='red')
plt.scatter(train_x[np.where(mini_train_label == 1)], train_y[np.where(mini_train_label == 1)], color='blue')
plt.scatter(train_x[np.where(mini_train_label == 2)], train_y[np.where(mini_train_label == 2)], color='yellow')
plt.scatter(train_x[np.where(mini_train_label == 3)], train_y[np.where(mini_train_label == 3)], color='black')

test_x = mini_test[:, 0]
test_y = mini_test[:, 1]
outputlabels = np.array(outputlabels)
plt.scatter(test_x[np.where(outputlabels == 0)], test_y[np.where(outputlabels == 0)], marker='^', color='red')
plt.scatter(test_x[np.where(outputlabels == 1)], test_y[np.where(outputlabels == 1)], marker='^', color='blue')
plt.scatter(test_x[np.where(outputlabels == 2)], test_y[np.where(outputlabels == 2)], marker='^', color='yellow')
plt.scatter(test_x[np.where(outputlabels == 3)], test_y[np.where(outputlabels == 3)], marker='^', color='black')

# save diagram as png file
# plt.show()

plt.savefig("miniknn.png")
