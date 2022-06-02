# -*- Coding: utf-8 -*-
# @Time     : 2022/2/2 0:21
# @Author   : Linqi Xiao
# @Software : PyCharm
# @Version  : python 3.6
# @Description : Practice the computation of KNN


dataset = [[0, 1, 0], [0, 1, 1], [1, 2, 1], [1, 2, 0],
           [1, 2, 2], [2, 2, 2], [1, 2, -1], [2, 2, 3],
           [-1, -1, -1], [0, -1, -2], [0, -1, 1], [-1, -2, 1]]
clas = ['A', 'A', 'A', 'A',
        'B', 'B', 'B', 'B',
        'C', 'C', 'C', 'C']


def euclideanDistance3D(pointA, pointB):
    x1, y1, z1 = pointA
    x2, y2, z2 = pointB
    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
    return dist


def kNNClassify(testdata: list, dataSet: list, labels: list, k: int):
    dist_dict = {}
    res = []

    for idx, data in enumerate(dataSet):
        dist = euclideanDistance3D(data, testdata)
        # dist_dict[dist] = labels[idx]
        dist_dict[labels[idx]] = dist
    # print(dist_dict)
    k_pairs = sorted(dist_dict.items(), key=lambda item: item[1])[:k]
    for label, dist in k_pairs:
        res.append(label)
    pred = max(res, key=res.count)
    return pred, k_pairs


if __name__ == '__main__':
    for K in range(1, 4):
        result, all_labels = kNNClassify(testdata=[1, 0, 1], dataSet=dataset, labels=clas, k=K)
        print(f'** K = {K},', 'Classified Label:', result)
        print(all_labels)
