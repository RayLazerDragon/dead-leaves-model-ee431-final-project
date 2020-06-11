import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imsave
from sklearn.cluster import KMeans
from PIL import Image


def segmentation(image, n_color=32, filename='unnamed'):
    model = KMeans(n_clusters=n_color)
    shape = image.shape

    image = np.reshape(image, (-1, 3))

    cluster_index = model.fit_predict(image)

    idx_img = np.reshape(cluster_index, (512, 512))

    bincount = []
    for i in range(4):
        for j in range(4):
            bin = idx_img[128*i:128*(i+1), 128*j:128*(j+1)]
            bin = np.reshape(bin, 128*128)
            tmp = list(np.bincount(bin) / len(bin))
            while len(tmp) < n_color:
                tmp.append(0)
            bincount.append(tmp)


    # bincount = list(np.bincount(cluster_index) / len(cluster_index))
    # print(bincount)

    cluster_centers = np.array(model.cluster_centers_, dtype=np.int)
    result = np.array([cluster_centers[idx] for idx in cluster_index], dtype=np.uint8).reshape(shape)
    imsave(f'../results/segmented/{filename}.png', result)
    return model, result, np.array(bincount)


if __name__ == '__main__':

    print(int(0.99/0.25))
    # img = cv2.imread('../dataset/mountain.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (512, 512))
    #
    # model, result, distribution = segmentation(img, n_color=6)
    # print(distribution.shape)

    # x = np.random.choice(6, 1, p=distribution)
    # print(int(x))
    # imsave('../results/segmented/sea1.png', result)
