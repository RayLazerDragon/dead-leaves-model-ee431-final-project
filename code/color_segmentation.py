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
    cluster_centers = np.array(model.cluster_centers_, dtype=np.int)
    result = np.array([cluster_centers[idx] for idx in cluster_index], dtype=np.uint8).reshape(shape)
    imsave(f'../results/segmented/{filename}.png', result)
    return model, result


if __name__ == '__main__':
    img = cv2.imread('../dataset/sea1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))

    _, result = segmentation(img, n_color=6)
    # imsave('../results/segmented/sea1.png', result)

