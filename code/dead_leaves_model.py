import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2
from color_segmentation import segmentation
from matplotlib.image import imsave


def compute_dead_leaves_model(img_size=500, sigma=3, shape='disk', num_iter=5000, rmin=0.01, rmax=1, n_color=32):
    result = np.zeros((img_size, img_size)) + np.Inf  # This will be our image to work on
    x = np.linspace(0, 1, img_size)
    Y, X = np.meshgrid(x, x)

    sampling_rate = 200
    rad_list = np.linspace(rmin, rmax, sampling_rate)
    rad_dist = 1 / (rad_list ** sigma)

    if sigma > 0:
        rad_dist -= 1 / (rmax ** sigma)

    csum = np.cumsum(rad_dist)
    rad_dist = (csum - np.min(csum)) / (np.max(csum) - np.min(csum))  # map the radius distance to 0~1

    ref_img = cv2.imread('../dataset/space1.jpg')  # choose your reference image to color
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    model, _ = segmentation(ref_img, n_color=n_color, filename='space1')

    for _ in range(num_iter):
        r = np.random.rand(1)
        min_index = np.argmin(np.abs(r - rad_dist))

        r = rad_list[min_index]
        x = np.random.rand(1)
        y = np.random.rand(1)
        a = np.random.randint(0, n_color)

        if shape == 'disk':
            table = np.isinf(result) & (((X - x) ** 2 + (Y - y) ** 2) < r ** 2)
            result[table] = a

        elif shape == 'square':

            table = np.isinf(result) & (np.abs(X - x) < r) & (np.abs(Y - y) < r)
            result[table] = a

        if np.inf not in result:
            break

    result = np.where(np.isinf(result), 0, result)

    return np.array(result, dtype=np.int), model


if __name__ == '__main__':
    cluster_index, model = compute_dead_leaves_model(img_size=512, sigma=4, shape='disk', n_color=6)

    shape = cluster_index.shape
    cluster_index = np.reshape(cluster_index, shape[0]*shape[1])

    cluster_centers = np.array(model.cluster_centers_, dtype=np.int)
    result = np.array([cluster_centers[idx] for idx in cluster_index], dtype=np.uint8).reshape((shape[0], shape[1], 3))

    plt.imshow(result)
    plt.show()
    imsave('../results/dead_leaves/generated_space1.png', result)

