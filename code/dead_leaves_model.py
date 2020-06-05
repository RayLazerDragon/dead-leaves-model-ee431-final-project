import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_dead_leaves_model(img_size=500, sigma=3, shape='disk', num_iter=5000, rmin=0.01, rmax=1):

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

    for _ in range(num_iter):
        r = np.random.rand(1)
        min_index = np.argmin(np.abs(r - rad_dist))

        r = rad_list[min_index]
        x = np.random.rand(1)
        y = np.random.rand(1)
        a = np.random.rand(1)

        if shape == 'disk':
            table = np.isinf(result) & (((X - x) ** 2 + (Y - y) ** 2) < r ** 2)
            result[table] = a

        elif shape == 'square':
            table = np.isinf(result) & ((np.abs(X - x) + np.abs(Y - y)) < r)
            result[table] = a

        if np.inf not in result:
            break

    result = np.where(np.isinf(result), 0, result)

    return result


if __name__ == '__main__':
    image = compute_dead_leaves_model(img_size=500, sigma=3)
    plt.imshow(image, cmap='gray')
    plt.show()


