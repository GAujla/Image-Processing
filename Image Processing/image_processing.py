import skimage.segmentation
import skimage.util
import skimage.feature
import numpy as np
import scipy
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage import io
from skimage.filters import threshold_mean
from skimage.transform import hough_line, hough_line_peaks

# Read the Image
img = io.imread('data/avengers_imdb.jpg')

# Displays width, height and channel
print('Image size is:',img.size)

# Converts the image to grayscale
grayscale_avengers = rgb2gray(img)
plt.imshow(grayscale_avengers, cmap=plt.cm.gray)
plt.title('Grayscale Avengers')
plt.savefig('outputs/grayscale_avengers.jpg')
plt.show()

# Converts an image to binary
thresh = threshold_mean(grayscale_avengers)
black_white_avengers= grayscale_avengers > thresh
plt.imshow(black_white_avengers, cmap='gray')
plt.title('Black and White Avengers')
plt.savefig('outputs/black_white_avengers.jpg')


# https://scikit-image.org/docs/0.12.x/api/skimage.filters.html#skimage.filters.gaussian_filter
# https://scikit-image.org/docs/dev/api/skimage.filters.html
bush_h= io.imread('data/bush_house_wikipedia.jpg')
# Adds random noise to the image with a variance of 0.1
gaussian_noise_bush = skimage.util.random_noise(bush_h, mode='gaussian', var = 0.1)
# Uses a gaussian mask with a sigma of 0.1
gaussian_mask_bush = skimage.filters.gaussian(gaussian_noise_bush,sigma=1,multichannel=False)
# Uses a uniform mask
uniform_mask_bush = scipy.ndimage.uniform_filter(gaussian_noise_bush, size=(9,9,1))

# Displays Image
plt.imshow(bush_h)
plt.title('Original')
plt.show()

plt.imshow(gaussian_noise_bush)
plt.title('Gaussian Random Noise')
plt.savefig('outputs/gaussian_noise_bush.jpg')
plt.show()

plt.imshow(gaussian_mask_bush)
plt.title('Gaussian Mask')
plt.savefig('outputs/gaussian_mask_bush.jpg')
plt.show()

plt.imshow(uniform_mask_bush)
plt.title('Uniform Mask')
plt.savefig('outputs/uniform_mask_bush.jpg')
plt.show()

# Question 3
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
forestry_c= io.imread('data/forestry_commission_gov_uk.jpg')
# Segments an image using kmeans into 5 segments using an iteration of 5
forest_kmeans_seg = skimage.segmentation.slic(forestry_c, n_segments=5, max_iter=5,start_label=1)
plt.title('Kmeans Segmentation n = 5')
plt.imshow(forest_kmeans_seg)
plt.savefig('outputs/forest_kmeans_seg.jpg')
plt.show()

# Performs Canny edge detection and apply Hough Transform
# https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.canny For canny edge and Mask
# https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html For np.mgrid to create Mask
rolland_garros = io.imread('data/rolland_garros_tv5monde.jpg')
grayscale_rolland = rgb2gray(rolland_garros)
# Defines x and y axis of the image, will be used to create a mask
x, y = np.mgrid[:grayscale_rolland.shape[0], :grayscale_rolland.shape[1]]
# Converts image into canny edge, using sigma, mask and high_threshold
canny_rolland_garros = skimage.feature.canny(grayscale_rolland, sigma = 0.89, mask = (x > 150) & (x < 950) & (y > 200) & (y < 1200),high_threshold=0.9)
plt.imshow(canny_rolland_garros,cmap='gray')
plt.title('Canny Edge')
plt.savefig('outputs/canny_forest.jpg')
plt.show()

# https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html#sphx-glr-auto-examples-edges-plot-line-hough-transform-py


out, angles, d = hough_line(canny_rolland_garros)

plt.imshow(canny_rolland_garros, cmap='gray')
plt.ylim((grayscale_rolland.shape[0], 0))
# Performs Hough Straight Line on canny image
for _, angle, dist in zip(*hough_line_peaks(out, angles, d)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    plt.axline((x0, y0), slope=np.tan(angle + np.pi / 2))
plt.tight_layout()
plt.title('Hough Straight line')
plt.show()
