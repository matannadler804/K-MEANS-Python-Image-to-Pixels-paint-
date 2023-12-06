from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


image = Image.open('meital.jpg')

image_array = np.array(image)

pixels = image_array.reshape(-1, 3)

num_clusters = 10  

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(pixels)

cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

clustered_pixels = cluster_centers[cluster_labels]
clustered_image = clustered_pixels.reshape(image_array.shape)

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image_array), plt.title('Original Image')
plt.subplot(122), plt.imshow(clustered_image.astype(np.uint8)), plt.title('Clustered Image')
plt.show()
