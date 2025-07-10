from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


img_path = './color_map.png'
img_path = './retriever.webp'

img = Image.open(img_path).convert('RGB')
img_array = np.array(img)
h, w, _ = img_array.shape

sample_rate = 10
pixels = img_array[::sample_rate, ::sample_rate, :].reshape(-1, 3)

pixels = img_array.reshape(-1, 3)
r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

# 그래프 설정
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(r, g, b, c=pixels/255.0, marker='o', s=0.5, alpha=0.6)

ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.set_title('3D RGB Pixel Distribution')

plt.show()
