from PIL import Image
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib import colormaps

cmap = colormaps["gray"]
knc: KNeighborsClassifier = joblib.load("model.pkl")

image = Image.open("pics/Pic0.bmp")

image_f = np.array(image).flatten() / 255.

print(*zip(knc.classes_, knc.predict_proba([image_f])[0]))

image_pred = knc.predict([image_f])[0]

plt.figure(image_pred)
plt.imshow(image, cmap=cmap)
plt.title(image_pred)

plt.show()

