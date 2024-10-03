from PIL import Image
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os


cmap = colormaps["gray"]
knc: KNeighborsClassifier = joblib.load("model.pkl")

pics_path = os.scandir("../pics")


def prepare(pic_path):
    image = Image.open(pic_path)
    return np.array(image).flatten() / 255.


images = np.array(list(map(prepare, pics_path)))
image_preds = knc.predict(images)
image_pred_probs = knc.predict_proba(images)

width = 28
rows, cols = 4, 4

fig, axs = plt.subplots(rows, cols)
for i in range(min(rows * cols, len(image_preds))):
    pic = np.resize(images[i], (width, width))
    this_axs = axs[i // rows, i % cols]
    this_axs.imshow(pic, cmap=cmap)
    this_axs.set_title(f"[{i}] {image_preds[i]} ({image_pred_probs[i].max()})")
    print(i, list(sorted(zip(knc.classes_, image_pred_probs[i]), key=lambda x: -x[1])))

for i in range(rows * cols):
    axs[i // rows, i % cols].axis('off')

plt.autoscale()
plt.show()
