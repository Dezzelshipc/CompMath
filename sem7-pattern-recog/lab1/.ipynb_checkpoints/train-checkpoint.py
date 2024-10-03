import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from matplotlib import colormaps

x, y = fetch_openml("mnist_784", return_X_y=True, as_frame=False, parser='liac-arff')

x_train, x_test, y_train, y_test = train_test_split(x, y)

print(list(map(len, [x_train, x_test, y_train, y_test])))

x_train = x_train / 255.0
x_test = x_test / 255.0

K = 10

knc = KNeighborsClassifier(n_neighbors=K)
knc = knc.fit(x_train, y_train)

import joblib
joblib.dump(knc, "model.pkl")
print("Saved model.pkl")

y_predicted = knc.predict(x_test)
score = accuracy_score(y_test, y_predicted)

print(score)

width = int(len(x_test[0]) ** 0.5)
cmap = colormaps["gray"]

inds = np.random.randint(len(y_test), size=9)
print(inds)
fig, axs = plt.subplots(3, 3)
for i in range(9):
    ii = inds[i]
    pic = np.resize(x_test[ii], (width, width))
    this_axs = axs[i // 3, i % 3]
    this_axs.imshow(pic, cmap=cmap)
    this_axs.set_title(f"p: {y_predicted[ii]} (r: {y_test[ii]})")
    this_axs.axis('off')

plt.show()