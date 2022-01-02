import random

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import fetch_olivetti_faces

data = fetch_olivetti_faces()


images = data.images

relevant_numbers = np.linspace(0, 390, 40, dtype=int)

print(len(relevant_numbers))
print(relevant_numbers)

random_number = random.choice(relevant_numbers)  # random number

random_image_raw = images[random_number]

# plt.imshow(random_image_raw, cmap='Greys_r')
# plt.show()



# random_image = np.reshape(random_image_raw, [64, 64])


from_figs_all = []
for idx in relevant_numbers:
    front_pic = (data.data[idx, :] - np.mean(data.data[idx, :]))  # / np.std(data.data[idx, :])
    from_figs_all.append(front_pic)

data_40 = np.array(np.transpose(from_figs_all))

cov_mat = np.cov(data_40)


from sklearn import decomposition
pca = decomposition.PCA(n_components=20, whiten=True)
pca.fit(data_40)
print(pca.components_.shape)

fig = plt.figure(figsize=(16, 6))
for i in range(30):
    ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape(data.images[0].shape),
              cmap=plt.cm.bone)



# vals, vecs = np.linalg.eig(cov_mat)
# idx = np.flip(np.argsort(vals))
# vals = vals[idx]
# vecs = vecs[idx]
#
# #
# p1 = vals[0] * np.reshape(vecs[0], [64, 64])  # *vals[0]
# p2 = vals[1] * np.reshape(vecs[1], [64, 64])  # *vals[1]
# # p3 = cov_mat * vecs[2] #*vals[2]
# # p4 = cov_mat * vecs[3] #*vals[3]
# # p5 = cov_mat * vecs[4] #*vals[4]
#
# plt.imshow(np.real(p1), cmap='Greys_r')
# plt.show()


# plt.bar(x=range(len(vals)), height=vals)
# plt.show()

print('done')
