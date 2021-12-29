import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from noisy_array import NoisyArray
from os import mkdir
from os.path import exists
OUT_PATH = 'output'
if not exists(OUT_PATH):
    mkdir(OUT_PATH)
    print(f'created {OUT_PATH}')
sns.set_style(style='white')


#  data
X = np.linspace(0, 10, 50)
Y = np.linspace(0, 5, 50)

X = NoisyArray(X, scale=1)
Y = NoisyArray(Y, scale=1)

print(X), print(Y)

# normalization
X = (X - np.mean(X))/np.std(X)
Y = (Y - np.mean(Y))/np.std(Y)

# Plot original Data
original_data = plt.figure(figsize=[4, 4])
plt.plot(X, Y, LineStyle='', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.tight_layout()
plt.grid()
ax = original_data.axes[0]
ax.set_aspect('equal', 'box')
plt.savefig(os.path.join(OUT_PATH, 'original_data.png'), dpi=600)


# Compute covariance matrix
XY = [X, Y]
print('\nX, Y array:\n', np.array(XY))
cov_mat = np.cov(XY)
print('\ncovariance matrix: \n', cov_mat)

# Compute PCs
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nPC eigenvalues: \n', eigen_vals)
print('\nPC eigenvectors: \n', eigen_vecs)

# Plot PCs at the plot of the original data
ax = original_data.axes[0]
ax.set_aspect('equal', 'box')
ax.arrow(0, 0, *eigen_vecs[0],
         head_width=0.2, head_length=0.3, width=0.05, color='red')  # PC1
ax.arrow(0, 0, *eigen_vecs[1],
         head_width=0.2, head_length=0.3, width=0.05, color='blue')  # PC2
# plt.grid()
# plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, 'PCA_2D.png'), dpi=600)
plt.close()

# check if orthogonal:
PC1 = eigen_vecs[0]
PC2 = eigen_vecs[1]
res = np.dot(PC1, PC2)
print(PC1)
print(PC2)
print(res)

# check if unit vectors
a2 = np.linalg.norm(PC1)
print(a2)

###
'''
here I plot data in the coordinate system PC1-PC2
'''
###

print(XY)

XY_ = np.dot(np.transpose(XY), eigen_vecs[:, :])

fig2 = plt.figure()

plt.scatter(XY_[:, 0], XY_[:, 1])

ax = fig2.axes[0]
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('PC1')
plt.ylabel('PC2')
ax.set_aspect('equal', 'box')
plt.grid()
plt.savefig(os.path.join(OUT_PATH, 'PC1PC2.png'), dpi=600)
# plt.show()
