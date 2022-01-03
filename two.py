"""
two-dimensional feature space
linearly correlated points with a random component
"""

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
XY = np.array([X, Y])
#XYmean = [np.mean(X), np.mean(Y)]
#print(f'mean {XYmean}')
print('\nX, Y array:\n', np.array(XY))
cov_mat = np.cov(XY)
print('\ncovariance matrix: \n', cov_mat)

# Compute PCs as the eigen-things of the covariance matrix
eigen_vals_non_sorted, eigen_vecs_non_sorted = np.linalg.eig(cov_mat)

# 03012021

# e_vec_1 = eigen_vecs_non_sorted[:,0]
# e_vec_2 = eigen_vecs_non_sorted[:,1]

# print('stop')

idx = np.flip(np.argsort(eigen_vals_non_sorted))
eigen_vals = eigen_vals_non_sorted[idx]
eigen_vecs = eigen_vecs_non_sorted[:,idx]
print('\nPC eigenvalues: \n', eigen_vals)
print('\nPC eigenvectors: \n', eigen_vecs)

# Plot PCs at the plot of the original data
ax = original_data.axes[0]
ax.set_aspect('equal', 'box')
ax.arrow(0, 0, *eigen_vecs[:,0],
         head_width=0.2, head_length=0.3, width=0.05, color='red', label='PC1')  # PC1
ax.arrow(0, 0, *eigen_vecs[:,1],
         head_width=0.2, head_length=0.3, width=0.05, color='blue', label='PC2')  # PC2
# plt.grid()
plt.legend()
# plt.tight_layout()
plt.savefig(os.path.join(OUT_PATH, 'PCA_2D.png'), dpi=600)
plt.close()

# check if orthogonal:
PC1 = eigen_vecs[:,0]
PC2 = eigen_vecs[:,1]
res = np.dot(PC1, PC2)
print(f'PC1 (vector): {PC1}')
print(f'PC2 (vector): {PC2}')
print(f'PC1 (scalar): {eigen_vals[0]}')
print(f'PC2 (scalar): {eigen_vals[1]}')
print(res)

# check if unit vectors
a2 = np.linalg.norm(PC1)
print(a2)

###
'''
here I plot data in the coordinate system PC1-PC2
'''
###

# print(XY)

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
plt.close(fig2)
# plt.show()

# print diagram
fig3 = plt.figure()
plt.bar(['PC1', 'PC2'], eigen_vals)
plt.savefig(os.path.join(OUT_PATH, 'PC_values.png'), dpi=600)
plt.close(fig3)
# check
product1 = np.matmul(cov_mat, PC1)  # sorting fails dramatically
product2 = eigen_vals[0] * PC1

print(product1, product2)

'''
here I plot reconstructed things using only the first PC
'''
fig2_ = plt.figure(figsize=[4,4])

WL_transpose = eigen_vecs[:, 0]
WL_normal = eigen_vecs[:, 0][np.newaxis]  # restricted eigenvectors
XY_reconstructed = np.matmul(np.transpose(XY), np.outer(WL_transpose, WL_normal))  # X * WL^T * WL
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('PC1')
plt.ylabel('PC2')
ax.set_aspect('equal', 'box')
plt.grid()
plt.scatter(XY_reconstructed[:, 0], XY_reconstructed[:,1])
plt.savefig('output/XY_reconstructed_1stPC.png', dpi=600)
plt.close(fig2_)

##################### TMP PART #########################################################################################
######################## REMOVE IT #####################################################################################
########################### IF YOU WANT TO HAVE PRODUCTION CODE ########################################################
# random

X = np.random.random_sample(20)
Y = np.random.random_sample(20)
X = (X - np.mean(X))/np.std(X)
Y = (Y - np.mean(Y))/np.std(Y)
plt.scatter(X, Y)
plt.show()
print(X, Y)
XY = np.array([X, Y])
cov_mat = np.cov(XY)
eigen_vals_non_sorted, eigen_vecs_non_sorted = np.linalg.eig(cov_mat)
idx = np.flip(np.argsort(eigen_vals_non_sorted))
eigen_vals = eigen_vals_non_sorted[idx]
eigen_vecs = eigen_vecs_non_sorted[idx]

print('\nPC eigenvalues: \n', eigen_vals)
print('\nPC eigenvectors: \n', eigen_vecs)

# Plot PCs at the plot of the original data
ax = original_data.axes[0]
ax.set_aspect('equal', 'box')
ax.arrow(0, 0, *eigen_vecs[0],
         head_width=0.2, head_length=0.3, width=0.05, color='red')  # PC1
ax.arrow(0, 0, *eigen_vecs[1],
         head_width=0.2, head_length=0.3, width=0.05, color='blue')  # PC2
plt.savefig(os.path.join(OUT_PATH, 'random_PCA_2D.png'), dpi=600)
plt.close()

# print diagram
fig3 = plt.figure()
plt.bar(['PC1', 'PC2'], eigen_vals)
plt.savefig(os.path.join(OUT_PATH, 'random_PC_values.png'), dpi=600)
plt.close(fig3)