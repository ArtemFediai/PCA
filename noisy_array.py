import numpy as np


class NoisyArray(np.ndarray):
    def __new__(cls, input_array, scale=1, *args, **kwargs):
        obj = np.asarray(input_array).view(cls) + np.random.random_sample(input_array.__len__())*scale
        return obj

X = NoisyArray([1.0, 2.0, 3.0], dtype=float)


print(X)
print(type(X))

print('\nX=\n', X)