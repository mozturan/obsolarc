import numpy as np

x = np.zeros((1,11))

print(x.shape)
t= np.array([1])
x = np.concatenate((x, [t]), axis=0)

print(x.shape)
print(x)