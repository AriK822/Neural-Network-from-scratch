import numpy as np

a = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])
a[0:2, 1:] = [[-1, -2], [-3, -4]]


print(a)
