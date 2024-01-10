import numpy as np
a = np.array([[1,2],[3,4],[5,6]])
c = a[(a>3)&(a<11)]
print(c)