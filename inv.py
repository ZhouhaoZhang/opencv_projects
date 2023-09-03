import numpy as np

P_I = np.matrix([[0, 1, 4, 0], [1, 2, 4, 0], [1, -1, 1, 0], [1, -1, 1, 1]], dtype="double")
P = P_I.I
A = np.matrix([[2, 1, 0, 0], [0, 2, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype="double")
b = np.matrix([[0, 1, 1, 1]], dtype="double").T
A_ = P * A * P_I
k_ = np.matrix([[9, 39, 121,0]],dtype="double")
k = k_*P
print(k)