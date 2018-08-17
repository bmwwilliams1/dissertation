import numpy as np
import matplotlib
import matplotlib.pyplot as plt

w1 = np.genfromtxt ('./weights_3/weights_16_1.csv', delimiter=",")
print(w1)

for row in range(0,len(w1)):
    for col in range(0,len(w1[0])):
        if w1[row][col] < 0:
            w1[row][col] = 0

print(w1)

np.savetxt("./weights_3/weights_16_1_zero.csv",w1, delimiter=",")
