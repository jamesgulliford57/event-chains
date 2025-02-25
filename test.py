import numpy as np 
import matplotlib.pyplot as plt

x_data = np.array([[1,2,3],
                   [1,2,3]])
y_data = np.array([[1,2,3],
                   [1,2,3]])
x = np.array([[0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5]])

y_interp = np.interp(x, x_data, y_data)
print(y_interp)

#plt.scatter(x_data, y_data, color='r')
#plt.scatter(x, y_interp, color='b')
#plt.show()

