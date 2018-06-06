import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(-2,2)
x2 = np.linspace(-2,0)
x3 = np.linspace(0,2)

y = np.linspace(0,1)

# step print_function

y1 = [0 for i in x2]
y2 = [1 for i in x3]
#
#

plt.subplot(1, 3, 1)
plt.plot(x2, y1,'b')
plt.plot(x3,1*x3,'b')
plt.grid()
plt.xlim(-2,2)
plt.ylim(-1.5,1.5)
plt.xlabel('x')
plt.ylabel('$\phi (x)$')
plt.subplot(1,3,1).set_title('ReLu Function')



plt.subplot(1, 3, 2)
plt.plot(x1,1/(1+np.exp(-5*x1)),'b')
plt.grid()
plt.xlim(-2,2)
plt.ylim(-1.5,1.5)
plt.xlabel('x')
plt.ylabel('$\phi (x)$')
plt.subplot(1,3,2).set_title('Sigmoid Function')

plt.subplot(1, 3, 3)
plt.plot(x2, y1,'b')
plt.plot(x3,y2,'b')
plt.plot([0 for i in y],y,'b')
plt.grid()
plt.xlim(-2,2)
plt.ylim(-1.5,1.5)
plt.xlabel('x')
plt.ylabel('$\phi (x)$')
plt.title('Common Activation Functions')
plt.subplot(1,3,3).set_title('Step Function')



# plt.ylabel('Damped oscillation')
#
#
#
#
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, 'r.-')
# plt.xlabel('time (s)')
# plt.ylabel('Undamped')
plt.show()
