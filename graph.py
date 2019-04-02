import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


import pickle


df = pd.read_csv('graph.csv')
acc = df['Accuracy']
print(np.mean(acc))
n1 = df['N_1']
n2 = df['N_2']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.bar3d(acc, n1, n2, c='r')

ax.scatter(acc, n1, n2, c='r', marker='o')
ax.set_xlabel('Accuracy')
ax.set_ylabel('n_1')
ax.set_zlabel('n_2')
fig.savefig('70-30.png')


fig2, bx = plt.subplots()
bx.scatter(acc,n1,c = 'b', marker = 'x')
bx.set_xlabel('Accuracy')
bx.set_ylabel('n_1')

fig3, cx = plt.subplots()
cx.scatter(acc,n2,c = 'g', marker = 'x')
cx.set_xlabel('Accuracy')
cx.set_ylabel('n_2')




#pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))
#figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))
#figx.show()

plt.show()