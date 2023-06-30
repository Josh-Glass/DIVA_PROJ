import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv('output_shj1.csv')
df2 = pd.read_csv('output_shj2.csv')
df3 = pd.read_csv('output_shj3.csv')
df4 = pd.read_csv('output_shj4.csv')
df5 = pd.read_csv('output_shj5.csv')
df6 = pd.read_csv('output_shj6.csv')


shj1 = np.array(df1['data'])
shj2 = np.array(df2['data'])
shj3 = np.array(df3['data'])
shj4 = np.array(df4['data'])
shj5 = np.array(df5['data'])
shj6 = np.array(df6['data'])

behavioral_all_structures = 1 - np.genfromtxt('data/shj/behavioral_nosofsky1994.csv', delimiter = ',')

human1 = behavioral_all_structures[0:16,[0]].reshape(16, 1)
human2 = behavioral_all_structures[0:16,[1]].reshape(16, 1)
human3 = behavioral_all_structures[0:16,[2]].reshape(16, 1)
human4 = behavioral_all_structures[0:16,[3]].reshape(16, 1)
human5 = behavioral_all_structures[0:16,[4]].reshape(16, 1)
human6 = behavioral_all_structures[0:16,[5]].reshape(16, 1)




block = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])


plt.plot(block, shj1, c='r', marker='s',label='shj1')
plt.plot(block, human1, c='r',linestyle='dashed', linewidth=3, alpha=0.4)

plt.plot(block, shj2, c='b', marker='*',label='shj2')
plt.plot(block, human2, c='b',linestyle='dashed', linewidth=3, alpha=0.4)

plt.plot(block, shj3, c='k', marker='o',label='shj3')
plt.plot(block, human3, c='k',linestyle='dashed', linewidth=3, alpha=0.4)

plt.plot(block, shj4, c='y', marker='^',label='shj4')
plt.plot(block, human4, c='y',linestyle='dashed', linewidth=3, alpha=0.4)

plt.plot(block, shj5, c='g', marker='X',label='shj5')
plt.plot(block, human5, c='g',linestyle='dashed', linewidth=3, alpha=0.4)

plt.plot(block, shj6, c='orange', marker='P',label='shj6')
plt.plot(block, human6, c='orange',linestyle='dashed', linewidth=3, alpha=0.4)







plt.title('DIVA performance on shj (about 0.1 SSD)')
plt.xlabel('learning block')
plt.ylabel('accuracy')
plt.legend()
plt.savefig("graphed_output.png")






