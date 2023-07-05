import pandas as pd
import numpy as np


df = pd.read_csv('skopt-diva_shj-0109-1453.csv')


getRidOfOpenBracket = df['[learn_rate, num_hidden_nodes, weight_range, beta]'].str.split("[", n = -1, expand = True)
df['this'] = getRidOfOpenBracket[0]
df['that'] = getRidOfOpenBracket[1]
df.drop(columns =["[learn_rate, num_hidden_nodes, weight_range, beta]"], inplace = True)
df.drop(columns =["this"], inplace = True)




getRidOfCloseBracket = df['that'].str.split("]", n = -1, expand = True)
df['newthat'] = getRidOfCloseBracket[0]
df.drop(columns =["that"], inplace = True)



splitTheData = df['newthat'].str.split(",", n = -1, expand = True)

df['learn_rate'] = splitTheData[0]
df['num_hidden_nodes'] = splitTheData[1]
df['weight_range'] = splitTheData[2]
df['beta'] = splitTheData[3]
df['c'] = splitTheData[4]

df.drop(columns =["newthat"], inplace = True)







df.to_csv('logs/splitData.csv',index=False)

