import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math

list_file = os.listdir('results/')

aaa = []
for i in list_file:
    fff = np.load('results/' + str(i))
    aaa.append(fff)
aaa = np.mean(aaa, axis = 0)
aaa = np.greater_equal(aaa,0.448)   #0.448
aaa = aaa.astype(int)

np.save('adj_matrix.npy', np.array(aaa))


