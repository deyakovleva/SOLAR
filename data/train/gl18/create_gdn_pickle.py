import os
import csv

import pickle
import numpy as np
np.random.seed(42)

from tqdm import tqdm

gnd_dict = { 'gnd':[ {'bbx':[341, 890, 935, 1457], 'easy':[1], 'hard':[2], 'junk':[2] }, 
{'bbx':[1198, 911, 1418, 1105], 'easy':[2], 'hard':[1], 'junk':[1]}], 
'imlist':['0', '1', '2', '3', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '19', '20', '21'], 
'qimlist':['4', '18']}


save_path = './gnd_gl18.pkl'

pickle.dump(gnd_dict, open(save_path, 'wb'))
