import os
import csv

import pickle
import numpy as np
np.random.seed(42)

from tqdm import tqdm

gnd_dict = { 'gnd':[ {'bbx':[1197, 331, 3610, 2650], 'easy':[0,1], 'hard':[2,3,4,5,6], 'junk':[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] },
{'bbx':[1198, 911, 1418, 1105], 'easy':[7,8,9,10,11,12,13,14,15,16,17], 'hard':[18,19], 'junk':[0,1,2,3,4,5,6,20,21,22,23,24,25]},
{'bbx':[100, 577, 935, 1381], 'easy':[15,16,17,20,21], 'hard':[], 'junk':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,22,23,24,25]},
{'bbx':[782, 330, 1156, 675], 'easy':[23,24,25], 'hard':[22], 'junk':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,1415,16,17,18,19,20,21]}],
'imlist':['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '20', '21', 'frame0023', 'frame0024', 'frame0028', 'frame0029', 'frame0054', 'bIwFrmlseFc', '4doj9tH2gqA', 'q3A00VrROrE', 'FMbUERTPFlc'],
'qimlist':['1', '0', 'frame0000', 'hVTCbdg4sr8']}

save_path = './gnd_gl18.pkl'

pickle.dump(gnd_dict, open(save_path, 'wb'))
