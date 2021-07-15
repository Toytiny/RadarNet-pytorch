import os 
import numpy as np
import math
import sys
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
palette = plt.get_cmap('Set1')
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300 
loss_path="/home/toytiny/Desktop/RadarNet2/loss.txt"
save_path="/home/toytiny/Desktop/RadarNet2/loss_figure/"
if not os.path.exists(save_path):
    os.mkdir(save_path)
loss_iter=[]
loss_epoch=[]
ap_epoch=[]
with open(loss_path,'r') as f:
    for line in f:
    
        if line[0:16]=='The loss of iter':
            this_iter=float(line.split('is')[1].split('The')[0])
            loss_iter.append(this_iter)
        if line[0:17]=='The loss of epoch':
            this_epoch=float(line.split('is')[1])
            loss_epoch.append(this_epoch)
        if line[0:15]=='The AP of epoch':
            ap=float(line.split('is')[1])
            ap_epoch.append(ap)
            
print('Finishing reading')
plt.clf()
plt.cla()

plt.figure(1)
plt.plot(loss_iter)
plt.xlabel('iteration')
plt.ylabel('loss')
#plt.ylim(-5,320)
plt.savefig(save_path+'loss_iteration.jpg')

plt.figure(2)
plt.plot(loss_epoch)
plt.xlabel('epoch')
plt.ylabel('loss')
#plt.ylim(-5,160)
plt.savefig(save_path+'loss_epoch.jpg')

plt.figure(3)
plt.plot(ap_epoch)
plt.xlabel('epoch')
plt.ylabel('Average Precision@0.5')
#plt.ylim(0,1)
plt.savefig(save_path+'AP_epoch.jpg')


