import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os


IMG_DIR_1 = '/home/toytiny/Desktop/RadarNet2/figures/mini_val/'   
out_path='/home/toytiny/3dd/'
out_put_video_name = 'bev_val.avi' 

IMG_DIR = [IMG_DIR_1]
file_type = '.jpg'
fps = 4 # speed*5  
size=(640,640) # 
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

for dir_num in range(0,len(IMG_DIR)):
    IMG_DIR_temp = IMG_DIR[dir_num]
    out_video_file_name = out_path+out_put_video_name 
    if not os.path.exists(out_path):
    	os.mkdir(out_path)
    videoWriter = cv2.VideoWriter(out_video_file_name,fourcc,fps,size)
    image_list = [f for f in os.listdir(IMG_DIR_temp) if f.endswith(file_type)]
    image_list.sort(key= lambda x:int(x[17:18]))
    for i in range(0, len(image_list)):
        frame = cv2.imread(IMG_DIR_temp+image_list[i])
        videoWriter.write(frame)
    videoWriter.release()
