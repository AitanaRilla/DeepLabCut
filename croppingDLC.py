import numpy as npy
import cv2
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["DLClight"]="True"
os.environ["Colab"]="True"

import datetime
import deeplabcut
import pandas as pd
import tables


def rotated_coord(points,M):
    points = npy.array([points])
    points = npy.array(points)
    ones = npy.ones(shape=(len(points),1))
    points_ones = npy.concatenate((points,ones), axis=1)
    transformed_pts = M.dot(points_ones.T).T
    return transformed_pts[0]

def cut_videos(root_dir,file_dir, quality_position):

		try:
			positions = pd.read_hdf (root_dir+"/frames/"+file_dir+"/"+".".join(file_dir.split(".")[:-1])+"DLC_resnet50_corners_rataNov18shuffle1_1030000.h5",start=0)
		except:
			positions = pd.read_hdf (root_dir+"/frames/"+file_dir+"/"+file_dir+"DLC_resnet50_corners_rataNov18shuffle1_1030000.h5",start=0)
		
		file1 = positions.iloc[quality_position]
		R_up = file1[0],file1[1]
		L_up = file1[3],file1[4]
		R_down = file1[6],file1[7]
		L_down = file1[9],file1[10]
		dx = R_up[0]-L_up[0]
		dy = R_up[1]-L_up[1]
		inclination = npy.arctan2(dy,dx) * 180 / npy.pi
		print(root_dir+"/frames/"+file_dir+"/"+file_dir)
		img = cv2.imread(root_dir+"/frames/"+file_dir+"/"+file_dir+"_0_saved.jpg",0)
		rows,cols = img.shape
		M = cv2.getRotationMatrix2D((cols/2,rows/2),inclination,1)
		img_rotate = cv2.warpAffine(img,M,(cols,rows))
		
		frame = img_rotate
		L_up = rotated_coord(L_up,M)
		R_down = rotated_coord(R_down,M)

		frame1 = frame[int(L_up[1]):int(R_down[1]),int(L_up[0]):int(R_down[0])]

		cap = cv2.VideoCapture(root_dir+"/"+ file_dir+".mp4")
		length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		print("length: ",length )
		frame_width = frame1.shape[1]
		frame_height = frame1.shape[0]
		print("width and height: ",frame_width,frame_height)
		fps = cap.get(cv2.CAP_PROP_FPS)
		print("frames per second: ", fps)
		fourcc = cv2.VideoWriter_fourcc(*'m', 'p', '4', 'v')
		
		#Create the directories in analisis/ordenador
		new_root = "/".join(root_dir.split("/")[1:])
		if not os.path.exists("analisis/ordenador/"+new_root):
			os.makedirs("analisis/ordenador/"+new_root)


		route = "analisis/ordenador/"+"/".join(root_dir.split("/")[1:])+"/"+file_dir+".mp4"
		if not os.path.exists(route):

			out2 = cv2.VideoWriter(route, fourcc, int(fps), (frame_width,frame_height), True)

			for index in tqdm(range(int(length))):
				ret, frame = cap.read()
				img_rotate = cv2.warpAffine(frame,M,(cols,rows))
				frame = img_rotate[int(L_up[1]):int(R_down[1]),int(L_up[0]):int(R_down[0])]
				out2.write(frame)
			out2.release()
			cap.release()
			df_files.loc[(df_files["file"]== file_dir),"done"] = int(1)

def extract_quality(deeplabcut_route,screening_route,root_dir, file_dir):
	global positions
	deeplabcut.analyze_time_lapse_frames(deeplabcut_route, screening_route+root_dir+"/frames/"+file_dir+"/" ,frametype='.jpg',shuffle=1,trainingsetindex=0,gputouse=None,save_as_csv=True,rgb=True)
	try:
		positions = pd.read_hdf (root_dir+"/frames/"+file_dir+"/"+".".join(file_dir.split(".")[:-1])+"DLC_resnet50_corners_rataNov18shuffle1_1030000.h5",start=0)
	except:
		positions = pd.read_hdf (root_dir+"/frames/"+file_dir+"/"+file_dir+"DLC_resnet50_corners_rataNov18shuffle1_1030000.h5",start=0)
	print("Number of videos to analyze: ",len(positions.index))
	quality_max = []
	c = 0
	for image in positions.index:
		print(image)
		file1 = positions.loc[image]
		L_up = file1[3],file1[4]
		R_down = file1[6],file1[7]
		R_up = file1[0],file1[1]
		quality = (file1[5]+file1[8])/2
		if file1[6]-file1[3] < 100:
			quality = quality - 0.5
		if file1[7]-file1[4] < 100:
			quality = quality - 0.5
		quality_max.append((quality,c))
		c = c+1
	quality_max.sort()
	print(quality_max[-1])
	df_files.loc[(df_files["file"]== file_dir),"quality"] = float(quality_max[-1][0])
	df_files.loc[(df_files["file"]== file_dir),"quality_position"] = float(quality_max[-1][1])


now = datetime.datetime.now()

lsFiles = []


lsDir = os.walk(os.path.normpath("temp/CARAS/"))
for root, dirs, files in lsDir:
    for file in files:
        (filename, extension) = os.path.splitext(file)
        if extension == ".mp4" :
            lsFiles.append((filename,root,0,0,0,0))


df_files = pd.DataFrame(lsFiles, columns=['file', 'root','extracted','done','quality',"quality_position"])

#Don't evaluate previously evaluated directories
doneDir =[]
lsDir = os.walk(os.path.normpath("temp/CARAS/"))
for root, dirs, files in lsDir:
    for file in files:
        if file == "done.txt" :
            doneDir.append((root))

for i in range(len(df_files.index)):
    if df_files["root"][i] in doneDir:
        df_files.loc[i,"done"] = 1

df_done = df_files[df_files["done"]==0]
                   
print("files to analyze: ", len(df_done["file"]))
print("creating new directories: ")
for n in range(len(df_files["file"])):
	if os.path.exists(df_files["root"][n]+"/frames/"+df_files["file"][n]+"/"):
		df_files.iloc[n,2] = 1

	if df_files["done"][n]==0:
		try:
			os.mkdir(df_files["root"][n]+"/frames/")
		except:
			pass
		try:	
			print(os.mkdir(df_files["root"][n]+"/frames/"+df_files["file"][n]+"/"))
			os.mkdir(df_files["root"][n]+"/frames/"+df_files["file"][n]+"/")
		except:
			pass	
#Process each file on the list	
print("Extracting frames to analyze corners...")
for i in range(len(df_files["extracted"])):
	if df_files["extracted"][i] == 0:
		path = (df_files["root"][i])+"/"
		file = df_files["file"][i]
		print(path+file)

		cap = cv2.VideoCapture(path+ file+".mp4")
		length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		frame_width = int(cap.get(3))
		frame_height = int(cap.get(4))
		fps = cap.get(cv2.CAP_PROP_FPS)
			
	#Extract a frame to search for corners
		
		for index in tqdm(range(int(length/4))):
			ret, frame = cap.read()
			if (index % int((length/4)/10)) == 0:
				cv2.imwrite(filename= path +"frames/" +  file +"/"+file+ "_" + str(index) + "_" + "saved.jpg",img=frame)
		        # Break the loop
			else:
				pass
		        #When everything's done, release the video capture and video write objects
		cap.release()
		df_files["extracted"][i] = 1
		df_files.to_csv("log.csv", sep=',', encoding='utf-8')
	else:
		pass
#In each directory
deeplabcut_route = "/Users/aitanarilla/Desktop/AitanaRatsDLCTest-2022-11-23/config.yaml"
screening_route = "/Users/aitanarilla/Desktop/AitanaRatsDLCTest-2022-11-23/CIA/"

print("Finding corners...")
for n in range(len(df_files["file"])):
	if df_files["done"][n]==0:
		extract_quality(deeplabcut_route,screening_route,df_files["root"][n],df_files["file"][n])

df_files.to_csv("log.csv", sep=',', encoding='utf-8')

print("Creating cropped videos...")
for n in range(len(df_files["quality"])):
	if 	df_files["done"][n]==0:	
		if float(df_files["quality"][n]) > 0.8:
			print("Cropping video...")
			cut_videos(df_files["root"][n],df_files["file"][n],int(df_files["quality_position"][n]))
			df_files["done"][n] == 1
		else:
			print("Insufficient quality")
			
			pass	

		df_files.to_csv("log.csv", sep=',', encoding='utf-8')