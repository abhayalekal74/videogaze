import face_recognition
import cv2
import pylab
import imageio
from cv2 import CascadeClassifier
import sys
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from GazeFollowing.videogaze_model import VideoGaze
import cv2
import math
from sklearn import metrics
from mtcnn.mtcnn import MTCNN


def create_dir_if_not_exists(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)


def get_dest_from_video_name(video_name):
	return video_name[:video_name.rindex('.')].replace('.', '-')


def download_video_frames(video_url):
	video_name = video_url[video_url.rindex('/') + 1: ]
	dest = get_dest_from_video_name(video_name) 
	os.system('sh download_video_frames.sh {} {} {} {}'.format(video_path, video_name, dest, fps))
	return dest


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--video-file', dest="videofile", help="Video file if already downloaded")
	parser.add_argument('--video-url', dest="videourl", help="Video URL")
	parser.add_argument('--fps', default=1, help="FPS for sampling input video")
	parser.add_argument('--dest', help="Save video frames in this directory")
	args = parser.parse_args()
	if not args.videofile and not args.videourl:
		if not args.dest:
			sys.exit("Error: Provide video file or video url")
	return args

if __name__ == '__main__':
	args = parse_args()
	model = VideoGaze(bs=32,side=20)
	checkpoint = torch.load('model.pth.tar', map_location=torch.device('cpu'))
	model.load_state_dict(checkpoint['state_dict'])
	cudnn.benchmark = True

	fps = args.fps 
	dest = args.dest
	if args.videofile:
		if not dest:
			dest = get_dest_from_video_name(args.videofile)
		create_dir_if_not_exists(dest)
		os.system("ffmpeg -i {} -f image2 -r {} {}/frame%09d.png".format(args.videofile, fps, dest))
	elif args.videourl:
		extracted_dest = download_video_frames(args.videourl)
		if not dest:
			dest = extracted_dest

	if not dest:
		sys.exit("Required parameters missing. Pass --video-file and --dest or --video-url or --dest to process frames")

	frame_list = []
	for f in os.listdir(dest):
		frame_list.append(cv2.imread(os.path.join(dest, f)))

	if not frame_list:
		sys.exit("Error: No frames to read")

	trans = transforms.ToTensor()

	#N corresponds to the number of frames in the window to explore
	N = 25

	#w_T corresponds to the number of frames to skip when sampling the target window
	w_T = 40

	w_fps = 1

	target_frame = torch.FloatTensor(N,3,227,227)

	face_frame = torch.FloatTensor(N,3,227,227)

	eyes = torch.zeros(N,3)

	x_offset, y_offset, x_center, y_center, person_looking_at_camera, frames_processed, frames_without_face = 0, 0, 0, 0, 0, 0, 0

	for i in range(len(frame_list)):
		print('Processing of frame %d out of %d' % (i,len(frame_list)))
		if i>w_fps*(N-1)//2 and i<(len(frame_list)-w_fps*(N-1)//2):
			#Avoid the problems with the video limit
			#Reading the image 
			frames_processed += 1
			top=False
			im = frame_list[i]
			h,w,c = im.shape

			face_locations = face_recognition.face_locations(im)
			for id,face_local in enumerate(face_locations):
				(top, right, bottom, left) = face_local

			#If detection, run the model
			if top:

				#Crop Face Image 
				crop_img = im[top:bottom, left:right]
				crop_img = cv2.resize(im,(227,227))

				#Resize Image   
				im = cv2.resize(im,(227,227))

				#Compute the center of the head and estimate the eyes location
				eyes[:,0] = (right+left)/(2*w)
				eyes[:,1] = (top+bottom)/(2*h)

				#Fill the tensors for the exploring window. Face and source frame are the same
				source_frame = trans(im).view(1,3,227,227)
				face_frame = trans(crop_img).view(1,3,227,227)
				for j in range(N-1):
					trans_im = trans(im).view(1,3,227,227)
					source_frame = torch.cat((source_frame,trans_im),0)
					crop_im = trans(crop_img).view(1,3,227,227)
					face_frame = torch.cat((face_frame,crop_im),0)

				#Fill the targets for the exploring window. 
				for j in range(N):
					target_im = frame_list[i+w_fps*(j-((N-1)//2))]
					#target_im = frame_list[i]
					target_im = cv2.resize(target_im,(227,227))
					target_im = trans(target_im)
					target_frame[j,:,:,:] = target_im
				
				#Run the model
				source_frame_var = torch.autograd.Variable(source_frame)
				target_frame_var = torch.autograd.Variable(target_frame)
				face_frame_var = torch.autograd.Variable(face_frame)
				eyes_var = torch.autograd.Variable(eyes)
				output,sigmoid= model(source_frame_var,target_frame_var,face_frame_var,eyes_var)

				#Recover the data from the variables
				sigmoid = sigmoid.data
				output = output.data

				#Pick the maximum value for the frame selection
				v,ids = torch.sort(sigmoid, dim=0, descending=True)
				index_target = ids[0,0]

				#Pick the frames corresponding to the maximum value
				target_im = frame_list[i+w_fps*(index_target-((N-1)//2))].copy()
				#target_im = frame_list[i].copy()
				output_target = cv2.resize(output[index_target,:,:,:].view(20,20).cpu().numpy(),(200,200))
				
				#Compute the gaze location
				map = np.reshape(output_target,(200*200))

				int_class = np.argmax(map)
				x_class = int_class % 200
				y_class = (int_class-x_class)//200
				y_float = y_class/200.0
				x_float = x_class/200.0
				x_point = math.floor(x_float*w)
				y_point = math.floor(y_float*h)
				
				if not x_offset:
					x_offset = math.floor(target_im.shape[1] * 0.1)
					y_offset = math.floor(target_im.shape[0] * 0.15)
					x_center = target_im.shape[1] // 2
					y_center = target_im.shape[0] // 2

				#Check if the person is looking at the center of the image (normally the position of the camera), looking in the line of face locals
				if (x_point >= x_center - x_offset and x_point <= x_center + x_offset and y_point >= y_center - y_offset and y_point <= y_center + y_offset) or (x_point >= left and x_point <= right and y_point >= top): 
					person_looking_at_camera += 1

				print("person_looking {}, x_class {}, y_class {}, x_float {}, y_float {}, x_point {}, y_point {}, target_im {}".format(person_looking_at_camera, x_class, y_class, x_float, y_float, x_point, y_point, target_im.shape))
			else:
				frames_without_face += 1
	if frames_processed > 0:
		print ("Person looking at the camera in {}% frames".format(math.ceil((person_looking_at_camera / frames_processed) * 100)))
		print ("{}% of frames did not have a face".format(math.ceil((frames_without_face / frames_processed) * 100)))
	else:
		print ("Error: No frames to process")
