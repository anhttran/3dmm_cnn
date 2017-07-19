#############################################################################
#Copyright 2016-2017, Anh Tuan Tran, Tal Hassner, Iacopo Masi, and Gerard Medioni
#The SOFTWARE provided in this page is provided "as is", without any guarantee
#made as to its suitability or fitness for any particular use. It may contain
#bugs, so use of this tool is at your own risk. We take no responsibility for
#any damage of any sort that may unintentionally be caused through its use.
# Please, cite the paper:
# @article{tran16_3dmm_cnn,
#   title={Regressing Robust and Discriminative {3D} Morphable Models with a very Deep Neural Network},
#   author={Anh Tran 
#       and Tal Hassner 
#       and Iacopo Masi
#       and G\'{e}rard Medioni}
#   journal={arXiv preprint},
#   year={2016}
# }
# if you find our code useful.
##############################################################################

import os
## Tu suppress the noise output of Caffe when loading a model
## polish the output (see http://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe)
os.environ['GLOG_minloglevel'] = '2' 

import numpy as np
from PIL import Image
from glob import glob
import caffe
import cv2
import time
import ntpath
import os.path
import scipy.io
import shutil
import sys
from skimage import io
import dlib
from Tkinter import Tk
from tkFileDialog import askopenfilename
import subprocess
import utils

# --------------------------------------------------------------------------- #
# Usage: python testModel_PoseExpr.py <outputDir>  [<save3D>]
# --------------------------------------------------------------------------- #
# CNN network spec
deploy_path = '../CNN/deploy_network.prototxt'
model_path  = '../CNN/3dmm_cnn_resnet_101.caffemodel'
mean_path = '../CNN/mean.binaryproto'
layer_name      = 'fc_ftnew'
## Modifed Basel Face Model
BFM_path = '../3DMM_model/BaselFaceModel_mod.mat'
#GPU ID we want to use
#GPU_ID = 0
## CNN template size
trg_size = 224

#################################### Initiate
predictor_path = "../dlib_model/shape_predictor_68_face_landmarks.dat";
if len(sys.argv) < 2:
		print "Usage: python testModel_PoseExpr.py <outputDir> [<save3D>]"
		exit(1)

Tk().withdraw() ## we don't want a full GUI, so keep the root window from appearing
image_path = askopenfilename(initialdir='data',filetypes=[("Image Files",("*.jpg","*.png"))]) # show an "Open" dialog box and return the path to the selected file

data_out = sys.argv[1];
output3D = 1
if len(sys.argv) > 2:
	output3D = int(sys.argv[2]);
if os.path.exists("tmp_ims"):
	shutil.rmtree('tmp_ims');
os.makedirs("tmp_ims");
if not os.path.exists(data_out):
	os.makedirs(data_out);

if os.path.exists("tmp_detect"):
	shutil.rmtree('tmp_detect');
os.makedirs("tmp_detect");

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

################################### Prepare images
print("> Prepare image "+image_path + ":")
impath_noext = image_path.split(image_path.split('.')[-1])[0][0:-1];
imname = ntpath.basename(image_path)
imname = imname.split(imname.split('.')[-1])[0][0:-1]
start_time = time.time()
img = cv2.imread(image_path);
if not os.path.exists(impath_noext + ".pts"):
		dlib_img = io.imread(image_path)
		img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
		# Detect face
    		dets = detector(img, 1)
    		print("    Number of faces detected: {}".format(len(dets)))
		if len(dets) == 0:
			print "    Error! No face detected!"
			quit();
		if len(dets) > 1:
			print "    Process only the first detected face!"
    		detected_face = dets[0];
		cv2.rectangle(img2, (detected_face.left(),detected_face.top()), (detected_face.right(),detected_face.bottom()), (0,0,255),2);
		fileout = open("tmp_detect/"+imname+".bbox","w");
		fileout.write("%f %f %f %f\n" % (detected_face.left(),detected_face.top(), detected_face.right(),detected_face.bottom()));
		fileout.close();
		# Detect landmarks
        	shape = predictor(dlib_img, detected_face)
		nLM = shape.num_parts
		fileout = open("tmp_detect/"+imname+".pts","w");
		for i in range(0,nLM):
			cv2.circle(img2, (shape.part(i).x, shape.part(i).y), 1, (255,0,0));
			fileout.write("%f %f\n" % (shape.part(i).x, shape.part(i).y));
		fileout.close();
		# Crop image
		img = utils.cropByLM(img, shape, img2)
		cv2.imwrite("tmp_detect/"+imname+"_detect.png",img2);
else:
		# Load landmarks
		lms = np.loadtxt(impath_noext + ".pts")
		img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
		for i in range(0,68):
			cv2.circle(img2, (int(lms[i,0]), int(lms[i,1])), 1, (255,0,0));
		cv2.imwrite("tmp_detect/"+imname+"_detect.png",img2);
		# Crop image
		img = utils.cropByInputLM(img, lms, img2);

img = cv2.resize(img,(trg_size, trg_size));
cv2.imwrite("tmp_ims/" + imname + ".png",img);
imPrepare_time = time.time() - start_time

################################### CNN fitting
# load net 
caffe.set_mode_cpu();
#caffe.set_device(GPU_ID);
start_time = time.time()
## Opening mean average image
proto_data = open(mean_path, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]

## Loading the CNN
net = caffe.Classifier(deploy_path, model_path);
## Setting up the right transformer for an input image
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
transformer.set_mean('data',mean)
print '> CNN Model loaded to regress 3D Shape and Texture!'
## Loading the Basel Face Model to write the 3D output
if output3D > 0:
	model = scipy.io.loadmat(BFM_path,squeeze_me=True,struct_as_record=False)
	model = model["BFM"]
	faces = model.faces-1
	print '> Loaded the Basel Face Model to write the 3D output!'
init_time = time.time() - start_time

net.blobs['data'].reshape(1,3,trg_size,trg_size)
start_time = time.time()
im = caffe.io.load_image(image_path)
## Transforming the image into the right format
net.blobs['data'].data[...] = transformer.preprocess('data', im)
## Forward pass into the CNN
net_output = net.forward()
## Getting the output
features = np.hstack( [net.blobs[layer_name].data[0].flatten()] )
modeling_time = time.time() - start_time
## Writing the regressed 3DMM parameters
outFile = data_out + "/" + imname
np.savetxt(outFile + '.ply.alpha', features[0:99])
np.savetxt(outFile + '.ply.beta', features[99:198])
print "*****************************************"
print "** Caffe loading    : %.3f s" % init_time
print "** Image cropping   : %.3f s" % imPrepare_time
print "** 3D Modeling      : %.3f s" % modeling_time
print "*****************************************"
#################################
## Mapping back the regressed 3DMM into the original
## Basel Face Model (Shape)
##################################
if output3D > 0:
	S,T = utils.projectBackBFM(model,features)
	print '> Writing 3D file in: ', outFile + '.ply'
	utils.write_ply(outFile + '.ply', S, T, faces)
## Pose + expression fitting and visualization
print "> Pose & expression estimation"
if not os.path.exists(impath_noext + ".pts"):
	os.system("cd ../bin; ./TestVisualization " + image_path + " ../demoCode/" + outFile + ".ply.alpha ../3DMM_model/BaselFaceModel_mod.h5 ../dlib_model/shape_predictor_68_face_landmarks.dat ../demoCode/tmp_detect/" + imname + ".pts; cd ../demoCode");
else:
	os.system("cd ../bin; ./TestVisualization " + image_path + " ../demoCode/" + outFile + ".ply.alpha ../3DMM_model/BaselFaceModel_mod.h5 ../dlib_model/shape_predictor_68_face_landmarks.dat " + impath_noext + ".pts; cd ../demoCode");

