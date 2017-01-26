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
###################
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
import utils
# --------------------------------------------------------------------------- #
# Usage: python demo.py <inputList> <outputDir> <needCrop> <useLM>
# --------------------------------------------------------------------------- #
# CNN network spec
deploy_path = 'CNN/deploy_network.prototxt'
model_path  = 'CNN/3dmm_cnn_resnet_101.caffemodel'
mean_path = 'CNN/mean.binaryproto'
layer_name      = 'fc_ftnew'
#GPU ID we want to use
GPU_ID = 0	
## Modifed Basel Face Model
BFM_path = './BaselFaceModel_mod.mat'
## CNN template size
trg_size = 224
#### Initiate ################################
predictor_path = "dlib_model/shape_predictor_68_face_landmarks.dat"
if len(sys.argv) < 3 or len(sys.argv) > 5 :
		print "Usage: python demo.py <inputList> <outputDir> <needCrop> <useLM>"
		exit(1)
fileList = sys.argv[1]
data_out = sys.argv[2]
needCrop = 1
if len(sys.argv) > 3:
	needCrop = bool(sys.argv[3]=='1')
useLM    = 1
if len(sys.argv) > 4:
	useLM    = bool(sys.argv[4]=='1')
if os.path.exists("tmp_ims"):
	shutil.rmtree('tmp_ims')
os.makedirs("tmp_ims")
if not os.path.exists(data_out):
	#shutil.rmtree('tmp_model')
	os.makedirs(data_out)
if needCrop:
	detector = dlib.get_frontal_face_detector()
	if os.path.exists("tmp_detect"):
		shutil.rmtree('tmp_detect')
	os.makedirs("tmp_detect")
	if useLM:
		predictor = dlib.shape_predictor(predictor_path)

##### Prepare images ##############################
with open(fileList, "r") as ins:
    for image_path in ins:
	if len(image_path) < 6:
		print 'Skipping ' + image_path + ' file path too short'
		continue
	image_path = image_path[:-1]
	print("> Prepare image "+image_path + ":")
	imname = ntpath.basename(image_path)
	#imname = imname[:-4]
	imname = imname.split(imname.split('.')[-1])[0][0:-1]
	img = cv2.imread(image_path)
	## If we do cropping on the image
	if needCrop:
		dlib_img = io.imread(image_path)
		img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
		dets = detector(img, 1)
		print(">     Number of faces detected: {}".format(len(dets)))
		if len(dets) == 0:
			print '> Could not detect the face, skipping the image...' + image_path
			continue
		if len(dets) > 1:
			print "> Process only the first detected face!"
		detected_face = dets[0]
		cv2.rectangle(img2, (detected_face.left(),detected_face.top()), \
			(detected_face.right(),detected_face.bottom()), (0,0,255),2)
		fileout = open("tmp_detect/"+imname+".bbox","w")
		fileout.write("%f %f %f %f\n" % (detected_face.left(),detected_face.top(), \
			detected_face.right(),detected_face.bottom()))
		fileout.close()
		## If we are using landmarks to crop
		if useLM:
			print "> cropByLM "
			shape = predictor(dlib_img, detected_face)
			nLM = shape.num_parts
			fileout = open("tmp_detect/"+imname+".pts","w")
			for i in range(0,nLM):
				cv2.circle(img2, (shape.part(i).x, shape.part(i).y), 5, (255,0,0))
				fileout.write("%f %f\n" % (shape.part(i).x, shape.part(i).y))
			fileout.close()
			img = utils.cropByLM(img, shape, img2)
		else:
			print "> cropByFaceDet "
			img = utils.cropByFaceDet(img, detected_face, img2)
		cv2.imwrite("tmp_detect/"+imname+"_detect.png",img2)

	img = cv2.resize(img,(trg_size, trg_size))
	cv2.imwrite("tmp_ims/" + imname + ".png",img)
#####CNN fitting ############################## 

# load net
try: 
	caffe.set_mode_gpu()
	caffe.set_device(GPU_ID)
except Exception as ex:
	print '> Could not setup Caffe in GPU ' +str(GPU_ID) + ' - Error: ' + ex
	print '> Reverting into CPU mode'
	caffe.set_mode_cpu()
## Opening mean average image
proto_data = open(mean_path, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]
## Loading the CNN
net = caffe.Classifier(deploy_path, model_path)
## Setting up the right transformer for an input image
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
transformer.set_mean('data',mean)
print '> CNN Model loaded to regress 3D Shape and Texture!'
## Loading the Basel Face Model to write the 3D output
model = scipy.io.loadmat(BFM_path,squeeze_me=True,struct_as_record=False)
model = model["BFM"]
faces = model.faces-1
print '> Loaded the Basel Face Model to write the 3D output!'
## For loop over the input images
count = 0
listImgs = glob("tmp_ims/*.png")
for image_path in listImgs:
	count = count + 1
	fig_name = ntpath.basename(image_path)
	outFile = data_out + "/" + fig_name[:-4]
	print '> Processing image: ', image_path, ' ', fig_name, ' ', str(count) + '/' + str(len(listImgs))
	net.blobs['data'].reshape(1,3,trg_size,trg_size)
	im = caffe.io.load_image(image_path)
	## Transforming the image into the right format
	net.blobs['data'].data[...] = transformer.preprocess('data', im)
	## Forward pass into the CNN
	net_output = net.forward()
	## Getting the output
	features = np.hstack( [net.blobs[layer_name].data[0].flatten()] )
	## Writing the regressed 3DMM parameters
	np.savetxt(outFile + '.ply.alpha', features[0:99])
	np.savetxt(outFile + '.ply.beta', features[99:198])
	#################################
	## Mapping back the regressed 3DMM into the original
	## Basel Face Model (Shape)
	##################################
	S,T = utils.projectBackBFM(model,features)
	print '> Writing 3D file in: ', outFile + '.ply'
	utils.write_ply(outFile + '.ply', S, T, faces)
