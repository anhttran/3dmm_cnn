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
import numpy as np
import scipy.io
import os

def convert(data):
	new = data.reshape(data.shape[0],1).astype('float64')
	return new
if not os.path.exists('01_MorphableModel.mat'):
	print "Cannot find '01_MorphableModel.mat'"
	exit(0)
model = scipy.io.loadmat('01_MorphableModel.mat',squeeze_me=True,struct_as_record=False)
mod_struct = scipy.io.loadmat('mod_struct.mat',squeeze_me=True,struct_as_record=False)
mod_struct = mod_struct['mod_struct'];
##Python indexing
vertex_indices=mod_struct.vertex_indices
numV_new = vertex_indices.shape[0]
##MATLAB : vertex_indices3 = [3*mod_struct.vertex_indices-2;3*mod_struct.vertex_indices-1;3*mod_struct.vertex_indices];
vertex_indices3 = [None] * (numV_new*3);
vertex_indices3[0::3] = [3*(i-1) for i in mod_struct.vertex_indices];
vertex_indices3[1::3] = [3*(i-1)+1 for i in mod_struct.vertex_indices];
vertex_indices3[2::3] = [3*(i-1)+2 for i in mod_struct.vertex_indices];
vertex_indices3 = np.array(vertex_indices3)
##MATLAB: BFM.shapeMU = model.shapeMU(vertex_indices3,:)/1000;
shapeMU = model['shapeMU'][vertex_indices3]/1000;
##MATLAB: BFM.shapeMU(1:3:end) = BFM.shapeMU(1:3:end) + mod_struct.tx;
shapeMU[0:-1:3] = shapeMU[0:-1:3] +  mod_struct.tx
##MATLAB: BFM.shapeEV = model.shapeEV(1:99)/1000;
shapeEV = model['shapeEV'][0:99]/1000.
##MATLAB BFM.shapePC = model.shapePC(vertex_indices3,1:99);
shapePC = model['shapePC'][vertex_indices3,0:99]
##MATLAB  BFM.texMU = model.texMU(vertex_indices3,:);
texMU = model['texMU'][vertex_indices3]
##MATLAB  BFM.texPC = model.texPC(vertex_indices3,1:99);
texPC = model['texPC'][vertex_indices3,0:99];
##MATLAB  BFM.texEV = model.texEV(1:99);
texEV = model['texEV'][0:99];
innerLandmarkIndex = mod_struct.innerLandmarkIndex
outerLandmarkIndex = mod_struct.outerLandmarkIndex
#% Trimming faces
#ind_inv = zeros(1,length(model.shapeMU)/3);
ind_inv = np.zeros(model['shapeMU'].shape[0]/3)
#ind_inv(mod_struct.vertex_indices) = 1: length(mod_struct.vertex_indices);
ind_inv[vertex_indices-1] = np.arange(1,vertex_indices.shape[0]+1)
#faces = ind_inv(model.tl);
faces = ind_inv[model['tl']-1]
#faces_keep = find(faces(:,1)>0 & faces(:,2)>0 & faces(:,3)>0);
faces_keep = ( faces[:,0]>0) & (faces[:,1]>0) & (faces[:,2]>0 )
#faces = faces(faces_keep,[2 1 3])
faces = np.dstack([faces[faces_keep,1], faces[faces_keep,0], faces[faces_keep,2]])
faces = faces[0]

shapeEV = convert(shapeEV)
outerLandmarkIndex = convert(outerLandmarkIndex)
innerLandmarkIndex = convert(innerLandmarkIndex)
texEV = convert(texEV)
shapeMU = convert(shapeMU)

data = { 'BFM' : {
			'faces' : faces,
            'shapeMU' :  shapeMU,
            'shapePC' :  shapePC,
            'shapeEV' : shapeEV,
            'texMU' :  texMU,
            'texPC' :  texPC,
            'texEV' :  texEV,
            'innerLandmarkIndex' :  innerLandmarkIndex,                                   
            'outerLandmarkIndex' :  outerLandmarkIndex,              
          }
       }
scipy.io.savemat('BaselFaceModel_mod.mat',data)
if os.path.exists('01_MorphableModel.mat'):
		os.remove('01_MorphableModel.mat')
