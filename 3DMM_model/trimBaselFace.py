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
import h5py 

with_forehead = False

def convert(data):
	new = data.reshape(data.shape[0],1).astype('float64')
	return new
if not os.path.exists('01_MorphableModel.mat'):
	print "Cannot find '01_MorphableModel.mat'"
	exit(0)

if not os.path.exists('Model_Expression.mat'):
	print "Cannot find 'Model_Expression.mat'"
	exit(0)

mod_struct = scipy.io.loadmat('mod_struct_expr.mat',squeeze_me=True,struct_as_record=False)
mod_struct = mod_struct['mod_struct'];

##############  Basel Face Model ##########################################
model = scipy.io.loadmat('01_MorphableModel.mat',squeeze_me=True,struct_as_record=False)
##Python indexing
if not with_forehead:
	vertex_indices=mod_struct.vertex_indices
else:
	vertex_indices=mod_struct.vertex_indices_wForehead
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
##MATLAB: BFM.shapePC = model.shapePC(vertex_indices3,1:99);
shapePC = model['shapePC'][vertex_indices3,0:99]
##MATLAB: BFM.texMU = model.texMU(vertex_indices3,:);
texMU = model['texMU'][vertex_indices3]
##MATLAB: BFM.texPC = model.texPC(vertex_indices3,1:99);
texPC = model['texPC'][vertex_indices3,0:99];
##MATLAB: BFM.texEV = model.texEV(1:99);
texEV = model['texEV'][0:99];

if not with_forehead:
	innerLandmarkIndex = mod_struct.innerLandmarkIndex
	outerLandmarkIndex = mod_struct.outerLandmarkIndex
	faces = mod_struct.faces
else:
	innerLandmarkIndex = mod_struct.innerLandmarkIndex_wForehead
	outerLandmarkIndex = mod_struct.outerLandmarkIndex_wForehead
	faces = mod_struct.faces_wForehead

##############  Expression Model #############################################
model_exp = scipy.io.loadmat('Model_Expression.mat',squeeze_me=True,struct_as_record=False)
##Python indexing
vertex_expr_ref=mod_struct.vertex_expr_ref
if not with_forehead:
	vertex_indices_exp=[vertex_expr_ref[i-1] for i in mod_struct.vertex_indices]
else:
	vertex_indices_exp=[vertex_expr_ref[i-1] for i in mod_struct.vertex_indices_wForehead]
##MATLAB : vertex_indices_exp3 = [3*mod_struct.vertex_indices_exp-2;3*mod_struct.vertex_indices_exp-1;3*mod_struct.vertex_indices_exp];
vertex_indices_exp3 = [None] * (numV_new*3);
vertex_indices_exp3[0::3] = [3*(i-1) for i in vertex_indices_exp];
vertex_indices_exp3[1::3] = [3*(i-1)+1 for i in vertex_indices_exp];
vertex_indices_exp3[2::3] = [3*(i-1)+2 for i in vertex_indices_exp];
vertex_indices_exp3 = np.array(vertex_indices_exp3)
##MATLAB: BFM.expMU = mu_exp(vertex_indices_exp3)/1000;
expMU = model_exp['mu_exp'][vertex_indices_exp3]/1000;
##MATLAB: BFM.expEV = mod_struct.expEV;
expEV = mod_struct.expEV;
##MATLAB: BFM.expPC = w_exp(vertex_indices_exp3,:)/1000;
expPC = model_exp['w_exp'][vertex_indices_exp3,:]/1000;

shapeEV = convert(shapeEV)
outerLandmarkIndex = convert(outerLandmarkIndex)
innerLandmarkIndex = convert(innerLandmarkIndex)
texEV = convert(texEV)
shapeMU = convert(shapeMU)
expMU = convert(expMU)
expEV = convert(expEV)


data = { 'BFM' : {
	    'faces' : faces,
            'shapeMU' :  shapeMU,
            'shapePC' :  shapePC,
            'shapeEV' : shapeEV,
            'texMU' :  texMU,
            'texPC' :  texPC,
            'texEV' :  texEV,
            'expMU' :  expMU,
            'expPC' :  expPC,
            'expEV' : expEV,
            'innerLandmarkIndex' :  innerLandmarkIndex,                                   
            'outerLandmarkIndex' :  outerLandmarkIndex,              
          }
       }
scipy.io.savemat('BaselFaceModel_mod.mat',data)

fH5 = h5py.File('BaselFaceModel_mod.h5', "w")
fH5['faces'] = faces;
fH5['shapeMU'] = shapeMU;
fH5['shapePC'] = shapePC;
fH5['shapeEV'] = shapeEV;
fH5['texMU'] = texMU;
fH5['texPC'] = texPC;
fH5['texEV'] = texEV;
fH5['expMU'] = expMU;
fH5['expPC'] = expPC;
fH5['expEV'] = expEV;
fH5['innerLandmarkIndex'] = innerLandmarkIndex;
fH5['outerLandmarkIndex'] = outerLandmarkIndex;           
fH5.close()
