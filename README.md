Python code for 3D face modeling from single image using **[our very deep neural network](http://www.openu.ac.il/home/hassner/projects/CNN3DMM)**
===========
**_New_: Please see our follow up project for [deep pose and 3D expression fitting](https://github.com/fengju514/Expression-Net).**

This page contains end-to-end demo code that estimates the 3D facial shape and texture directly from an unconstrained 2D face image. For a given input image, it produces a standard ply file of the face shape and texture. It accompanies the deep network described in our paper [1]. We also include demo code of pose and expression fitting from landmarks in this release. 

This release is part of an on-going face recognition and modeling project. Please, see **[this page](http://www.openu.ac.il/home/hassner/projects/CNN3DMM)** for updates and more data.

![Teaser](http://www-bcf.usc.edu/~iacopoma/img/3dmm_code_teaser.png)


## Features
* **End-to-End code** to be used for **3D shape and texture estimation** directly from image intensities
* Designed and tested on **face images in unconstrained conditions**, including the challenging LFW, YTF and IJB-A benchmarks
* The 3D face shape and texture parameters extracted using our network were **shown for the first time to be descriminative and robust**, providing near state of the art face recognition performance with 3DMM representations on these benchmarks
* **No expensive, iterative optimization, inner loops** to regress the shape. 3DMM fitting is therefore extremely fast
* Extra code for **head pose and expression estimation from detected facial landmarks**, with the use of the regressed 3D face model

## Dependencies

## Library requirements

* [Dlib Python and C++ library](http://dlib.net/)
* [OpenCV Python and C++ library](http://opencv.org/)
* [Caffe](caffe.berkeleyvision.org) (**version 1.0.0-rc3 or above required**)
* [Numpy](http://www.numpy.org/)
* [Python2.7](https://www.python.org/download/releases/2.7/)

The code has been tested on Linux only. On Linux you can rely on the default version of python, installing all the packages needed from the package manager or on Anaconda Python and install required packages through `conda`. A bit more effort is required to install caffé, dlib, and libhdf5.

## Automatic Install of the Dependencies 
Check this [useful script on the wiki](https://github.com/anhttran/3dmm_cnn/wiki/Installation-Script) by [seva100](https://github.com/seva100)

## Data requirements

Before running the code, please, make sure to have all the required data in the following specific folder:
- **[Download our CNN](http://www.openu.ac.il/home/hassner/projects/CNN3DMM)** and move the CNN model (3 files: `3dmm_cnn_resnet_101.caffemodel`,`deploy_network.prototxt`,`mean.binaryproto`) into the `CNN` folder
- **[Download the Basel Face Model](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads)** and move `01_MorphableModel.mat` into the `3DMM_model` folder
- **[Acquire 3DDFA Expression Model](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Code/3DDFA.zip)**, run its code to generate `Model_Expression.mat` and move this file the `3DMM_model` folder
- Go into `3DMM_model` folder. Run the script `python trimBaselFace.py`. This should output 2 files `BaselFaceModel_mod.mat` and `BaselFaceModel_mod.h5`.
- **[Download dlib face prediction model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)** and move the `.dat` file into the `dlib_model` folder.

## Installation (pose & expression fitting code)

- Install **cmake**: 
```
	apt-get install cmake
```
- Install **opencv** (2.4.6 or higher is recommended):
```
	(http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html)
```
- Install **libboost** (1.5 or higher is recommended):
```
	apt-get install libboost-all-dev
```
- Install **OpenGL, freeglut, and glew**
```
	sudo apt-get install freeglut3-dev
	sudo apt-get install libglew-dev
```
- Install **libhdf5-dev** library
```
	sudo apt-get install libhdf5-dev
```
- Install **Dlib C++ library**. Dlib should be compiled to shared objects. Check the comments in its CMakeList.txt.
```
	(http://dlib.net/)
```
- Update Dlib directory paths (`DLIB_INCLUDE_DIR` and `DLIB_LIB_DIR`) in `CMakeLists.txt`
- Make build directory (temporary). Make & install to bin folder
```
	mkdir build
	cd build
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../bin ..
	make
	make install
```
  This code should generate `TestVisualization` in `bin` folder

## Usage

### 3DMM fitting on a set of input images

* Go into `demoCode` folder. The demo script can be used from the command line with the following syntax:

```bash
$ Usage: python testBatchModel.py <inputList> <outputDir> <needCrop> <useLM>
```

where the parameters are the following:
- `<inputList>` is a text file containing the paths to each of the input images, one in each line.
- `<outputDir>` is the path to the output directory, where ply files are stored.
- `<needCrop>` tells the demo if the images need cropping (1) or not (0). Default 1. If your input image size is equal (square) and has a CASIA-like [2] bounding box, you can set `<needCrop>` as 0. Otherwise, you have to set it as 1.
- `<useLM>` is an option to refine the bounding box using detected landmarks (1) or not (0). Default 1.

Example for `<inputList>`:
<pre>
data/1.jpg
data/2.jpg
....
</pre>

* The demo code should produce an output similar to this:

```bash
user@system:~/Desktop/3dmm_release$ python testBatchModel.py input.txt out/
> Prepare image data/1.jpg:
>     Number of faces detected: 1
> Prepare image data/2.jpg:
>     Number of faces detected: 1
> CNN Model loaded to regress 3D Shape and Texture!
> Loaded the Basel Face Model to write the 3D output!
> Processing image:  tmp_ims/2.png   2.png   1/2
> Writing 3D file in:  out//2.ply
> Processing image:  tmp_ims/1.png   1.png   2/2
> Writing 3D file in:  out//1.ply

```

The final 3D shape and texture can be displayed using standard off-the-shelf 3D (ply file) visualization software such as [MeshLab](http://meshlab.sourceforge.net). Using MeshLab, the output may be displayed as follows:

`user@system:~/Desktop/3dmm_release$ meshlab out/1.ply`

`user@system:~/Desktop/3dmm_release$ meshlab out/2.ply`

which should produce something similar to:

![Teaser](http://www-bcf.usc.edu/~iacopoma/img/meshlab_disp.png)


### 3D Face modeling + pose & expression estimation on a single input image

* Go into `demoCode` folder. The demo script can be used from the command line with the following syntax:

```bash
$ Usage: python testModel_PoseExpr.py <outputDir> <save3D>
```

where the parameters are the following:
- `<outputDir>` is the path to the output directory, where 3DMM (and ply) files are stored.
- `<save3D>` is an option to save the ply file (1) or not (0). Default 1.

* The program will pop up a dialog to select an input image. Then it will estimate 3DMM paramters (with the CNN model), estimate pose+expression and visualize the result (with C++ program)

Example:
```bash
user@system:~/Desktop/3dmm_release$ python testModel_PoseExpr.py out/
(Select `Anders_Fogh_Rasmussen_0004.jpg`)
> Prepare image /home/anh/Downloads/PoseExprFromLM-master/demoCode/data/Anders_Fogh_Rasmussen_0004.jpg:
    Number of faces detected: 1
> CNN Model loaded to regress 3D Shape and Texture!
> Loaded the Basel Face Model to write the 3D output!
*****************************************
** Caffe loading    : 1.007 s
** Image cropping   : 0.069 s
** 3D Modeling      : 1.145 s
*****************************************
> Writing 3D file in:  out/Anders_Fogh_Rasmussen_0004.ply
> Pose & expression estimation
load ../3DMM_model/BaselFaceModel_mod.h5
** Pose+expr fitting: 0.153 s
** Visualization    : 0.052 s
*****************************************
```

The pop up window should look similar to:
![Teaser](https://sites.google.com/site/anhttranusc/PoseExpr_Demo.png)



## Citation

If you find this work useful, please cite our paper [1] with the following bibtex:

```latex
@inproceedings{tran2017regressing,
  title={Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network},
  author={Tran, Anh Tuan and Hassner, Tal and Masi, Iacopo and Medioni, Gerard},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```

## Troubleshooting

### Problem: Old Caffe Engine ###

```C++
"F0210 10:49:17.604714 24046 net.cpp:797] Check failed:
target_blobs.size() == source_layer.blobs_size() (5 vs. 3) Incompatible
number of blobs for layer bn_conv1"
```

### Solution: install caffe 1.0.0-rc3 or above. ###
For more info on caffe  verson please see https://github.com/BVLC/caffe/releases

To check your caffe version from python:

```python
In [3]: import caffe
In [4]: caffe.__version__
Out[4]: '1.0.0-rc3'
```

## References

[1] A. Tran, T. Hassner, I. Masi, G. Medioni, "[Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network](https://arxiv.org/abs/1612.04904)", arxiv pre-print 2016 

[2] Dong Yi, Zhen Lei, Shengcai Liao and Stan Z. Li, "Learning Face Representation from Scratch". arXiv preprint arXiv:1411.7923. 2014.

## Changelog
- Jan 2017, First Release 

## License and Disclaimer
Please, see [the LICENSE here](LICENSE.txt)

## Contacts

If you have any questions, drop an email to _anhttran@usc.edu_ , _hassner@isi.edu_ and _iacopoma@usc.edu_  or leave a message below with GitHub (log-in is needed).
