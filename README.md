Python code for estimating 3DMM parameters using **[our very deep neural network](http://www.openu.ac.il/home/hassner/projects/CNN3DMM)**
===========

This page contains end-to-end demo code that estimates the 3D facial shape and texture directly from an unconstrained 2D face image. For a given input image, it produces a standard ply file of the face shape and texture. It accompanies the deep network described in our paper [1].

This release is part of an on-going face recognition and modeling project. Please, see **[this page](http://www.openu.ac.il/home/hassner/projects/CNN3DMM)** for updates and more data.

**A new version of this code which includes also pose and expression fitting, is being prepared and will be released soon.**

![Teaser](http://www-bcf.usc.edu/~iacopoma/img/3dmm_code_teaser.png)


## Features
* **End-to-End code** to be used for **3D shape and texture estimation** directly from image intensities
* Designed and tested on **face images in unconstrained conditions**, including the challenging LFW, YTF and IJB-A benchmarks
* The 3D face shape and texture parameters extracted using our network were **shown for the first time to be descriminative and robust**, providing near state of the art face recognition performance with 3DMM representations on these benchmarks
* **No expensive, iterative optimization, inner loops** to regress the shape. 3DMM fitting is therefore extremely fast

## Dependencies

## Library requirements

* [Dlib Python Wrapper](http://dlib.net/)
* [OpenCV Python Wrapper](http://opencv.org/)
* [Caffe](caffe.berkeleyvision.org) (version 1.0.0-rc3 or above required)
* [Numpy](http://www.numpy.org/)
* [Python2.7](https://www.python.org/download/releases/2.7/)

The code has been tested on Linux only. On Linux you can rely on the default version of python, installing all the packages needed from the package manager or on Anaconda Python and install required packages through `conda`. A bit more effort is required to install caffé.

## Data requirements

Before running the code, please, make sure to have all the required data in the following specific folder:
- **[Download our CNN](http://www.openu.ac.il/home/hassner/projects/CNN3DMM)**
- **[Download the Basel Face Model](http://faces.cs.unibas.ch/bfm/main.php?nav=1-2&id=downloads)** 
- Move the CNN model (3 files: `3dmm_cnn_resnet_101.caffemodel`,`deploy_network.prototxt`,`mean.binaryproto`) into the `CNN` folder
- Copy  the Basel Face Model (`01_MorphableModel.mat`) in the same folder of `demo.py` file.
- Run the script `python trimBaselFace.py`. This should output a file `BaselFaceModel_mod.mat` and remove the original one automatically.
- Download [dlib face prediction model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and move the `.dat` file into the `dlib_model` folder.

## Usage

### Input

The demo script can be used from the command line with the following syntax:

```bash
$ Usage: python demo.py <inputList> <outputDir> <needCrop> <useLM>
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

### Output
The demo code should produce an output similar to this:

```bash
user@system:~/Desktop/3dmm_release$ python demo.py input.txt out/
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

## Citation

If you find this work useful, please cite our paper [1] with the following bibtex:

``` latex
@article{tran16_3dmm_cnn,
  title={Regressing Robust and Discriminative {3D} Morphable Models with a very Deep Neural Network},
  author={Anh Tran 
      and Tal Hassner 
      and Iacopo Masi
      and G\'{e}rard Medioni}
  journal={arXiv preprint},
  year={2016}
}
```

## References

[1] A. Tran, T. Hassner, I. Masi, G. Medioni, "[Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network](https://arxiv.org/abs/1612.04904)", arxiv pre-print 2016 

[2] Dong Yi, Zhen Lei, Shengcai Liao and Stan Z. Li, "Learning Face Representation from Scratch". arXiv preprint arXiv:1411.7923. 2014.

## Changelog
- Jan 2017, First Release 

## Disclaimer

_The SOFTWARE PACKAGE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use._

## Contacts

If you have any questions, drop an email to _anhttran@usc.edu_ , _hassner@isi.edu_ and _iacopo.masi@usc.edu_  or leave a message below with GitHub (log-in is needed).
