#include "DlibWrapper.h"
#include <iostream>

using namespace dlib;
using namespace std;

// Load Dlib with shape-predictor model path
DlibWrapper::DlibWrapper(char* dlib_model){
	detector = get_frontal_face_detector();
        deserialize(dlib_model) >> sp;
}

// Detect faces and the corresponding landmarks
//    Input:
//       im          : input image
//       maxNumFaces : maximum number of images to process
//       scale       : scale ration for faster face tection
//    Ouput:
//       A list of 2D landmarks (68x2)
std::vector<cv::Mat> DlibWrapper::detectLM(cv::Mat im, int maxNumFaces, float scale){
	    cv::Mat im_re;
	    cv::resize(im,im_re,cv::Size(),scale,scale);
	    dlib::array2d<bgr_pixel> img;
            dlib::assign_image(img, dlib::cv_image<bgr_pixel>(im_re));
	    dlib::array2d<bgr_pixel> img_ori;
            dlib::assign_image(img_ori, dlib::cv_image<bgr_pixel>(im));

            // Now tell the face detector to give us a list of bounding boxes
            std::vector<rectangle> dets = detector(img);
	    std::vector<cv::Mat> out;
            //cout << "Number of faces detected: " << dets.size() << endl;

            std::vector<full_object_detection> shapes;
            for (unsigned long j = 0; j < dets.size() && j < maxNumFaces; ++j)
            {   
		// Scale back and detect landmark points
		rectangle newRect( dets[j].left()/scale,dets[j].top()/scale,dets[j].right()/scale,dets[j].bottom()/scale);
                full_object_detection shape = sp(img_ori, newRect);
		cv::Mat lms(68,2,CV_32F);
                for (int p=0;p<shape.num_parts();p++){
		    lms.at<float>(p,0) = shape.part(p)(0);
		    lms.at<float>(p,1) = shape.part(p)(1);
                }
                out.push_back(lms);
            }
	    return out;
}

// Detect faces
//    Input:
//       im          : input image
//       maxNumFaces : maximum number of images to process
//       scale       : scale ration for faster face tection
//    Ouput:
//       A list of 2D bounding boxes
std::vector<cv::Rect_<float> > DlibWrapper::detectBB(cv::Mat im, int maxNumFaces, float scale){
	    cv::Mat im_re;
	    cv::resize(im,im_re,cv::Size(),scale,scale);
	    dlib::array2d<bgr_pixel> img;
            dlib::assign_image(img, dlib::cv_image<bgr_pixel>(im_re));
	    std::vector<rectangle> dets = detector(img);
	    std::vector<cv::Rect_<float> > out;
            for (unsigned long j = 0; j < dets.size() && j < maxNumFaces; ++j)
            {
		cv::Rect_<float> newRect(dets[j].left()/scale,dets[j].top()/scale,(dets[j].right()-dets[j].left())/scale,(dets[j].bottom()-dets[j].top())/scale);
                out.push_back(newRect);
            }
	    return out;
}

// Detect LM points (68x2), given face bounding box
cv::Mat DlibWrapper::detectLM(cv::Mat im, cv::Rect_<float> &bb){
	dlib::array2d<bgr_pixel> img_ori;
        dlib::assign_image(img_ori, dlib::cv_image<bgr_pixel>(im));
        rectangle newRect( bb.tl().x,bb.tl().y,bb.br().x,bb.br().y);
        full_object_detection shape = sp(img_ori, newRect);
	cv::Mat lms(68,2,CV_32F);
        for (int p=0;p<shape.num_parts();p++){
		    lms.at<float>(p,0) = shape.part(p)(0);
		    lms.at<float>(p,1) = shape.part(p)(1);
        }    
        return lms;
}
