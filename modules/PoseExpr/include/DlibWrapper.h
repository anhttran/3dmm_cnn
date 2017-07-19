#pragma once
#include "cv.h"
#include "highgui.h"
#define DLIB_JPEG_SUPPORT
#define DLIB_PNG_SUPPORT

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_transforms/assign_image.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

class DlibWrapper {
   dlib::frontal_face_detector detector;
   dlib::shape_predictor sp;

public:
   // Load Dlib with shape-predictor model path
   DlibWrapper(char* dlibmodel);

   // Detect faces and the corresponding landmarks
   //    Input:
   //       im          : input image
   //       maxNumFaces : maximum number of images to process
   //       scale       : scale ration for faster face tection
   //    Ouput:
   //       A list of 2D landmarks (68x2)
   std::vector<cv::Mat> detectLM(cv::Mat im, int maxNumFaces = 1, float scale = 0.5);

   // Detect faces
   //    Input:
   //       im          : input image
   //       maxNumFaces : maximum number of images to process
   //       scale       : scale ration for faster face tection
   //    Ouput:
   //       A list of 2D bounding boxes
   std::vector<cv::Rect_<float> > detectBB(cv::Mat im, int maxNumFaces = 1, float scale = 0.5);

   // Detect LM points (68x2), given face bounding box
   cv::Mat detectLM(cv::Mat im, cv::Rect_<float> &bb);
};

