#include <fstream>

#include "cv.h"
#include "highgui.h"

#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include "FaceServices2.h"
#include <sstream>
#include "DlibWrapper.h"
#include <string>
#include <iostream>
#include "H5Cpp.h"

using namespace std;
using namespace cv;

// Load landmark from file (68x2)
cv::Mat loadLM(const char* LMfile){
	ifstream in_stream(LMfile);
	if (!in_stream.is_open()) {
		return cv::Mat();
	}
	std::vector<float> vals;
    	string line;
	while (!in_stream.eof())
	{
		line.clear();
		std::getline(in_stream, line);
		if (line.size() == 0 || line.at(0) == '#')
			continue;
	    	std::istringstream iss(line);
	    	float x, y;
	    	if (!(iss >> x >> y)) { continue; }
		vals.push_back(x);
		vals.push_back(y);
	}
	int N = vals.size()/2;
	cv::Mat lms(vals.size()/2,2,CV_32F);
	for (int i=0;i<N;i++){
		lms.at<float>(i,0) = vals[2*i];
		lms.at<float>(i,1) = vals[2*i+1];
	}
	return lms;
}

// Load shape parameters from file (99x1)
cv::Mat loadWeight(const char* inputFile){
	ifstream in_stream(inputFile);
	if (!in_stream.is_open()) {
		return cv::Mat();
	}
	std::vector<float> vals;
    	string line;
	while (!in_stream.eof())
	{
		line.clear();
		std::getline(in_stream, line);
	    	std::istringstream iss(line);
	    	double w;
	    	if (!(iss >> w)) { continue; }
		vals.push_back(w);
	}
	int N = vals.size();
	cv::Mat out(N,1,CV_32F);
	for (int i=0;i<N;i++){
		out.at<float>(i,0) = vals[i];
	}
	return out;
}

// Get cropped image:
//     Inputs:
//	  oriIm   : original image
//	  oriLMs  : original landmarks (68x2)
//     Outputs:
//	  newLMs  : landmarks of the cropped image
//	  return the cropped image
cv::Mat getCroppedIm(Mat& oriIm, cv::Mat &oriLMs, cv::Mat &newLMs)
{	
	float padding = 1.7;			
	newLMs = oriLMs.clone();
	Mat xs = newLMs(Rect(0, 0, 1, oriLMs.rows));
	Mat ys = newLMs(Rect(1, 0, 1, oriLMs.rows));
	
	// get LM-tight bounding box
	double min_x, max_x, min_y, max_y;
	cv::minMaxLoc(xs, &min_x, &max_x);
	cv::minMaxLoc(ys, &min_y, &max_y);
	float width = max_x - min_x;
	float height = max_y - min_y;

        if (width < 5 || height < 5 || width*height < 100) {
		std::cout << "-> Error: Input face is too small" << std::endl;
                return cv::Mat();
        }

	// expand bounding box
	int minCropX = max((int)(min_x-padding*width/3.0),0);
	int minCropY = max((int)(min_y-padding*height/3.0),0);

	int widthCrop = min((int)(width*(3+2*padding)/3.0f), oriIm.cols - minCropX - 1);
	int heightCrop = min((int)(height*(3+2*padding)/3.0f), oriIm.rows - minCropY - 1);

	if(widthCrop <= 0 || heightCrop <=0) return cv::Mat();

	// normalize image size to get a stable pose estimation, assuming focal length 1000
	double minRes = 250*250 * (3+2*padding)/5.0f;
	double maxRes = 300*300 * (3+2*padding)/5.0f;
	
	double scaling = 1;
	if (widthCrop*heightCrop < minRes) scaling = std::sqrt(widthCrop*heightCrop/minRes);
	else if (widthCrop*heightCrop > maxRes) scaling = std::sqrt(widthCrop*heightCrop/maxRes);

	// first crop the image
	cv::Mat display_image = oriIm(Rect((int)(minCropX), (int)(minCropY), (int)(widthCrop), (int)(heightCrop)));

	// now scale it
	if (scaling != 1)
		cv::resize(display_image.clone(), display_image, Size(), 1/scaling, 1/scaling);

	int nrows = display_image.rows/4 * 4;
	int ncols = display_image.cols/4 * 4;
	if (nrows != display_image.rows || ncols != display_image.cols){
		display_image = display_image(Rect(0,0,ncols,nrows)).clone();
	}
	xs = (xs - minCropX)/scaling;
	ys = (ys - minCropY)/scaling;
	return display_image;	
}

int main(int argc, char** argv)
{
    char text[200];
    if (argc < 5) {
	printf("Usage: Visualize3D <imPath> <input 3D alpha> <BaselFace.dat path> <dlib path> [<LM path>]\n");
	return -1;
    }
    char imPath[200];
    char lmPath[200] = "";
    if (argc > 5){
        strcpy(lmPath,argv[5]);
    }
    int overlaidType = 1;              // 0: Windows         1: Docked in the second column         2: Hidden
    bool showInput = true;             // Config the 1st column in the output: the input image (true), or the cropped one (false)

    strcpy(imPath,argv[1]);
    //BaselFace::load_BaselFace_data(argv[3]);
    DlibWrapper dw(argv[4]);
    cv::Mat alpha = loadWeight(argv[2]);
    if (alpha.rows != 99){
		std::cout << "-> Error: Invalid alpha input!" << std::endl;
		return -1;
    }
    int outSize = 380;			// Size of visualized image

    // original image
    cv::Mat oriImg = imread(imPath);
    if( oriImg.empty() )  return 0;

    FaceServices2 fservice(argv[3]);
    cv::Scalar normTextColor(0,0,0);
    cv::Scalar boldTextColor(0,0,255);
    cv::Scalar labelTextColor(255,0,0);
		
    cv::Mat vecR, vecT, exprWeights;
    cv::Mat im1, im2, imOut, imOut2, zbuffer, imFinal;
    cv::Mat lms_init, lms;

    // Get landmarks on the original image
    if (strlen(lmPath) > 0){
	lms_init = loadLM(lmPath);
	if (lms_init.rows != 68){
		printf("Bad landmark input file with %d points!\n", lms_init.rows);
		return -1;
	}
    }
    else {
	std::vector<cv::Mat> lms0 = dw.detectLM(oriImg);
	if (lms0.size() == 0){
		printf("No face is detected!\n");
		return -1;
	}
	lms_init = lms0[0].clone();
    }

    // get cropped image & landmarks
    cv::Mat croppedImg = getCroppedIm(oriImg, lms_init, lms);

    // Scale to fit output size.
    cv::Mat croppedImg1 = croppedImg.clone();
    float scale = outSize/((float)croppedImg1.rows);
    float scaleX = floor(croppedImg1.cols*scale)/((float)(croppedImg1.cols));
    cv::resize(croppedImg1, croppedImg,cv::Size(croppedImg1.cols*scale,outSize));
    lms(cv::Rect(0, 0, 1, 68)) *= scaleX;
    lms(cv::Rect(1, 0, 1, 68)) *= scale;
    int ncols = croppedImg.cols/4 * 4;
    if (ncols != croppedImg.cols)
    	croppedImg = croppedImg(cv::Rect(0,0,ncols,outSize)).clone();
    
    fservice.init(croppedImg.cols,croppedImg.rows,1000.0f*scale); // Scale also the focal length

    // estimate pose + expression
    double ti = (double)getTickCount(); 
    fservice.estimatePoseExpr(croppedImg, lms, alpha, vecR, vecT, exprWeights);
    ti = ((double)getTickCount() - ti)/getTickFrequency(); 

    // render face shape overlaying on the input image
    double ti2 = (double)getTickCount(); 
    fservice.renderShape(croppedImg, alpha, vecR, vecT, exprWeights, im2, zbuffer);  // render
    fservice.mergeIm(&im2,croppedImg,zbuffer);                                       // overlay
    ti2 = ((double)getTickCount() - ti2)/getTickFrequency(); 
    printf("** Pose+expr fitting: %.3f s\n", ti);
    printf("** Visualization    : %.3f s\n", ti2);
    printf("*****************************************\n");
    
    if (overlaidType == 0) {
    	imshow("Overlaid",im2);
    	cv::moveWindow("Overlaid", 40, 550);
    }
    cv::Mat overlaidIm = im2.clone();
    cv::Mat noOverlaidIm = croppedImg.clone();
    

    // Prepare animated image
    //FaceServices2 fservice2;
    fservice.init(outSize,outSize,1000.0f*scale);
    vecR = cv::Mat::zeros(3,1,CV_32F)+ 0.00001;
    vecT = cv::Mat::zeros(3,1,CV_32F)+ 0.00001;
    vecT.at<float>(0,0) = 5.156312*scale;
    vecT.at<float>(1,0) = 11.053386*scale;
    vecT.at<float>(2,0) = -709.398865*scale;
    int currFrame = -1;
    cv::Mat animatedImg = cv::Mat::zeros(outSize,outSize,CV_8UC3);

    // 1st column in the output
    if (showInput) {
	    scale = outSize/((float)oriImg.rows);
	    cv::resize(oriImg, im1,cv::Size(oriImg.cols*scale,outSize));
    }
    else im1 = noOverlaidIm.clone();		
    
    // Instruction texts
    cv::Mat imInstruct(32, im1.cols + outSize + (overlaidType == 1)*overlaidIm.cols, CV_8UC3, cv::Scalar(150, 150, 150));
    cv::putText(imInstruct,string("Input"), cv::Point((im1.cols-40)/2,18), FONT_HERSHEY_TRIPLEX, 0.5, labelTextColor);
    if (overlaidType == 1)
    	cv::putText(imInstruct,string("Overlaid"), cv::Point(im1.cols + (overlaidIm.cols-64)/2,18), FONT_HERSHEY_TRIPLEX, 0.5, labelTextColor);
    cv::putText(imInstruct,string("3D Model"), cv::Point(im1.cols + (overlaidType == 1)*overlaidIm.cols + (outSize-64)/2,18), FONT_HERSHEY_TRIPLEX, 0.5, labelTextColor);

    cv::Mat imInstruct2(32, im1.cols + outSize + (overlaidType == 1)*overlaidIm.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::putText(imInstruct2,string("Press     to quit"), cv::Point(20,18), FONT_HERSHEY_TRIPLEX, 0.5, normTextColor);
    cv::putText(imInstruct2,string("ESC"), cv::Point(70,18), FONT_HERSHEY_TRIPLEX, 0.5, boldTextColor);


    // animate
    bool firstFrame = true;
    bool useOverlaid = true;
    for(;;)
    {
	  // get next pose & render
          fservice.nextMotion(currFrame, vecR, vecT, exprWeights);   
	  im2 = fservice.renderShape(animatedImg, alpha, vecR, vecT, exprWeights);

	  // display
          if (overlaidType != 1) {
		  hconcat(im1,im2,imOut);
		  vconcat(imOut,imInstruct2,imFinal);
		  imshow("Model", imFinal);
	  }
          else {
		if (useOverlaid) hconcat(im1,overlaidIm,imOut);
		else hconcat(im1,noOverlaidIm,imOut);
		hconcat(imOut,im2,imOut2);
		vconcat(imOut2,imInstruct,imOut);
		vconcat(imOut,imInstruct2,imFinal);
		imshow("Model", imFinal);
	  }
          if (firstFrame) {
          	cv::moveWindow("Model", 40, 20);
		firstFrame = false;
	  }

	  // check input key
	  char ch = waitKey(15);
	  if(ch == 27 ) break; // stop capturing by pressing ESC 
          else if (ch == 't') {              // turn on/off overlay
		useOverlaid = !useOverlaid;
		if (overlaidType != 1) {
			if (useOverlaid) imshow("Overlaid",overlaidIm);
			else imshow("Overlaid",noOverlaidIm);
		}
		else {
			if (useOverlaid) hconcat(im1,overlaidIm,imOut);
			else hconcat(im1,noOverlaidIm,imOut);
			hconcat(imOut,im2,imOut2);
			vconcat(imOut2,imInstruct,imOut);
			vconcat(imOut,imInstruct2,imFinal);
			imshow("Model", imFinal);
		}
	  }
    }
    return 0;
}
