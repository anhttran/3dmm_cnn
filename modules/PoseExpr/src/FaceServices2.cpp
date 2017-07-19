/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "FaceServices2.h"
#include <fstream>
#include "opencv2/contrib/contrib.hpp"
#include <Eigen/SparseLU>
//#include <Eigen/SPQRSupport>
#include <omp.h>

using namespace std;
using namespace cv;

FaceServices2::FaceServices2(const std::string & model_file)
{
	omp_set_num_threads(8);
	mstep = 0.0001;
	countFail = 0;
	maxVal = 4;
	im_render = nullptr;
        printf("load %s\n",model_file.c_str());
	festimator.load3DMM(model_file);
}

FaceServices2::~FaceServices2(void)
{
}

// Setup with image size (w,h) and focal length f
void FaceServices2::init(int w, int h, float f){
	// Instrinsic matrix
	memset(_k,0,9*sizeof(float));
	_k[8] = 1;
	_k[0] = -f;
	_k[4] = f;
	_k[2] = w/2.0f;
	_k[5] = h/2.0f;
	if (faces.empty())
        	faces = festimator.getFaces() - 1;
	cv::Mat shape = festimator.getShape(cv::Mat(99,1,CV_32F));
	tex = shape*0 + 128;
	
	// Initialize image renderer
	if (im_render == nullptr)
	{
		im_render = new FImRenderer(cv::Mat::zeros(h, w, CV_8UC3));
		im_render->loadMesh(shape, shape * 0, faces);
	}
	else im_render->init(cv::Mat::zeros(h, w, CV_8UC3));
}

// Estimate pose and expression from landmarks
//     Inputs:
//	   colorIm    : Input image
//         lms        : 2D landmarks (68x2)
//         alpha      : Subject-specific shape parameters (99x1)
//     Outputs:
//         vecR       : Rotation angles (3x1)
//         vecT       : Translation vector (3x1)
//         exprWeight : Expression parameters (29x1)
bool FaceServices2::estimatePoseExpr(cv::Mat colorIm, cv::Mat lms, cv::Mat alpha, cv::Mat &vecR, cv::Mat &vecT, cv::Mat &exprW){
	char text[200];
	float renderParams[RENDER_PARAMS_COUNT];
	float renderParams2[RENDER_PARAMS_COUNT];
	Mat k_m(3,3,CV_32F,_k);
	BFMParams params;
	params.init();
	exprW = cv::Mat::zeros(29,1,CV_32F);

	int M = 99;
	// get subject shape
	cv::Mat shape = festimator.getShape(alpha);

	// get 3D landmarks
	Mat landModel0 = festimator.getLM(shape,0);
	write_plyFloat("lm.ply",landModel0);
	int nLM = landModel0.rows;

	// compute 3D pose w/ the first 60 2D-3D correspondences
	Mat landIm = cv::Mat( 60,2,CV_32F);
	Mat landModel = cv::Mat( 60,3,CV_32F);
	for (int i=0;i<60;i++){
		landModel.at<float>(i,0) = landModel0.at<float>(i,0);
		landModel.at<float>(i,1) = landModel0.at<float>(i,1);
		landModel.at<float>(i,2) = landModel0.at<float>(i,2);
		landIm.at<float>(i,0) = lms.at<float>(i,0);
		landIm.at<float>(i,1) = lms.at<float>(i,1);
	}
	festimator.estimatePose3D(landModel,landIm,k_m,vecR,vecT);

	// reselect 3D landmarks given estimated yaw angle
	float yaw = -vecR.at<float>(1,0);
	landModel0 = festimator.getLM(shape,yaw);
	// select landmarks to use based on estimated yaw angle
	std::vector<int> lmVisInd;
	for (int i=0;i<60;i++){
		if (i > 16 || abs(yaw) <= M_PI/10 || (yaw > M_PI/10 && i > 7) || (yaw < -M_PI/10 && i < 9))
			lmVisInd.push_back(i);
	}
	landModel = cv::Mat( lmVisInd.size(),3,CV_32F);
	landIm = cv::Mat::zeros( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landModel.at<float>(i,0) = landModel0.at<float>(ind,0);
		landModel.at<float>(i,1) = landModel0.at<float>(ind,1);
		landModel.at<float>(i,2) = landModel0.at<float>(ind,2);
		landIm.at<float>(i,0) = lms.at<float>(ind,0);
		landIm.at<float>(i,1) = lms.at<float>(ind,1);
	}
	// resetimate 3D pose
	festimator.estimatePose3D(landModel,landIm,k_m,vecR,vecT);
	
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_R+i] = vecR.at<float>(i,0);
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_T+i] = vecT.at<float>(i,0);
	memcpy(renderParams,params.initR,sizeof(float)*RENDER_PARAMS_COUNT);
	
	// add the inner mouth landmark points for expression estimation
	for (int i=60;i<68;i++) lmVisInd.push_back(i);
	landIm = cv::Mat::zeros( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landIm.at<float>(i,0) = lms.at<float>(ind,0);
		landIm.at<float>(i,1) = lms.at<float>(ind,1);
	}

	float bCost, cCost, fCost;
	int bestIter = 0;
	bCost = 10000.0f;

	params.weightLM = 8.0f;
	Mat alpha0;
	int iter=0;
	int badCount = 0;
	memset(params.doOptimize,true,sizeof(bool)*6);

	// optimize pose+expression from landmarks
	int EM = 29;
	float renderParams_tmp[RENDER_PARAMS_COUNT];

	for (;iter<60;iter++) {
			if (iter%20 == 0) {
				cCost = updateHessianMatrix(alpha,renderParams,faces,colorIm,lmVisInd,landIm,params, exprW);
				if (countFail > 10) {
					countFail = 0;
					break;
				}
			}
			sno_step(alpha, renderParams, faces,colorIm,lmVisInd,landIm,params,exprW);
		}
	iter = 60;

	// optimize expression only
	memset(params.doOptimize,false,sizeof(bool)*6);countFail = 0;
	for (;iter<200;iter++) {
			if (iter%60 == 0) {
				cCost = updateHessianMatrix(alpha,renderParams,faces,colorIm,lmVisInd,landIm,params, exprW);
				if (countFail > 10) {
					countFail = 0;
					break;
				}
			}
			sno_step(alpha, renderParams, faces,colorIm,lmVisInd,landIm,params,exprW);
		}
}

// Render texture-less face mesh
//     Inputs:
//	   colorIm    : Input image
//         alpha      : Subject-specific shape parameters (99x1)
//         r          : Rotation angles (3x1)
//         t          : Translation vector (3x1)
//         exprW      : Expression parameters (29x1)
//     Outputs:
//         Rendered texture-less face w/ the defined shape, expression, and pose
cv::Mat FaceServices2::renderShape(cv::Mat colorIm, cv::Mat alpha,cv::Mat vecR,cv::Mat vecT,cv::Mat exprW){
	float renderParams[RENDER_PARAMS_COUNT];
	for (int i =0;i<3;i++)
		renderParams[i] = vecR.at<float>(i,0);
	for (int i =0;i<3;i++)
		renderParams[i+3] = vecT.at<float>(i,0);
	
	// Ambient
	renderParams[RENDER_PARAMS_AMBIENT] = 0.69225;
	renderParams[RENDER_PARAMS_AMBIENT+1] = 0.69225;
	renderParams[RENDER_PARAMS_AMBIENT+2] = 0.69225;
	// Diffuse
	renderParams[RENDER_PARAMS_DIFFUSE] = 0.30754;
	renderParams[RENDER_PARAMS_DIFFUSE+1] = 0.30754;
	renderParams[RENDER_PARAMS_DIFFUSE+2] = 0.30754;
	// LIGHT
	renderParams[RENDER_PARAMS_LDIR] = 3.1415/4;
	renderParams[RENDER_PARAMS_LDIR+1] = 3.1415/4;
	// OTHERS
	renderParams[RENDER_PARAMS_CONTRAST] = 1;
	renderParams[RENDER_PARAMS_GAIN] = renderParams[RENDER_PARAMS_GAIN+1] = renderParams[RENDER_PARAMS_GAIN+2] = RENDER_PARAMS_GAIN_DEFAULT;
	renderParams[RENDER_PARAMS_OFFSET] = renderParams[RENDER_PARAMS_OFFSET+1] = renderParams[RENDER_PARAMS_OFFSET+2] = RENDER_PARAMS_OFFSET_DEFAULT;
		
	float* r = renderParams + RENDER_PARAMS_R;
	float* t = renderParams + RENDER_PARAMS_T;

	shape = festimator.getShape2(alpha,exprW);
	im_render->copyShape(shape);

	// estimate shaded colors
	cv::Mat colors;
	tex = shape*0 + 128;
	rs.estimateColor(shape,tex,faces,renderParams,colors);
	im_render->copyColors(colors);
	im_render->loadModel();

	// render
	cv::Mat outRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	cv::Mat outDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
	im_render->render(r,t,_k[4],outRGB,outDepth);
	return outRGB;
}

// Render texture-less face mesh
//     Inputs:
//	   colorIm    : Input image
//         alpha      : Subject-specific shape parameters (99x1)
//         r          : Rotation angles (3x1)
//         t          : Translation vector (3x1)
//         exprW      : Expression parameters (29x1)
//     Outputs:
//         outRGB      : Rendered texture-less face w/ the defined shape, expression, and pose
//         outDepth   : The corresponding Z-buffer
void FaceServices2::renderShape(cv::Mat colorIm, cv::Mat alpha,cv::Mat vecR,cv::Mat vecT,cv::Mat exprW, cv::Mat &outRGB, cv::Mat &outDepth){
	float renderParams[RENDER_PARAMS_COUNT];
	for (int i =0;i<3;i++)
		renderParams[i] = vecR.at<float>(i,0);
	for (int i =0;i<3;i++)
		renderParams[i+3] = vecT.at<float>(i,0);
	
	// Ambient
	renderParams[RENDER_PARAMS_AMBIENT] = 0.69225;
	renderParams[RENDER_PARAMS_AMBIENT+1] = 0.69225;
	renderParams[RENDER_PARAMS_AMBIENT+2] = 0.69225;
	// Diffuse
	renderParams[RENDER_PARAMS_DIFFUSE] = 0.30754;
	renderParams[RENDER_PARAMS_DIFFUSE+1] = 0.30754;
	renderParams[RENDER_PARAMS_DIFFUSE+2] = 0.30754;
	// LIGHT
	renderParams[RENDER_PARAMS_LDIR] = 3.1415/4;
	renderParams[RENDER_PARAMS_LDIR+1] = 3.1415/4;
	// OTHERS
	renderParams[RENDER_PARAMS_CONTRAST] = 1;
	renderParams[RENDER_PARAMS_GAIN] = renderParams[RENDER_PARAMS_GAIN+1] = renderParams[RENDER_PARAMS_GAIN+2] = RENDER_PARAMS_GAIN_DEFAULT;
	renderParams[RENDER_PARAMS_OFFSET] = renderParams[RENDER_PARAMS_OFFSET+1] = renderParams[RENDER_PARAMS_OFFSET+2] = RENDER_PARAMS_OFFSET_DEFAULT;
		
	float* r = renderParams + RENDER_PARAMS_R;
	float* t = renderParams + RENDER_PARAMS_T;

	shape = festimator.getShape2(alpha,exprW);
	im_render->copyShape(shape);

	// estimate shaded colors
	cv::Mat colors;
	tex = shape*0 + 128;
	rs.estimateColor(shape,tex,faces,renderParams,colors);
	im_render->copyColors(colors);
	im_render->loadModel();

	// render
	outRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	outDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
	im_render->render(r,t,_k[4],outRGB,outDepth);
}

// Get next motion for the animated face visualization. In this sample code, only rotation is changed
//     Inputs:
//         currFrame  : current frame index. Will be increased after this call
//     Outputs:
//         vecR       : Rotation angles (3x1)
//         vecT       : Translation vector (3x1)
//         exprWeights: Expression parameters (29x1)
void FaceServices2::nextMotion(int &currFrame, cv::Mat &vecR, cv::Mat &vecT, cv::Mat &exprWeights){
    float stepYaw = 1;
    float PPI = 3.141592;
    int maxYaw = 70/stepYaw;
    float stepPitch = 1;
    int maxPitch = 45/stepPitch;

    int totalFrames = 4 * (maxYaw + maxPitch);
    currFrame = (currFrame + 1) % totalFrames;
    vecR = vecR*0 + 0.00001;
    // Rotate left
    if (currFrame < maxYaw) vecR.at<float>(1,0) = -currFrame*stepYaw * PPI/180;
    else if (currFrame < 2*maxYaw) vecR.at<float>(1,0) = -stepYaw * (2*maxYaw - currFrame) * PPI/180;
    // Rotate right
    else if (currFrame < 3*maxYaw) vecR.at<float>(1,0) = (currFrame-2*maxYaw) * stepYaw * PPI/180;
    else if (currFrame < 4*maxYaw) vecR.at<float>(1,0) = stepYaw * (4*maxYaw - currFrame) * PPI/180;
    
    // Rotate up
    else if (currFrame < 4*maxYaw + maxPitch) vecR.at<float>(0,0) = -(currFrame-4*maxYaw)*stepPitch * PPI/180;
    else if (currFrame < 4*maxYaw + 2*maxPitch) vecR.at<float>(0,0) = -stepPitch * (4*maxYaw+2*maxPitch - currFrame) * PPI/180;
    // Rotate right
    else if (currFrame < 4*maxYaw + 3*maxPitch) vecR.at<float>(0,0) = (currFrame-4*maxYaw-2*maxPitch) * stepPitch * PPI/180;
    else if (currFrame < 4*maxYaw + 4*maxPitch) vecR.at<float>(0,0) = stepPitch * (4*maxPitch+4*maxYaw - currFrame) * PPI/180;
}
	
// Adding background to the rendered face image
//     Inputs:
//         bg         : Background image 
//         depth      : Z-buffer of the rendered face
//     Input & output:
//         target     : The rendered face image
void FaceServices2::mergeIm(cv::Mat* target,cv::Mat bg,cv::Mat depth){
	for (int i=0;i<bg.rows;i++){
		for (int j=0;j<bg.cols;j++){
			if (depth.at<float>(i,j) >= 0.9999)
				target->at<Vec3b>(i, j) = bg.at<Vec3b>(i,j);
		}
	}
}

//////////////////////////////////////////// Supporting functions ///////////////////////////////////////////////////
// Compute Hessian matrix diagonal
float FaceServices2::updateHessianMatrix(cv::Mat alpha, float* renderParams, cv::Mat faces, cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat exprW ){
	int M = alpha.rows;
	int EM = exprW.rows;
	float step;
	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	params.hessDiag.release();
	params.hessDiag = cv::Mat::zeros(2*M+EM+RENDER_PARAMS_COUNT,1,CV_32F);
	cv::Mat alpha2, expr2;
	float renderParams2[RENDER_PARAMS_COUNT];

	cv::Mat rVec(3,1,CV_32F,renderParams+RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F,renderParams+RENDER_PARAMS_T);
	cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	float currEF = eF(alpha, lmInds, landIm, renderParams, exprW);
	cEF = currEF;

	// expr
	step = mstep*5;
	if (params.optimizeExpr) {
		for (int i=0;i<EM; i++){
			expr2.release(); expr2 = exprW.clone();
			expr2.at<float>(i,0) += step;
			float tmpEF1 = eF(alpha, lmInds, landIm, renderParams,expr2);
			expr2.at<float>(i,0) -= 2*step;
			float tmpEF2 = eF(alpha, lmInds, landIm, renderParams,expr2);
			params.hessDiag.at<float>(2*M+i,0) = params.weightLM * (tmpEF1 - 2*currEF + tmpEF2)/(step*step) 
				+ params.weightRegExpr * 2/(0.25f*29) ;
		}
	}
	// r
	step = mstep*2;
	if (params.doOptimize[RENDER_PARAMS_R]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_R+i] += step;
			float tmpEF1 = eF(alpha, lmInds, landIm, renderParams2,exprW);

			renderParams2[RENDER_PARAMS_R+i] -= 2*step;
			float tmpEF2 = eF(alpha, lmInds, landIm, renderParams2,exprW);
			params.hessDiag.at<float>(2*M+EM+i,0) = params.weightLM * (tmpEF1 - 2*currEF + tmpEF2)/(step*step) + 2.0f/params.weightReg[RENDER_PARAMS_R+i];

		}
	}
	// t
	step = mstep*10;
	if (params.doOptimize[RENDER_PARAMS_T]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_T+i] += step;
			float tmpEF1 = eF(alpha, lmInds, landIm, renderParams2,exprW);
			renderParams2[RENDER_PARAMS_T+i] -= 2*step;
			float tmpEF2 = eF(alpha, lmInds, landIm, renderParams2,exprW);
			params.hessDiag.at<float>(2*M+EM+RENDER_PARAMS_T+i,0) = params.weightLM * (tmpEF1 - 2*currEF + tmpEF2)/(step*step) 
				+ 2.0f/params.weightReg[RENDER_PARAMS_T+i];
		}
	}
	return 0;
}

// Compute gradient
cv::Mat FaceServices2::computeGradient(cv::Mat alpha, float* renderParams, cv::Mat faces,cv::Mat colorIm, std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, std::vector<int> &inds, cv::Mat exprW){
	int M = alpha.rows;
	int EM = exprW.rows;
	int nTri = 40;
	float step;
	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	cv::Mat out(2*M+EM+RENDER_PARAMS_COUNT,1,CV_32F);

	cv::Mat alpha2, expr2;
	float renderParams2[RENDER_PARAMS_COUNT];
	cv::Mat rVec(3,1,CV_32F,renderParams+RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F,renderParams+RENDER_PARAMS_T);
	cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	float currEF = eF(alpha, lmInds, landIm, renderParams,exprW);
	cEF = currEF;
	
	#pragma omp parallel for
	for (int target=0;target<EM+6; target++){
	  if (target < EM) {
		// expr
		float step = mstep*5;
		if (params.optimizeExpr) {
				int i = target;
				std::vector<cv::Point2f> pPoints;
				cv::Mat expr2 = exprW.clone();
				expr2.at<float>(i,0) += step;
				float tmpEF = eF(alpha, lmInds, landIm, renderParams,expr2);
				out.at<float>(2*M+i,0) = params.weightLM * (tmpEF - currEF)/step
					+ params.weightRegExpr * 2*exprW.at<float>(i,0)/(0.25f*29);
		}
	   }
	   else if (target < EM+3) {
		// r
		float step = mstep*2;
		if (params.doOptimize[RENDER_PARAMS_R]) {
			int i = target-EM;
				float renderParams2[RENDER_PARAMS_COUNT];
				memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
				renderParams2[RENDER_PARAMS_R+i] += step;
				float tmpEF = eF(alpha, lmInds, landIm, renderParams2,exprW);
				out.at<float>(2*M+EM+i,0) = params.weightLM * (tmpEF - currEF)/step;
				out.at<float>(2*M+EM+i,0) += 2*(renderParams[RENDER_PARAMS_R+i] - params.initR[RENDER_PARAMS_R+i])/params.weightReg[RENDER_PARAMS_R+i];
		}
	  }
	  else {
		// t
		float step = mstep*10;
		if (params.doOptimize[RENDER_PARAMS_T]) {
			int i = target-EM-3;
				float renderParams2[RENDER_PARAMS_COUNT];
				memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
				renderParams2[RENDER_PARAMS_T+i] += step;
				float tmpEF = eF(alpha, lmInds, landIm, renderParams2,exprW);
				out.at<float>(2*M+EM+RENDER_PARAMS_T+i,0) = params.weightLM * (tmpEF - currEF)/step 
					+ 2*(renderParams[RENDER_PARAMS_T+i] - params.initR[RENDER_PARAMS_T+i])/params.weightReg[RENDER_PARAMS_T+i];
		}
	  }
	}
	return out;
}

// Compute landmark error
float FaceServices2::eF(cv::Mat alpha, std::vector<int> inds, cv::Mat landIm, float* renderParams, cv::Mat exprW){
	Mat k_m(3,3,CV_32F,_k);
	cv::Mat mLM = festimator.getLMByAlpha(alpha,-renderParams[RENDER_PARAMS_R+1], inds, exprW);
	
	cv::Mat rVec(3,1,CV_32F, renderParams + RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F, renderParams + RENDER_PARAMS_T);
	std::vector<cv::Point2f> allImgPts;
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );

	cv::projectPoints( mLM, rVec, tVec, k_m, distCoef, allImgPts );
	float err = 0;
	for (int i=0;i<mLM.rows;i++){
		float val = landIm.at<float>(i,0) - allImgPts[i].x;
		err += val*val;
		val = landIm.at<float>(i,1) - allImgPts[i].y;
		err += val*val;
	}
	return sqrt(err/mLM.rows);
}

// Newton optimization step
void FaceServices2::sno_step(cv::Mat &alpha, float* renderParams, cv::Mat faces,cv::Mat colorIm, std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW){
	float lambda = 0.05;
	std::vector<int> inds;
	cv::Mat dE = computeGradient(alpha, renderParams, faces, colorIm, lmInds, landIm, params,inds, exprW);
	params.gradVec.release(); params.gradVec = dE.clone();
	cv::Mat dirMove = dE*0;

	int M = alpha.rows;
	int EM = exprW.rows;
	if (params.optimizeExpr){
		for (int i=0;i<EM;i++)
			if (abs(params.hessDiag.at<float>(2*M+i,0)) > 0.0000001) {
				dirMove.at<float>(2*M+i,0) = - lambda*dE.at<float>(2*M+i,0)/abs(params.hessDiag.at<float>(2*M+i,0));
			}
	}

	for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
		if (params.doOptimize[i]){
			if (abs(params.hessDiag.at<float>(2*M+EM+i,0)) > 0.0000001) {
				dirMove.at<float>(2*M+EM+i,0) = - lambda*dE.at<float>(2*M+EM+i,0)/abs(params.hessDiag.at<float>(2*M+EM+i,0));
			}
		}
	}
	float pc = line_search(alpha, renderParams, dirMove,inds, faces, colorIm, lmInds, landIm, params, exprW, 10);
	if (pc == 0) countFail++;
	else {
		if (params.optimizeExpr){
			for (int i=0;i<EM;i++) {
				exprW.at<float>(i,0) += pc*dirMove.at<float>(i+2*M,0);
				if (exprW.at<float>(i,0) > 3) exprW.at<float>(i,0) = 3;
				else if (exprW.at<float>(i,0) < -3) exprW.at<float>(i,0) = -3;
			}
		}

		for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
			if (params.doOptimize[i]){
				renderParams[i] += pc*dirMove.at<float>(2*M+EM+i,0);
				if (i == RENDER_PARAMS_CONTRAST || (i>=RENDER_PARAMS_AMBIENT && i<RENDER_PARAMS_DIFFUSE+3) ) {
					if (renderParams[i] > 1.0) renderParams[i] = 1.0;
					if (renderParams[i] < 0) renderParams[i] = 0;

				}
				else if (i >= RENDER_PARAMS_GAIN && i<=RENDER_PARAMS_GAIN+3) {
					if (renderParams[i] > 3.0) renderParams[i]  = 3;
					if (renderParams[i] < 0.3) renderParams[i] = 0.3;
				}
			}
		}
	}
}

// Line search optimization
float FaceServices2::line_search(cv::Mat &alpha, float* renderParams, cv::Mat &dirMove,std::vector<int> inds, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW, int maxIters){
	float step = 1.0f;
	float sstep = 2.0f;
	float minStep = 0.0001f;
	cv::Mat alpha2, exprW2;
	float renderParams2[RENDER_PARAMS_COUNT];
	alpha2 = alpha.clone();
	exprW2 = exprW.clone();
	memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);

	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	std::vector<cv::Point2f> pPoints;
	cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	int M = alpha.rows;
	int EM = exprW.rows;
	float ssize = 0;
	for (int i=0;i<dirMove.rows;i++) ssize += dirMove.at<float>(i,0)*dirMove.at<float>(i,0);
	ssize = sqrt(ssize);
	if (ssize > (2*M+EM+RENDER_PARAMS_COUNT)/5.0f) {
		step = (2*M+EM+RENDER_PARAMS_COUNT)/(5.0f * ssize);
		ssize = (2*M+EM+RENDER_PARAMS_COUNT)/5.0f;
	}
	if (ssize < minStep){
		return 0;
	}
	int tstep = floor(log(ssize/minStep));
	if (tstep < maxIters) maxIters = tstep;

	float curCost = computeCost(cEF, alpha, renderParams, params, exprW );

	bool hasNoBound = false;
	int iter = 0;
	for (; iter<maxIters; iter++){
		if (params.optimizeExpr){
			for (int i=0;i<EM;i++) {
				float tmp = exprW.at<float>(i,0) + step*dirMove.at<float>(2*M+i,0);
				if (tmp >= 3) exprW2.at<float>(i,0) = 3;
				else if (tmp <= -3) exprW2.at<float>(i,0) = -3;
				else {
					exprW2.at<float>(i,0) = tmp;
					hasNoBound = true;
				}
			}
		}

		for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
			if (params.doOptimize[i]){
				float tmp = renderParams[i] + step*dirMove.at<float>(2*M+EM+i,0);
				if (i == RENDER_PARAMS_CONTRAST || (i>=RENDER_PARAMS_AMBIENT && i<RENDER_PARAMS_DIFFUSE+3) ) {
					if (tmp > 1.0) renderParams2[i] = 1.0f;
					else if (tmp < -1.0) renderParams2[i] = -1.0f;
					else {
						renderParams2[i] = tmp;
						hasNoBound = true;
					}
				}
				else if (i >= RENDER_PARAMS_GAIN && i<=RENDER_PARAMS_GAIN+3) {
					if (tmp >= 3.0) renderParams2[i] = 3.0f;
					else if (tmp <= -3.0) renderParams2[i] = -3.0f;
					else {
						renderParams2[i] = tmp;
						hasNoBound = true;
					}
				}
				else renderParams2[i] = tmp;
			}
		}
		if (!hasNoBound) {
			iter = maxIters; break;
		}
		float tmpEF = cEF;
		if (params.weightLM > 0) tmpEF = eF(alpha2, lmInds,landIm,renderParams2, exprW2);
		float tmpCost = computeCost(tmpEF, alpha2, renderParams2, params,exprW2);
		if (tmpCost < curCost) {
			break;
		}
		else {
			step = step/sstep;
		}
	}
	if (iter >= maxIters) return 0;
	else return step;
}

// Cost function
float FaceServices2::computeCost(float vEF, cv::Mat &alpha, float* renderParams, BFMParams &params, cv::Mat &exprW ){
	float val = params.weightLM*vEF;
	int M = alpha.rows;
	if (params.optimizeExpr){
		for (int i=0;i<exprW.rows;i++)
			val += params.weightRegExpr * exprW.at<float>(i,0)*exprW.at<float>(i,0)/(0.5f*29);
	}

	for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
		if (params.doOptimize[i]){
			val += (renderParams[i] - params.initR[i])*(renderParams[i] - params.initR[i])/params.weightReg[i];
		}
	}
	return val;
}

