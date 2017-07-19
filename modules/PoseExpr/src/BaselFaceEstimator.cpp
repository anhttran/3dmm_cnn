/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "BaselFaceEstimator.h"
#include <string.h>
#include "utility.h"
#include "epnp.h"
#include <vector>

using namespace cv;

BaselFaceEstimator::BaselFaceEstimator()
{
}

bool BaselFaceEstimator::load3DMM(const std::string& model_file){
	bf.load(model_file);
}

// Compute 3D object (shape/texture/expression) from weights, given 3DMM basis (MU, PCs, EV). Ouput size Vx3.
//       The input vector "weight" is vertical, with float numbers
cv::Mat BaselFaceEstimator::coef2object(cv::Mat weight, cv::Mat MU, cv::Mat PCs, cv::Mat EV){
	int M = weight.rows;
	Mat tmpShape;
	if (M == 0) 
		tmpShape = MU.clone();
	else {
		Mat subPC = PCs(Rect(0,0,M,PCs.rows));
		Mat subEV = EV(Rect(0,0,1,M));
		tmpShape = MU + subPC * weight.mul(subEV);
	}
	return tmpShape.reshape(1,tmpShape.rows/3);
}

BaselFaceEstimator::~BaselFaceEstimator(void)
{
}

// Get triangle connectivity. Output size Fx3
cv::Mat BaselFaceEstimator::getFaces(){
	cv::Mat out;
	bf.faces.convertTo(out, CV_32S);
	return out;
}

// Get 3D shape given subject-specific weight (99x1) + expression weight (29x1). Output size Vx3
//        Use only 1 CPU
cv::Mat BaselFaceEstimator::getShape(cv::Mat weight, cv::Mat exprWeight){
	return coef2object(weight,bf.shapeMU,bf.shapePC,bf.shapeEV) + coef2object(exprWeight,bf.exprMU,bf.exprPC,bf.exprEV);
}

// Get 3D shape given subject-specific weight (99x1) + expression weight (29x1). Output size Vx3
//        Use multiple CPUs
cv::Mat BaselFaceEstimator::getShape2(cv::Mat alpha, cv::Mat exprWeight){
	cv::Mat alpha2 = alpha.clone();
	cv::Mat exp2 = exprWeight.clone();
	for (int i=0;i<alpha.rows;i++) alpha2.at<float>(i,0) *= bf.shapeEV.at<float>(i,0);
	for (int i=0;i<exp2.rows;i++) exp2.at<float>(i,0) *= bf.exprEV.at<float>(i,0);
	int N = bf.shapePC.rows/3;
	Mat tmpShape(N,3,CV_32F);
	
	#pragma omp parallel for
	for (int i=0;i<N;i++){
		for (int j=0;j<3;j++) {
			float val = bf.shapeMU.at<float>(3*i+j,0) + bf.exprMU.at<float>(3*i+j,0);
			int k=0;
                        const float* pPC = bf.shapePC.ptr<float>(3*i+j);
			// Speed up computation w/ CPU vectorization
			for (;k<=alpha2.rows-5;k+=5) {
				val += alpha2.at<float>(k,0) * pPC[k] + alpha2.at<float>(k+1,0) * pPC[k+1]
					+ alpha2.at<float>(k+2,0) * pPC[k+2]
					+ alpha2.at<float>(k+3,0) * pPC[k+3]
					+ alpha2.at<float>(k+4,0) * pPC[k+4];
			}
			for (;k<alpha2.rows;k++) {
				val += alpha2.at<float>(k,0) * pPC[k];
			}
                        const float* pEPC = bf.exprPC.ptr<float>(3*i+j);
			for (k = 0;k<=exp2.rows-5;k+=5) {
				val += exp2.at<float>(k,0) * pEPC[k] + exp2.at<float>(k+1,0) * pEPC[k+1]
					+ exp2.at<float>(k+2,0) * pEPC[k+2]
					+ exp2.at<float>(k+3,0) * pEPC[k+3]
					+ exp2.at<float>(k+4,0) * pEPC[k+4];
			}
			for (;k<exp2.rows;k++) {
				val += exp2.at<float>(k,0) * pEPC[k];
			}
			tmpShape.at<float>(i,j) = val;
		}
	}
	alpha2.release();
	exp2.release();
	return tmpShape;
}

// Get 3D texture given subject-specific weight (99x1). Output size Vx3
//        Use only 1 CPU
cv::Mat BaselFaceEstimator::getTexture(cv::Mat weight){
	return coef2object(weight,bf.texMU,bf.texPC,bf.texEV);
}

// Get 3D landmarks points (68x3), given 3D shape (Vx3) and yaw angle
cv::Mat BaselFaceEstimator::getLM(cv::Mat shape, float yaw){
	cv::Mat lm(bf.lmInd.rows,3,CV_32F);
	for (int i=0;i<lm.rows;i++){
		int ind;
		if (yaw > 0 && yaw < M_PI/9 && i < 8) ind = bf.lmInd.at<int>(i,0)-1;
		else if (yaw < 0 && yaw > -M_PI/9 && i > 8 && i < 17) ind = bf.lmInd.at<int>(i,0)-1;
		else
		ind = bf.lmInd2.at<int>(i)-1;
		for (int j=0;j<3;j++){
			lm.at<float>(i,j) = shape.at<float>(ind,j);
		}
	}
	return lm;
}

// Get 3D landmarks points (Ux3). 
//    Inputs:
//        alpha      : subject-specific weight (99x1) 
//        yaw        : yaw angle
//        inds       : list of LM to be used. If inds is empty, use all LM points (0-67)
//        exprWeight : expression weight (29x1)
cv::Mat BaselFaceEstimator::getLMByAlpha(cv::Mat alpha, float yaw, std::vector<int> inds, cv::Mat exprWeight){
	cv::Mat alpha2 = alpha.clone();
	cv::Mat exp2 = exprWeight.clone();
	if (inds.size() == 0){
		for (int i=0;i<68;i++) inds.push_back(i);
	}
	for (int i=0;i<alpha.rows;i++) alpha2.at<float>(i,0) *= bf.shapeEV.at<float>(i,0);
	for (int i=0;i<exp2.rows;i++) exp2.at<float>(i,0) *= bf.exprEV.at<float>(i,0);
	int N = inds.size();
	Mat tmpShape(N,3,CV_32F);
	float val;
	for (int i=0;i<inds.size();i++){
		// Get vertex index by yaw angle
		int ind;
		if (yaw > 0 && yaw < M_PI/9 && i < 8) ind = bf.lmInd.at<int>(inds[i],0)-1;
		else if (yaw < 0 && yaw > -M_PI/9 && i > 8 && i < 17) ind = bf.lmInd.at<int>(inds[i],0)-1;
		else
		ind = bf.lmInd2.at<int>(inds[i],0)-1;

		for (int j=0;j<3;j++) {
			val = bf.shapeMU.at<float>(3*ind+j,0) + bf.exprMU.at<float>(3*ind+j,0);
                        const float* pPC = bf.shapePC.ptr<float>(3*ind+j);
			// Speed up computation w/ CPU vectorization
			int k=0;
			for (;k<=alpha2.rows-5;k+=5) {
				val += alpha2.at<float>(k,0) * pPC[k] + alpha2.at<float>(k+1,0) * pPC[k+1]
					+ alpha2.at<float>(k+2,0) * pPC[k+2]
					+ alpha2.at<float>(k+3,0) * pPC[k+3]
					+ alpha2.at<float>(k+4,0) * pPC[k+4];
			}
			for (;k<alpha2.rows;k++) {
				val += alpha2.at<float>(k,0) * pPC[k];
			}
                        const float* pEPC = bf.exprPC.ptr<float>(3*ind+j);
			for (k = 0;k<=exp2.rows-5;k+=5) {
				val += exp2.at<float>(k,0) * pEPC[k] + exp2.at<float>(k+1,0) * pEPC[k+1]
					+ exp2.at<float>(k+2,0) * pEPC[k+2]
					+ exp2.at<float>(k+3,0) * pEPC[k+3]
					+ exp2.at<float>(k+4,0) * pEPC[k+4];
			}
			for (;k<exp2.rows;k++) {
				val += exp2.at<float>(k,0) * pEPC[k];
			}
			tmpShape.at<float>(i,j) = val;
		}
	}
	alpha2.release();
	exp2.release();
	return tmpShape;
}

// Estimate 3D pose. 
//    Inputs:
//        landModel  : 3D landmarks points (Ux3)
//        landImage  : 2D landmarks points (Ux3)
//        k_m        : instrinsic camera matrix
//        exprWeight : expression weight (29x1)
//    Outputs:
//        r          : rotation angles (3x1)
//        t          : trnslation vector (3x1)
void BaselFaceEstimator::estimatePose3D(cv::Mat landModel, cv::Mat landImage, cv::Mat k_m, cv::Mat &r, cv::Mat &t){
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );

	double R_est[3][3], t_est[3];
	cv::Mat rVec( 3, 1, CV_64F );
	cv::Mat tVec( 3, 1, CV_64F );	

	epnp PnP;
	PnP.set_internal_parameters(k_m.at<float>(0,2),k_m.at<float>(1,2),k_m.at<float>(0,0),k_m.at<float>(1,1));
	PnP.set_maximum_number_of_correspondences(landImage.rows);

	std::vector<cv::Point3f> allObjPts;
	std::vector<cv::Point2f> allObj2DPts;
	std::vector<cv::Point2f> allImgPts;
	for ( int i=0; i<landModel.rows; ++i )
	{
		allObjPts.push_back(Point3f(landModel.at<float>(i,0),landModel.at<float>(i,1),landModel.at<float>(i,2)));
		allObj2DPts.push_back(Point2f(landImage.at<float>(i,0),landImage.at<float>(i,1)));
	}
	PnP.reset_correspondences();
	for(int ind = 0; ind < landImage.rows; ind++){
		PnP.add_correspondence(allObjPts[ind].x,allObjPts[ind].y,allObjPts[ind].z, allObj2DPts[ind].x,allObj2DPts[ind].y);
	}
	double err2 = PnP.compute_pose(R_est, t_est);
	cv::Mat rMatP( 3, 3, CV_64F, R_est);
	cv::Rodrigues(rMatP, rVec);
	rVec.convertTo(rVec,CV_32F);
	cv::Mat tVecP( 3, 1, CV_64F, t_est);
	tVec = tVecP.clone();
	tVec.convertTo(tVec,CV_32F);

	r = rVec.clone();
	t = tVec.clone();
}

