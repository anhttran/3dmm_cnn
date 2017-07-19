/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#pragma once
#include "cv.h"
#include "highgui.h"
#include "FTModel.h"
#include "BaselFace.h"

// A set of method to compose 3D model given shape/texture/expression weights
class BaselFaceEstimator
{

// Compute 3D object (shape/texture/expression) from weights, given 3DMM basis (MU, PCs, EV). Ouput size Vx3.
//       The input vector "weight" is vertical, with float numbers
	cv::Mat coef2object(cv::Mat weight, cv::Mat MU, cv::Mat PCs, cv::Mat EV);
	BaselFace bf;

public:
	BaselFaceEstimator();

	bool load3DMM(const std::string& model_file);

// Get triangle connectivity. Output size Fx3
	cv::Mat getFaces();

// Get water-tight triangle connectivity. Output size F'x3
	cv::Mat getFaces_fill();

// Get 3D shape given subject-specific weight (99x1) + expression weight (29x1). Output size Vx3
//        Use only 1 CPU
	cv::Mat getShape(cv::Mat weight, cv::Mat exprWeight = cv::Mat());

// Get 3D shape given subject-specific weight (99x1) + expression weight (29x1). Output size Vx3
//        Use multiple CPUs
	cv::Mat getShape2(cv::Mat weight, cv::Mat exprWeight = cv::Mat());

// Get 3D texture given subject-specific weight (99x1). Output size Vx3
//        Use only 1 CPU
	cv::Mat getTexture(cv::Mat weight);

// Get 3D landmarks points (68x3), given 3D shape (Vx3) and yaw angle
	cv::Mat getLM(cv::Mat shape, float yaw);

// Get 3D landmarks points (Ux3). 
//    Inputs:
//        alpha      : subject-specific weight (99x1) 
//        yaw        : yaw angle
//        inds       : list of LM to be used. If inds is empty, use all LM points (0-67)
//        exprWeight : expression weight (29x1)
	cv::Mat getLMByAlpha(cv::Mat alpha, float yaw, std::vector<int> inds, cv::Mat exprWeight = cv::Mat());


// Estimate 3D pose. 
//    Inputs:
//        landModel  : 3D landmarks points (Ux3)
//        landImage  : 2D landmarks points (Ux3)
//        k_m        : instrinsic camera matrix
//        exprWeight : expression weight (29x1)
//    Outputs:
//        r          : rotation angles (3x1)
//        t          : trnslation vector (3x1)
	void estimatePose3D(cv::Mat landModel, cv::Mat landImage, cv::Mat k_m, cv::Mat &r, cv::Mat &t);

	~BaselFaceEstimator(void);
};

