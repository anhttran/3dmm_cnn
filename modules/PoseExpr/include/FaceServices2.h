/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#pragma once
#include "cv.h"
#include "highgui.h"
#include "FImRenderer.h"
#include "BaselFaceEstimator.h"
#include "RenderModel.h"
#include <Eigen/Sparse>

using namespace std;
using namespace cv;

typedef struct BFMParams {
	float weightLM;				// weight for LM fitting error
	float weightReg[RENDER_PARAMS_COUNT];   // weights for rendering parameter regularization
	float weightRegExpr;			// weights for expression parameter regularization

	float initR[RENDER_PARAMS_COUNT];	// initial values of rendering parameters
	bool  doOptimize[RENDER_PARAMS_COUNT];  // flags indicating which rendering parameters is optimizing
	bool  optimizeExpr;			// flags indicating whether expression is optimizing

	cv::Mat hessDiag;  			// diagonal of the Hessian matrix
	cv::Mat gradVec;  			// gradient vector
	
	void init(){
		memset(doOptimize,false,sizeof(bool)*RENDER_PARAMS_COUNT);
		optimizeExpr = true;
		weightRegExpr = 1;
		for (int i=0;i<6;i++) doOptimize[i] = true;

		// initial values
		if (RENDER_PARAMS_AMBIENT < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_AMBIENT+i] = RENDER_PARAMS_AMBIENT_DEFAULT;
		}
		if (RENDER_PARAMS_DIFFUSE < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_DIFFUSE+i] = 0.0f;
		}
		if (RENDER_PARAMS_LDIR < RENDER_PARAMS_COUNT){
			for (int i=0;i<2;i++) initR[RENDER_PARAMS_LDIR+i] = 0.0f;
		}
		if (RENDER_PARAMS_CONTRAST < RENDER_PARAMS_COUNT){
			initR[RENDER_PARAMS_CONTRAST] = RENDER_PARAMS_CONTRAST_DEFAULT;
		}
		if (RENDER_PARAMS_GAIN < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_GAIN+i] = RENDER_PARAMS_GAIN_DEFAULT;
		}
		if (RENDER_PARAMS_OFFSET < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_OFFSET+i] = RENDER_PARAMS_OFFSET_DEFAULT;
		}
		if (RENDER_PARAMS_SPECULAR < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) initR[RENDER_PARAMS_SPECULAR+i] = RENDER_PARAMS_SPECULAR_DEFAULT;
		}
		if (RENDER_PARAMS_SHINENESS < RENDER_PARAMS_COUNT){
			initR[RENDER_PARAMS_SHINENESS] = RENDER_PARAMS_SHINENESS_DEFAULT;
		}
		
		// weightReg
		for (int i=0;i<3;i++) weightReg[RENDER_PARAMS_R+i] = (M_PI/6)*(M_PI/6);
		for (int i=0;i<3;i++) weightReg[RENDER_PARAMS_T+i] = 900.0f;
		if (RENDER_PARAMS_AMBIENT < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) weightReg[RENDER_PARAMS_AMBIENT+i] = 1;
		}
		if (RENDER_PARAMS_DIFFUSE < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) weightReg[RENDER_PARAMS_DIFFUSE+i] = 1;
		}
		if (RENDER_PARAMS_LDIR < RENDER_PARAMS_COUNT){
			for (int i=0;i<2;i++) weightReg[RENDER_PARAMS_LDIR+i] = M_PI*M_PI;
		}
		if (RENDER_PARAMS_CONTRAST < RENDER_PARAMS_COUNT){
			weightReg[RENDER_PARAMS_CONTRAST] = 1;
		}
		if (RENDER_PARAMS_GAIN < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) weightReg[RENDER_PARAMS_GAIN+i] = 4.0f;
		}
		if (RENDER_PARAMS_OFFSET < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) weightReg[RENDER_PARAMS_OFFSET+i] = 10000.0f;
		}
		if (RENDER_PARAMS_SPECULAR < RENDER_PARAMS_COUNT){
			for (int i=0;i<3;i++) weightReg[RENDER_PARAMS_SPECULAR+i] = 10000.0f;
		}
		if (RENDER_PARAMS_SHINENESS < RENDER_PARAMS_COUNT){
			weightReg[RENDER_PARAMS_SHINENESS] = 1000000.0f;
		}
	}
} BFMParams;


class FaceServices2
{
	float _k[9];
	FImRenderer* im_render;		// face renderer
	cv::Mat faces, shape, tex;
	BaselFaceEstimator festimator;
	RenderServices rs;
	
	float cEF;
	float mstep;
	int countFail;
	float maxVal;

	// Supporting functions
	float updateHessianMatrix(cv::Mat alpha, float* renderParams, cv::Mat faces, cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat exprW = cv::Mat() );
	cv::Mat computeGradient(cv::Mat alpha, float* renderParams, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params,std::vector<int> &inds, cv::Mat exprW);
	void sno_step(cv::Mat &alpha, float* renderParams, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW);
	float line_search(cv::Mat &alpha, float* renderParams, cv::Mat &dirMove,std::vector<int> inds, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW, int maxIters = 4);
	float computeCost(float vEF, cv::Mat &alpha, float* renderParams, BFMParams &params, cv::Mat &exprW);
	
	float eF(cv::Mat alpha, std::vector<int> inds, cv::Mat landIm, float* renderParams, cv::Mat exprW);
	

public:
	FaceServices2(const std::string & model_file);
	~FaceServices2(void);

	// Setup with image size (w,h) and focal length f
	void init(int w, int h, float f);

	// Estimate pose and expression from landmarks
	//     Inputs:
        //	   colorIm    : Input image
	//         lms        : 2D landmarks (68x2)
	//         alpha      : Subject-specific shape parameters (99x1)
        //     Outputs:
	//         vecR       : Rotation angles (3x1)
	//         vecT       : Translation vector (3x1)
        //         exprWeight : Expression parameters (29x1)
	bool estimatePoseExpr(cv::Mat colorIm, cv::Mat lms, cv::Mat alpha, cv::Mat &vecR, cv::Mat &vecT, cv::Mat &exprWeight);

	// Render texture-less face mesh
	//     Inputs:
        //	   colorIm    : Input image
	//         alpha      : Subject-specific shape parameters (99x1)
	//         r          : Rotation angles (3x1)
	//         t          : Translation vector (3x1)
	//         exprW      : Expression parameters (29x1)
        //     Outputs:
	//         Rendered texture-less face w/ the defined shape, expression, and pose
	cv::Mat renderShape(cv::Mat colorIm, cv::Mat alpha, cv::Mat r, cv::Mat t, cv::Mat exprW);

	// Render texture-less face mesh
	//     Inputs:
        //	   colorIm    : Input image
	//         alpha      : Subject-specific shape parameters (99x1)
	//         r          : Rotation angles (3x1)
	//         t          : Translation vector (3x1)
	//         exprW      : Expression parameters (29x1)
        //     Outputs:
	//         out        : Rendered texture-less face w/ the defined shape, expression, and pose
        //         refDepth   : The corresponding Z-buffer
        void renderShape(cv::Mat colorIm, cv::Mat alpha, cv::Mat r, cv::Mat t, cv::Mat exprW, cv::Mat &out, cv::Mat &refDepth);

	// Get next motion for the animated face visualization. In this sample code, only rotation is changed
	//     Inputs:
	//         currFrame  : current frame index. Will be increased after this call
        //     Outputs:
	//         vecR       : Rotation angles (3x1)
	//         vecT       : Translation vector (3x1)
	//         exprWeights: Expression parameters (29x1)
	void nextMotion(int &currFrame, cv::Mat &vecR, cv::Mat &vecT, cv::Mat &exprWeights);

	// Adding background to the rendered face image
        //     Inputs:
	//         bg         : Background image 
	//         depth      : Z-buffer of the rendered face
        //     Input & output:
	//         target     : The rendered face image
	void mergeIm(cv::Mat* target,cv::Mat bg,cv::Mat depth);
};

