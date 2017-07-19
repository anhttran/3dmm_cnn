/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#pragma once
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>

#include "FTModel.h"
#include "FBRender.h"

// Face renderer 
class FImRenderer
{	
public:
	FBRender *render_;
	Face* face_;
	cv::Mat img_;
	float zNear, zFar;

	// Initiate with input image
	FImRenderer(cv::Mat img);
	void init(const cv::Mat & img);

	// Load face model from file
	void loadPLYFile(char* filename, bool useTexture = false);
	// Load face model from matrices
	void loadMesh(cv::Mat shape, cv::Mat tex, cv::Mat faces);
	void copyFaces(cv::Mat faces);
	void copyShape(cv::Mat shape);
	void copyColors(cv::Mat colors);
	void copyNormals(cv::Mat normals);
	// Compute vertex normals
	void computeNormals();
	// Load 3D face model into OpenGL
	void loadModel();
	// Render the face w/ OpenGL.
        //    Inputs:
        //        r      : rotation angles (3)
        //        t      : translation vector (3)
        //        f      : focal length
        //    Output: 
        //        color  : rendered RGB image
        //        depth  : depth image from OpenGL depth buffer
	void render( float *r, float *t, float f, cv::Mat &color, cv::Mat &depth );

	// Some supporting methods
	void mapRendering(float* r, float* t, float f, cv::Mat *im, cv::Mat *mask);
	void unmapRendering();
	
	~FImRenderer(void);
};
