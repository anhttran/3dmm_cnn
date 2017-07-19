/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#pragma once
#include "cv.h"
#include "highgui.h"

#define RENDER_PARAMS_COUNT 21

#define RENDER_PARAMS_R 0
#define RENDER_PARAMS_T 3
#define RENDER_PARAMS_AMBIENT	6
#define RENDER_PARAMS_DIFFUSE	9
#define RENDER_PARAMS_LDIR	12
#define RENDER_PARAMS_CONTRAST	14
#define RENDER_PARAMS_GAIN	15
#define RENDER_PARAMS_OFFSET	18
#define RENDER_PARAMS_SPECULAR	21
#define RENDER_PARAMS_SHINENESS	24

#define RENDER_PARAMS_AMBIENT_DEFAULT	0.5f
#define RENDER_PARAMS_DIFFUSE_DEFAULT	0.5f
#define RENDER_PARAMS_CONTRAST_DEFAULT	1.0f
#define RENDER_PARAMS_GAIN_DEFAULT	1.0f
#define RENDER_PARAMS_OFFSET_DEFAULT	0.0f
#define RENDER_PARAMS_SPECULAR_DEFAULT	80.0f
#define RENDER_PARAMS_SHINENESS_DEFAULT	0.8f

class RenderServices
{
	// Given 3D shape (Vx3) and triangle connectivity (Fx3), compute at a specific vertex (face_id, vertex_id) its normals (nx, ny, nz)
	bool triangleNormalFromVertex(cv::Mat shape, cv::Mat faces, int face_id, int vertex_id, float &nx, float &ny, float &nz);
	// Given 3D shape (Vx3) and triangle connectivity (Fx3), compute at a specific triangle (face_id) its normals (nx, ny, nz)
	float triangleNormal(cv::Mat shape, cv::Mat faces, int face_id, float &nx, float &ny, float &nz);
public:
	// Given 3D shape(Vx3) and triangle connectivity (Fx3), estimates vertex normals (Vx3)
	bool estimateVertexNormals(cv::Mat shape, cv::Mat faces, cv::Mat &normals);
	
	// Estimate vertex normals
	//     Inputs:
	//         shape        : 3D shape (Vx3)
	//         tex          : 3D texture (Vx3)
	//         faces        : triangle connectivity
        //         render_model : rendering parameters
        //     Outputs:
        //         colors       : shaded vertex colors
	bool estimateColor(cv::Mat shape, cv::Mat tex, cv::Mat faces, float* render_model, cv::Mat &colors);

	// Estimate vertex normals + return computed vertex normals
	bool estimateColor(cv::Mat shape, cv::Mat tex, cv::Mat faces, float* render_model, cv::Mat &colors, cv::Mat &normals);
};
