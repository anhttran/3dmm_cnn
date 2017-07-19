/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
//----------------------------------------------------------------------
// File: FTModel.h
//      Authors:  Yann Dumortier (yann.dumortier@gmail.com),
//                Jongmoo Choi (jongmooc@usc.edu),
//                Sang-il Choi (csichoisi@gmail.net)
// Description: 
// This file is part of the "Real-time 3D Face Tracking and Modeling Using a Webcam" 
//      developed at the University of Southern California by:
//
// Yann Dumortier (yann.dumortier@gmail.com),
// Jongmoo Choi (jongmooc@usc.edu),
// Gerard Medioni (medioni@usc.edu).
//----------------------------------------------------------------------
//      Copyright (c) 2011 University of Southern California.  All Rights Reserved.
//



#ifndef FTMODEL_H
#define FTMODEL_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cfloat>

#include <cv.h>
#include <highgui.h>

#include "SimpleMesh.h"
#include "utility.h"
//#include "ASM.h"

#define MESH_COLOR	1
#define MESH_NORMAL	2

#define PROP_X		0
#define PROP_Y		1
#define PROP_Z		2
#define PROP_R		3
#define PROP_G		4
#define PROP_B		5
#define PROP_NX		6
#define PROP_NY		7
#define PROP_NZ		8

#define MODEL_SCALE 15.f
#define MODEL_TX .0f
#define MODEL_TY .3f
#define MODEL_TZ .75f

#define MAX_ERR 5.f
#define MAX_LDMKS 85
class StatiCam
{
public:
	StatiCam( float f=0.f, float cx=0.f, float cy=.0f ): f_( f ), cx_( cx ), cy_( cy ){}
	~StatiCam(){}

	int calibIntWoRot( float cx, float cy, unsigned n, float *pts2D, float *pts3D, float *t );
	int calibIntWoRot2( float cx, float cy, unsigned n, float *pts2D, float *pts3D, float *t );
	
public:
	float f_;
	float cx_, cy_;
};

class Face
{
public:
	Face( unsigned id=0 );
	Face( Face &f, unsigned id=0 );
	~Face();
	
	float& tx() { return t_[0]; }
	float& ty() { return t_[1]; }
	float& tz() { return t_[2]; }
	
	float& rx() { return R_[0]; }
	float& ry() { return R_[1]; }
	float& rz() { return R_[2]; }

	int loadPLYModel( const char* fileName );
	int loadPLYModel2( const char* fileName );
	int loadPLYLandmarks( const char* fileName );
	void loadMesh( cv::Mat shape, cv::Mat tex, cv::Mat faces );
	void savePLYModel( const char* fileName );
	bool estimateNormals( );
private:
	void triangleNormalFromVertex(int face_id, int vertex_id, float &nx, float &ny, float &nz);
	int invalidPLYFile()
	{
		std::cerr << "Invalid PLY file.\n";
		return -1;
	}

public:
	unsigned id_;					//Face ID, 0 means generic face

	SimpleMesh mesh_;				//3D Face mesh
	
	int landmarks_[MAX_LDMKS];		//Face landmark vertices' index
	unsigned nLdmks_;				//Number of landmark

	float *R_;
	float *t_;
};

class CylCoord {
public:
	uchar r, g, b;
	float depth, theta, y_cyl;

	CylCoord() {
		r= g= b= depth= theta= y_cyl = 0;
	}

	CylCoord( uchar r, uchar g, uchar b, float depth, float theta, float y_cyl ) {
		this->r = r;
		this->g = g;
		this->b = b;

		this->depth = depth;
		this->theta = theta;
		this->y_cyl = y_cyl;
	}

};

#endif /*FACEMODEL_H*/
