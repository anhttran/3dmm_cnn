/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#ifndef CVGL_FBRENDER_H
#define CVGL_FBRENDER_H

#include <iostream>
#include <cstdlib>

#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES 1
#endif
//GLEW needs to be first
#include <GL/glew.h>//
#include <GL/glu.h>
#include <GL/glext.h>//
#include <GL/glut.h>
//#include <GL/osmesa.h>
//#include <cuda_gl_interop.h>
//
//#include <cutil_inline.h>

#include "cv.h"
#include "highgui.h"

#include "SimpleMesh.h"
#include "Material.h"

//#if defined(_WIN32) || defined(_WIN64)
//#include <windows.h>
//#include <tchar.h>
//#else
//#include <X11/X.h>
//#include <X11/Xlib.h>
//#include <GL/glx.h>
//#endif

class FBRender {

	int wdId;
	int wdId2;
	//void initOSMesaContext();
//	OSMesaContext ctx;
protected:
	
	//off-screen framebuffer size
	unsigned int fbWidth;
	unsigned int fbHeight;
	
	//these are set to small values, since we are not even using the window
	static const unsigned int winWidth;
	static const unsigned int winHeight;
	
	Material mat;

	GLuint g_frameBuffer;
	GLuint g_depthRenderBuffer;
	GLuint g_dynamicTextureID;
	//cudaGraphicsResource_t cuResource;
	
	//void createContext(bool showWindow);

	void initGL();
	void initGLBuffers();
	
public:
	
	FBRender( int width, int height, bool showWindow = false);
	~FBRender(); 
    void init(int width, int height, bool showWindow = false);  // Yuval

	void mapRendering( SimpleMesh& mesh );
	void unmapRendering();
	
	GLuint getFrameBufID(){ return g_frameBuffer; }
	GLuint getDepthBufID(){ return g_depthRenderBuffer; }
	GLuint getTextBufID(){ return g_dynamicTextureID; }

	virtual void loadIntrinGL(CvMat *K, double znear, double zfar, int imgWidth, int imgHeight);
	virtual void loadExtrinGL(CvMat *H);

	virtual void readFB( cv::Mat &img );
	virtual void readDB( cv::Mat &depth );
	
	void checkProjection(double x, double y, double z);
	void checkModelView(double x, double y, double z);
	void checkClip(double x, double y, double z);
};

#endif
