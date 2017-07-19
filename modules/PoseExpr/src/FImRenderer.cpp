/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "FImRenderer.h"
using namespace cv;

// Initiate with input image
FImRenderer::FImRenderer(cv::Mat img)
{
	face_ = new Face();
	img_ = img.clone();
	render_ = new FBRender( img_.cols, img_.rows,true );
	zNear = 50;
	zFar = 10000;
}


void FImRenderer::init(const cv::Mat & img)
{
    img_ = img.clone();
    render_->init(img.cols, img.rows);
}

FImRenderer::~FImRenderer(void)
{
	delete render_;
}

// Load face model from file
void FImRenderer::loadPLYFile(char* filename, bool useTexture){
	face_->loadPLYModel2(filename);
	if (useTexture && face_->mesh_.colors_){
		delete face_->mesh_.colors_;
		face_->mesh_.colors_ = 0;
		face_->mesh_.colorid = 0;
	}
}

// Load face model from matrices
void FImRenderer::loadMesh(cv::Mat shape, cv::Mat tex, cv::Mat faces){
	face_->loadMesh(shape, tex, faces);
}

void FImRenderer::copyColors(cv::Mat colors){
	if (face_->mesh_.colors_ == 0) face_->mesh_.colors_ = new float[4*face_->mesh_.nVertices_];
	for (int i=0;i<face_->mesh_.nVertices_;i++){
		face_->mesh_.colors_ [4*i] = colors.at<float>(i,0)/255.0f;
		face_->mesh_.colors_ [4*i+1] = colors.at<float>(i,1)/255.0f;
		face_->mesh_.colors_ [4*i+2] = colors.at<float>(i,2)/255.0f;
		face_->mesh_.colors_ [4*i+3] = 1.0f;
	}
}

void FImRenderer::copyNormals(cv::Mat normals){
	if (face_->mesh_.normals == 0) face_->mesh_.normals = new float[3*face_->mesh_.nVertices_];
	for (int i=0;i<face_->mesh_.nVertices_;i++){
		face_->mesh_.normals [3*i] = normals.at<float>(i,0);
		face_->mesh_.normals [3*i+1] = normals.at<float>(i,1);
		face_->mesh_.normals [3*i+2] = normals.at<float>(i,2);
	}
}
void FImRenderer::copyShape(cv::Mat shape){
	for (int i=0;i<face_->mesh_.nVertices_;i++){
		face_->mesh_.vertices_ [3*i] = shape.at<float>(i,0);
		face_->mesh_.vertices_ [3*i+1] = shape.at<float>(i,1);
		face_->mesh_.vertices_ [3*i+2] = shape.at<float>(i,2);
	}
}
void FImRenderer::copyFaces(cv::Mat faces){
	for (int i=0;i<face_->mesh_.nFaces_;i++){
		face_->mesh_.faces_ [3*i] = faces.at<int>(i,0);
		face_->mesh_.faces_ [3*i+1] = faces.at<int>(i,1);
		face_->mesh_.faces_ [3*i+2] = faces.at<int>(i,2);
	}
}

// Load 3D face model into OpenGL
void FImRenderer::loadModel(){
	if (face_->mesh_.texcoords_)
	{
		face_->mesh_.tex_.loadImageGL();
	}
	face_->mesh_.loadGeometryGL(true);
}


// Render the face w/ OpenGL.
//    Inputs:
//        r      : rotation angles (3)
//        t      : translation vector (3)
//        f      : focal length
//    Output: 
//        color  : rendered RGB image
//        depth  : depth image from OpenGL depth buffer
void FImRenderer::render( float *r, float *t, float f, cv::Mat &color, cv::Mat &depth ){
	double mn, mx;
	for (int i=0;i<5;i++){
		mapRendering(r,t,f,&color,&depth);unmapRendering();
		cv::minMaxLoc(depth, &mn, &mx);
		if (mx != mn) break;
	}
}

// Some supporting methods
void FImRenderer::mapRendering( float *rot, float *t, float f, cv::Mat *img, cv::Mat *mask )
{
	glFlush();
	//setup camera intrinsics
	CvMat *K = cvCreateMat( 3, 3, CV_64FC1 );
	cvSetIdentity( K );
	cvmSet( K, 0, 0, f  );
	cvmSet( K, 1, 1, f  );
	cvmSet( K, 0, 2, img_.cols/2 );
	cvmSet( K, 1, 2, img_.rows/2 );
	render_->loadIntrinGL( K, zNear, zFar, img_.cols, img_.rows );
	cvReleaseMat( &K );

	CvMat *H = cvCreateMat( 4, 4, CV_64FC1 );
	cvSetIdentity( H );
	render_->loadExtrinGL( H );
	cvReleaseMat( &H );

	//update extrinsics and render
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();	

	//Translations
	glTranslatef( t[0], t[1], t[2] );

	//Axis angle
	//rot = { tRx, tRy, tRz } , angle = ||rot|| 
	float angle = sqrt( rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2] );
	glRotatef( angle/3.14159f*180.f, rot[0]/angle, rot[1]/angle, rot[2]/angle );

	render_->mapRendering( face_->mesh_ );
	glPopMatrix();	
	glFlush();
	//glutSwapBuffers ();
	
	render_->readFB( *img  );
	render_->readDB( *mask );
}
void
FImRenderer::unmapRendering()
{
	render_->unmapRendering();
}

// Compute vertex normals
void FImRenderer::computeNormals(){
	face_->estimateNormals();
}
