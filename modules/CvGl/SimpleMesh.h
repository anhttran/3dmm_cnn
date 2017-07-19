/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#ifndef CVGL_SIMPLEMESH_H
#define CVGL_SIMPLEMESH_H

#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES 1
#endif
//GLEW needs to be first
#include <GL/glew.h>//
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>//

#include "Material.h"
#include "Texture.h"

class SimpleMesh {
	
public:
	//vbo id for points
	unsigned int ptid;
	
	//vbo id for normals
	unsigned int normid;
	
	//vbo id for texture coords
	unsigned int txid;
	
	//vbo id for indexes
	unsigned int idxid;
	
	//vbo id for colors
	unsigned int colorid;
public:
	//number of vertices
	//int count;			//YANN deprecated

	unsigned nVertices_;		//YANN
	unsigned nFaces_;			//YANN
	
	//array of vertex coordinates (3*nVertices_)
	//float *points;		//YANN deprecated
	float *vertices_;	//YANN
	float *colors_;
	
	//array of vertex normal coordinates (3*nVertices_)
	float *normals;
	
	//array of vertex texture coordinates (2*nVertices_)
	//float *texcoords;		//YANN deprecated
	float *texcoords_;		//YANN
	float *texdepth_;		//YANN
	
	//array of vertex indices (3*nFaces_)
	//unsigned int *indexes;	//YANN deprecated
	unsigned *faces_;		//YANN

	Material mat;			//DEPRECATED
	Texture tex_;			//YANN
	
	SimpleMesh() {
		ptid = 0;
		normals = NULL;
		normid = 0;
		txid = 0;
		idxid = 0;

		vertices_ = NULL;
		nVertices_ = 0;
		faces_ = NULL;
		nFaces_ = 0;
		texcoords_ = 0;
		colors_ = NULL;
		colorid = 0;
	}
		
	//SimpleMesh(vector<cv::Point3f> points) {
	//	ptid = 0;
	//	normals = NULL;
	//	normid = 0;
	//	txid = 0;
	//	idxid = 0;

	//	faces_ = NULL;
	//	nFaces_ = 0;
	//	texcoords_ = 0;
	//	nVertices_ = points.size();
	//	vertices_ = new float[3 * nVertices_ * sizeof(float)];
	//	for (int i=0;i<nVertices_;i++){
	//		vertices_[3*i] = points[i].x;
	//		vertices_[3*i+1] = points[i].y;
	//		vertices_[3*i+2] = points[i].y;
	//	}
	//}
	
	//call once
	void loadGeometryGL(bool forced = false) {
		//printf("before %d %d %d %d %d\n",ptid,normid,txid,idxid,colorid);
		if (forced){
			if(ptid > 0)
				glDeleteBuffers(1, &ptid);
			if(normid > 0)
				glDeleteBuffers(1, &normid);
			if(txid > 0)
				glDeleteBuffers(1, &txid);
			if(idxid > 0)
				glDeleteBuffers(1, &idxid);
			if(colorid > 0)
				glDeleteBuffers(1, &colorid);

			ptid = normid = txid = idxid = colorid = 0;
		}
		if(vertices_ && ptid == 0) {
			glGenBuffers(1, &ptid);
			glBindBuffer(GL_ARRAY_BUFFER, ptid);
			glBufferData(GL_ARRAY_BUFFER, 
			             3 * nVertices_ * sizeof(float), vertices_, 
			             GL_STATIC_DRAW);
		}
		if(normals && normid == 0) {
			glGenBuffers(1, &normid);
			glBindBuffer(GL_ARRAY_BUFFER, normid);
			glBufferData(GL_ARRAY_BUFFER, 
			             3 * nVertices_ * sizeof(float), normals, 
			             GL_STATIC_DRAW);
		}
		if(texcoords_ && txid == 0) {
			glGenBuffers(1, &txid);
			glBindBuffer(GL_ARRAY_BUFFER, txid);
			glBufferData(GL_ARRAY_BUFFER, 
			             2 * nVertices_ * sizeof(float), texcoords_, 
			             GL_STATIC_DRAW);
		}
		if(colors_ && colorid == 0) {
			glGenBuffers(1, &colorid);
			glBindBuffer(GL_ARRAY_BUFFER, colorid);
			glBufferData(GL_ARRAY_BUFFER, 
			             4 * nVertices_ * sizeof(float), colors_, 
			             GL_STATIC_DRAW);
		}
		if(faces_ && idxid == 0) {
			glGenBuffers(1, &idxid);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idxid);
			//here we assume faces_ is for triangles
			//a geometry-type field can be added for flexibility
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
						 3 * nFaces_ * sizeof(unsigned int), faces_, 
			             GL_STATIC_DRAW);
		}
		//printf("after %d %d %d %d %d\n",ptid,normid,txid,idxid,colorid);
	}
	
	~SimpleMesh() {
		if(ptid > 0)
			glDeleteBuffers(1, &ptid);
		if(normid > 0)
			glDeleteBuffers(1, &normid);
		if(txid > 0)
			glDeleteBuffers(1, &txid);
		if(idxid > 0)
			glDeleteBuffers(1, &idxid);
		
		delete[] vertices_;
		delete[] normals;
		delete[] texcoords_;
		delete[] faces_;
	}
};

#endif
