/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#ifndef CVGL_TEXTURE_H
#define CVGL_TEXTURE_H

//opencv data structure is used to store images
//this can be changed if desired
#include <cxcore.h>
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES 1
#endif
//GLEW needs to be first
#include <GL/glew.h>//
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>

class Texture {

	//opengl texture id
	unsigned int id;

public:
	//image data
	cv::Mat	img_;

	
	Texture() {
		id = 0;
	}
	
	unsigned int getTextureID() {
		return id;
	}
	
	bool isLoaded() {
		return id > 0;
	}
	
	//load the image to opengl (call once)
	void loadImageGL(bool force = false) {
		if(img_.data && (force || id == 0)) {
			glGenTextures(1, &id);
			glBindTexture(GL_TEXTURE_2D, id);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
			if(img_.channels() == 1) {
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
					        img_.cols, img_.rows, 0,
				            GL_LUMINANCE, GL_UNSIGNED_BYTE, img_.data );
			}
			else {
				glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
				             img_.cols, img_.rows, 0,
				             GL_BGR, GL_UNSIGNED_BYTE, img_.data );
			}
		}
	}
	
	//update the image pointer
	//does not load the image into opengl
	void setImage(IplImage *_img) {
		if(id > 0) {
			glDeleteTextures(1, &id);
			id = 0;
		}
		img_ = cv::cvarrToMat(_img);
	}
	
	~Texture() {
		if(id > 0)
			glDeleteTextures(1, &id);
	}
};

#endif
