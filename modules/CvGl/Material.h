/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#ifndef CVGL_MATERIAL_H
#define CVGL_MATERIAL_H

class Material {

public:
	float ambient[4];
	float diffuse[4];
	float specular[4];
	float emission[4];
	float shininess;
	
	Material() {				
		ambient[0] = 0.f; 
		ambient[1] = 0.f; 
		ambient[2] = 0.f; 
		ambient[3] = 1.0f;
		diffuse[0] = 0.f;
		diffuse[1] = 0.f;
		diffuse[2] = 0.f;
		diffuse[3] = 1.0f;
		
		specular[0] = 0.0f;
		specular[1] = 0.0f;
		specular[2] = 0.0f;
		specular[3] = 1.0f;

		
		emission[0] = 0.0f;
		emission[1] = 0.0f;
		emission[2] = 0.0f;
		emission[3] = 1.0f;
		
		shininess = 0.0f;
	}
};

#endif
