/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "utility.h"

int splittext(char* str, char** pos){
	int count = 0;
	char * pch;
	pch = strtok (str," ,\n");
	while (pch != NULL)
	{
		pos[count] =  pch;
		count++;
		pch = strtok (NULL, " ,\n");
	}
	return count;
}

// Save PLY file, given 3D shape (Vx3), texture (Vx3), and triangle connectivity (Fx3)
void write_plyFloat(char* outname, cv::Mat mat_Depth, cv::Mat mat_Color, cv::Mat mat_Faces){
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << mat_Depth.rows << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "property uchar red\n";
	ply2 << "property uchar green\n";
	ply2 << "property uchar blue\n";
	ply2 << "element face " << mat_Faces.rows << std::endl;
	ply2 << "property list uchar int vertex_indices\n";
	ply2 << "end_header\n";
	float colors[3];
	for( int i = 0; i < mat_Depth.rows ; i++ )
	{
		for (int j=0;j<3;j++) {
			colors[j] = mat_Color.at<float>(i,j);
			colors[j] = (colors[j]<0)?0:colors[j];
			colors[j] = (colors[j]>255)?255:colors[j];
		}
		ply2 << mat_Depth.at<float>(i,0) << " " << mat_Depth.at<float>(i,1) << " " << mat_Depth.at<float>(i,2) << " "
			<<  (int)colors[0] << " " << (int)colors[1] << " " << (int)colors[2] << std::endl;
	}
	for( int i = 0; i < mat_Faces.rows ; i++ )
	{
		ply2 << "3 " << mat_Faces.at<int>(i,0) << " " << mat_Faces.at<int>(i,1) << " " << mat_Faces.at<int>(i,2) << " " <<  std::endl;
	}
	ply2.close();
}

// Save PLY file, given 3D shape (Vx3), texture (Vx3), and triangle connectivity (Fx3)
void write_plyFloat(char* outname, cv::Mat mat_Depth){
	std::ofstream ply2( outname );
	ply2 << "ply\n";
	ply2 << "format ascii 1.0\n";
	ply2 << "element vertex " << mat_Depth.rows << std::endl;
	ply2 << "property float x\n";
	ply2 << "property float y\n";
	ply2 << "property float z\n";
	ply2 << "end_header\n";
	float colors[3];
	for( int i = 0; i < mat_Depth.rows ; i++ )
	{
		ply2 << mat_Depth.at<float>(i,0) << " " << mat_Depth.at<float>(i,1) << " " << mat_Depth.at<float>(i,2) << std::endl;
	}
	ply2.close();
}
