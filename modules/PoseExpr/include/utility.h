/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#pragma once
#include "cv.h"
#include "highgui.h"
#include <string>
#include <fstream>
#include <Eigen/Dense>

int splittext(char* str, char** pos);

// Save PLY file, given 3D shape (Vx3), texture (Vx3), and triangle connectivity (Fx3)
void write_plyFloat(char* outname, cv::Mat mat_shape, cv::Mat mat_tex, cv::Mat faces);
void write_plyFloat(char* outname, cv::Mat mat_shape);

