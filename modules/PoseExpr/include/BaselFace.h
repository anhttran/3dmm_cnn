#ifndef BASELFACE_H
#define BASELFACE_H

// Includes
#include "cv.h"
#include "highgui.h"

#include <string>

    class BaselFace
    {

        public:
        cv::Mat faces;
        cv::Mat shapeMU, shapePC, shapeEV;
        cv::Mat texMU, texPC, texEV;
        cv::Mat exprMU, exprPC, exprEV;
        cv::Mat lmInd, lmInd2;

        void load(const std::string& model_file);
    };

#endif // BASELFACE_H
