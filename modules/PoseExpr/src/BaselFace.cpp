#include "BaselFace.h"
#include <fstream>

// HDF5
#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
#include <string>

#ifndef H5_NO_NAMESPACE
#ifndef H5_NO_STD
    using std::cout;
    using std::endl;
#endif  // H5_NO_STD
#endif

#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

    cv::Mat readH5Dataset(const H5::H5File& file, const std::string& datasetName)
    {
        cv::Mat out;

        // Open the specified dataset in the file
        H5::DataSet dataset = file.openDataSet(datasetName);

        // Get dataset info
        H5T_class_t type_class = dataset.getTypeClass();
        H5::DataSpace filespace = dataset.getSpace();
        hsize_t dims[2];    // dataset dimensions
        int rank = filespace.getSimpleExtentDims(dims);

        // Read dataset
        int sizes[2] = { (int)dims[0], (int)dims[1] };
        out.create(rank, sizes, CV_32FC1);
        dataset.read(out.data, H5::PredType::NATIVE_FLOAT, H5::DataSpace(rank, dims), filespace);

        return out;
    }

    void BaselFace::load(const std::string & model_file)
    {
        try
        {
            // Turn off the auto-printing when failure occurs so that we can
            // handle the errors appropriately
            H5::Exception::dontPrint();

            // Open the specified file and the specified dataset in the file
            H5::H5File file(model_file.c_str(), H5F_ACC_RDONLY  );
            cv::Mat faces_tmp = readH5Dataset(file, "/faces");
            cv::Mat lmInd_tmp = readH5Dataset(file, "/innerLandmarkIndex");
            cv::Mat lmInd2_tmp = readH5Dataset(file, "/outerLandmarkIndex");
            shapeMU = readH5Dataset(file, "/shapeMU");
            shapePC = readH5Dataset(file, "/shapePC");
            shapeEV = readH5Dataset(file, "/shapeEV");
            texMU = readH5Dataset(file, "/texMU");
            texPC = readH5Dataset(file, "/texPC");
            texEV = readH5Dataset(file, "/texEV");
            exprMU = readH5Dataset(file, "/expMU");
            exprPC = readH5Dataset(file, "/expPC");
            exprEV = readH5Dataset(file, "/expEV");

            // Convert faces to unsigned int
            float* faces_data = (float*)faces_tmp.data;
            int faces_size = faces_tmp.total();
            faces.create(faces_tmp.size(), CV_16U);
            unsigned short* out_faces_data = (unsigned short*)faces.data;
            for (int i = 0; i < faces_size; ++i)
                *out_faces_data++ = (unsigned short)(*faces_data++);

            // Convert lmInd to int
            float* ind_data = (float*)lmInd_tmp.data;
            int ind_size = lmInd_tmp.total();
            lmInd.create(lmInd_tmp.size(), CV_32S);
            int* out_lmInd_data = (int*)lmInd.data;
            for (int i = 0; i < ind_size; ++i)
                *out_lmInd_data++ = (int)(*ind_data++);

            // Convert lmInd2 to int
            ind_data = (float*)lmInd2_tmp.data;
            lmInd2.create(lmInd2_tmp.size(), CV_32S);
            out_lmInd_data = (int*)lmInd2.data;
            for (int i = 0; i < ind_size; ++i)
                *out_lmInd_data++ = (int)(*ind_data++);
        }
        catch (H5::DataSetIException error)
        {
            throw std::runtime_error(error.getDetailMsg());
        }

    }
