# Local libraries
include_directories(../CvGl)
include_directories(include)
include_directories(${DLIB_INCLUDE_DIR})
add_definitions(-std=c++11)
add_definitions(-DDLIB_JPEG_SUPPORT)
add_definitions(-DDLIB_PNG_SUPPORT)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_BUILD_TYPE Release)
SET(SOURCE
    src/utility.cpp
    src/epnp.cpp
    src/BaselFace.cpp
    src/BaselFaceEstimator.cpp
    src/FTModel.cpp
    src/FImRenderer.cpp
    src/RenderModel.cpp
    src/FaceServices2.cpp
    src/DlibWrapper.cpp
)

SET(HEADERS
    include/utility.h	
    include/epnp.h
    include/BaselFace.h
    include/BaselFaceEstimator.h
    include/FTModel.h
    include/FImRenderer.h
    include/RenderModel.h
    include/FaceServices2.h
    include/DlibWrapper.h
)
SET(CMAKE_CXX_FLAGS "-w -fopenmp")
add_library(PoseExprLib STATIC ${SOURCE} ${HEADERS})

if(WIN32)
	target_link_libraries(PoseExprLib PUBLIC ${OpenCVLibraries})
endif(WIN32)
if(UNIX)
	target_link_libraries(PoseExprLib PUBLIC ${OpenCV_LIBS} ${Boost_LIBRARIES})
endif(UNIX)
target_link_libraries(PoseExprLib PUBLIC CvGl)
target_link_libraries(PoseExprLib PUBLIC /usr/lib/libblas.so)
target_link_libraries(PoseExprLib PUBLIC ${DLIB_LIB_DIR}/libdlib.so)
target_link_libraries(PoseExprLib PUBLIC -fopenmp)
target_link_libraries(PoseExprLib PUBLIC libhdf5.so libhdf5_cpp.so)
install (TARGETS PoseExprLib DESTINATION .)
