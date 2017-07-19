cmake_minimum_required(VERSION 2.8)
include_directories(${OpenCV_INCLUDE_DIRS})

SET(SOURCE
    FBRender.cpp
)

SET(HEADERS
    cvgl.h
    FBRender.h	
    Material.h
    SimpleMesh.h
    Texture.h	
)

add_library( CvGl STATIC ${SOURCE} ${HEADERS})
target_link_libraries(CvGl PUBLIC -lGLU 
	-lGLEW -lglut -lGL -lm)

install (TARGETS CvGl DESTINATION .)
