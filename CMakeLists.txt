cmake_minimum_required(VERSION 3.13)
project(depthComp)

#TODO: make this detect version/minor version
FIND_PACKAGE(Boost COMPONENTS python35 REQUIRED)
FIND_PACKAGE(PythonInterp 3 REQUIRED)
FIND_PACKAGE(PythonLibs 3 REQUIRED)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} submod/pyboostcvconverter/include/)

add_library(depthComp SHARED
    depthComp.cpp
    depthCompPy.cpp
    submod/pyboostcvconverter/src/pyboost_cv4_converter.cpp
    submod/pyboostcvconverter/src/pyboost_cv2_converter.cpp
    submod/pyboostcvconverter/src/pyboost_cv3_converter.cpp)

target_link_libraries (depthComp ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
