#============================================================================#
#                                                                            #
#   @file       CMakeLists.txt                                               #
#   @author     Chirantan Ekbote (ekbote@seas.harvard.edu)                   #
#   @date       2012/11/04                                                   #
#   @version    0.3                                                          #
#   @brief      Project file for cross-platform compatibility                #
#                                                                            #
#============================================================================#

project( pvrtex )

cmake_minimum_required(VERSION 2.8)

# Add the local module directory so cmake can find the right files
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

# Get OpenMP
find_package(OpenMP)
if(${OPENMP_FOUND})
  add_definitions(-DUSE_OPENMP)
  list(APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
endif()

# Get FreeImage
find_package(FreeImage REQUIRED)

# Get Eigen3
find_package(Eigen3 REQUIRED)

# Get project sources
file(GLOB HDR ${PROJECT_SOURCE_DIR}/inc/*.h)
file(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Create the project executable
include_directories( ${FREEIMAGE_INCLUDE_PATH} ${EIGEN3_INCLUDE_DIR} )
add_executable( pvrtex ${HDR} ${SRC} )
target_link_libraries( pvrtex ${FREEIMAGE_LIBRARIES} )

