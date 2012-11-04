#=====================================================================================#
#
#	@file		CMakeLists.txt
#	@author		Chirantan Ekbote (ekbote@seas.harvard.edu)
#   @date       2012/11/04
#   @version    0.1
#	@brief		Project file for cross-platform compatibility
#
#=====================================================================================#

project( pvrtex )

cmake_minimum_required(VERSION 2.8)

# Add the local module directory so cmake can find the right files
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

# Get FreeImage
find_package(FreeImage REQUIRED)

# Get project sources
FILE(GLOB HDR ${PROJECT_SOURCE_DIR}/inc/*.h)
FILE(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Create the project executable
if( ${FREEIMAGE_FOUND} )
  include_directories( ${FREEIMAGE_INCLUDE_PATH} )

  add_executable( pvrtex ${HDR} ${SRC} )

  target_link_libraries( pvrtex ${FREEIMAGE_LIBRARIES} )
endif()

