Project( CUDAPVR )

cmake_minimum_required(VERSION 2.8)

SET( BIN_DIR ${CMAKE_SOURCE_DIR}/bin64/ )
SET( LIB_DIR ${CMAKE_SOURCE_DIR}/lib64/ )
SET( INC_DIR ${CMAKE_SOURCE_DIR}/include )
SET( SRC_DIR ${CMAKE_SOURCE_DIR}/src )

#------------------------------------------------------------
# Global project configuration
#------------------------------------------------------------
# whatever you wish..

#------------------------------------------------------------
# Find CUDA
#------------------------------------------------------------
FIND_PACKAGE(CUDA)

#------------------------------------------------------------
# CUTIL directory
#------------------------------------------------------------
#
# Try to find the cutil64 library and include path.
# Once done this will define
#
# CUTIL_FOUND
# CUTIL_INCLUDE_PATH
# CUTIL_LIBRARY
# 

IF (WIN32)
	FIND_PATH( CUTIL_INCLUDE_PATH cutil.h
		${PROJECT_SOURCE_DIR}/include/
		DOC "The directory where cutil.h resides")
	FIND_LIBRARY( CUTIL_LIBRARY
		NAME cutil64
		PATHS
		${PROJECT_SOURCE_DIR}/lib64
		DOC "The cutil64 library")
ELSE (WIN32)
	FIND_PATH( CUTIL_INCLUDE_PATH cutil.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where cutil.h resides")
	FIND_LIBRARY( CUTIL_LIBRARY
		NAME cutil64
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
		DOC "The cutil64 library")
ENDIF (WIN32)

SET(CUTIL_LIBRARIES ${CUTIL_LIBRARY})

IF (CUTIL_INCLUDE_PATH AND CUTIL_LIBRARY)
	SET( CUTIL_FOUND TRUE CACHE BOOL "Set to TRUE if CUTIL is found, FALSE otherwise")
ELSE (CUTIL_INCLUDE_PATH AND CUTIL_LIBRARY)
	SET( CUTIL_FOUND FALSE CACHE BOOL "Set to TRUE if CUTIL is found, FALSE otherwise")
ENDIF (CUTIL_INCLUDE_PATH AND CUTIL_LIBRARY)

MARK_AS_ADVANCED(
	CUTIL_FOUND 
	CUTIL_LIBRARY
	CUTIL_LIBRARIES
	CUTIL_INCLUDE_PATH)

#------------------------------------------------------------
# FreeImage directory
#------------------------------------------------------------
#
# Try to find the FreeImage library and include path.
# Once done this will define
#
# FREEIMAGE_FOUND
# FREEIMAGE_INCLUDE_PATH
# FREEIMAGE_LIBRARY
# 

IF (WIN32)
	FIND_PATH( FREEIMAGE_INCLUDE_PATH FreeImage.h
		${PROJECT_SOURCE_DIR}/include/
		DOC "The directory where FreeImage.h resides")
	FIND_LIBRARY( FREEIMAGE_LIBRARY
		NAMES FreeImage freeimage
		PATHS
		${PROJECT_SOURCE_DIR}/lib64
		DOC "The FreeImage library")
ELSE (WIN32)
	FIND_PATH( FREEIMAGE_INCLUDE_PATH FreeImage.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where FreeImage.h resides")
	FIND_LIBRARY( FREEIMAGE_LIBRARY
		NAMES FreeImage freeimage
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
		DOC "The FreeImage library")
ENDIF (WIN32)

SET(FREEIMAGE_LIBRARIES ${FREEIMAGE_LIBRARY})

IF (FREEIMAGE_INCLUDE_PATH AND FREEIMAGE_LIBRARY)
	SET( FREEIMAGE_FOUND TRUE CACHE BOOL "Set to TRUE if FREEIMAGE is found, FALSE otherwise")
ELSE (FREEIMAGE_INCLUDE_PATH AND FREEIMAGE_LIBRARY)
	SET( FREEIMAGE_FOUND FALSE CACHE BOOL "Set to TRUE if FREEIMAGE is found, FALSE otherwise")
ENDIF (FREEIMAGE_INCLUDE_PATH AND FREEIMAGE_LIBRARY)

MARK_AS_ADVANCED(
	FREEIMAGE_FOUND 
	FREEIMAGE_LIBRARY
	FREEIMAGE_LIBRARIES
	FREEIMAGE_INCLUDE_PATH)


#------------------------------------------------------------
# PVRTexLib directory
#------------------------------------------------------------
#
# Try to find the PVRTexLib library and include path.
# Once done this will define
#
# PVRTEXLIB_FOUND
# PVRTEXLIB_INCLUDE_PATH
# PVRTEXLIB_LIBRARY
# 

IF (WIN32)
	FIND_PATH( PVRTEXLIB_INCLUDE_PATH PVRTexLib.h
		${PROJECT_SOURCE_DIR}/include/
		DOC "The directory where PVRTexLib.h resides")
	FIND_LIBRARY( PVRTEXLIB_LIBRARY
		NAMES PVRTexLib pvrtexlib
		PATHS
		${PROJECT_SOURCE_DIR}/lib64
		DOC "The PVR Texture library")
ELSE (WIN32)
	FIND_PATH( PVRTEXLIB_INCLUDE_PATH PVRTexLib.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where PVRTexLib.h resides")
	FIND_LIBRARY( PVRTEXLIB_LIBRARY
		NAMES PVRTexLib pvrtexlib
		PATHS
		/usr/lib64
		/usr/lib
		/usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
		DOC "The PVR Texture library")
ENDIF (WIN32)

SET(PVRTEXLIB_LIBRARIES ${PVRTEXLIB_LIBRARY})

IF (PVRTEXLIB_INCLUDE_PATH AND PVRTEXLIB_LIBRARY)
	SET( PVRTEXLIB_FOUND TRUE CACHE BOOL "Set to TRUE if PVRTEXLIB is found, FALSE otherwise")
ELSE (PVRTEXLIB_INCLUDE_PATH AND PVRTEXLIB_LIBRARY)
	SET( PVRTEXLIB_FOUND FALSE CACHE BOOL "Set to TRUE if PVRTEXLIB is found, FALSE otherwise")
ENDIF (PVRTEXLIB_INCLUDE_PATH AND PVRTEXLIB_LIBRARY)

MARK_AS_ADVANCED(
	PVRTEXLIB_FOUND 
	PVRTEXLIB_LIBRARY
	PVRTEXLIB_LIBRARIES
	PVRTEXLIB_INCLUDE_PATH)

#------------------------------------------------------------
# Set up compilation
#------------------------------------------------------------
FILE(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)
FILE(GLOB HDR ${PROJECT_SOURCE_DIR}/include/*.h)
FILE(GLOB CUDA_SRC ${PROJECT_SOURCE_DIR}/src/*.cu)

# depends on OS

IF (UNIX)

LINK_DIRECTORIES( ${LIB_DIR} ${PVRTEXLIB_LIBRARY} ${FREEIMAGE_LIBRARY} ${CUTIL_LIBRARY})
INCLUDE_DIRECTORIES( ${INC_DIR} ${CUTIL_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS} ${PVRTEXLIB_INCLUDE_PATH} ${FREEIMAGE_INCLUDE_PATH} )
CUDA_INCLUDE_DIRECTORIES( ${CUTIL_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS} )
CUDA_ADD_EXECUTABLE( cudapvr ${SRC} ${CUDA_SRC} ${HDR} ) 
TARGET_LINK_LIBRARIES( cudapvr ${FREEIMAGE_LIBRARIES} ${PVRTEXLIB_LIBRARIES} ${CUTIL_LIBRARIES})
CUDA_ADD_LIBRARY( libcudapvr ${CUDA_SRC} ) 
TARGET_LINK_LIBRARIES( libcudapvr ${CUTIL_LIBRARIES})



ELSE (UNIX)

  MESSAGE(STATUS "NOW RUNNING WIN32 MODE")
  
  # Copy DLLs to binary directory
	FILE(GLOB_RECURSE DLLS ${PROJECT_SOURCE_DIR}/bin64/*.dll)
	SOURCE_GROUP("DLL Files" FILES ${DLLS})
	FILE(COPY ${DLLS} DESTINATION ${CMAKE_BINARY_DIR}/)
	MESSAGE( "DLL files are copied " ${DLLS} )

  #SET(CMAKE_CONFIGURATION_TYPES "Debug;Release;EmuDebug;EmuRelease" CACHE STRING "must be force" FORCE)

  #IF(CMAKE_CONFIGURATION_TYPES)
 
  #ENDIF(CMAKE_CONFIGURATION_TYPES)

  # Emulation or Device	
  #SET(CUDA_BUILD_TYPE Device)#Emulation)  
  #SET(CUDA_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../cuda-win32-x86-0.2a.1173)

  # for Visual C++ 2005
  SET( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIBS:LIBCMT" )
  SET( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIBS:MSVCRT" )	
  SET( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIBS:MSVCPRTD" )	

	
	LINK_DIRECTORIES( ${LIB_DIR} ${CUTIL_LIBRARY} )
	INCLUDE_DIRECTORIES( ${INC_DIR} ${CUTIL_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS} ${PVRTEXLIB_INCLUDE_PATH} ${FREEIMAGE_INCLUDE_PATH} )
	CUDA_INCLUDE_DIRECTORIES( ${CUTIL_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS} )

	CUDA_ADD_EXECUTABLE( cudapvr ${SRC} ${CUDA_SRC} ${HDR} ) 
						 
	TARGET_LINK_LIBRARIES( cudapvr cutil64.lib PVRTexLib.lib FreeImage.lib )
	CUDA_ADD_LIBRARY( libcudapvr ${CUDA_SRC} ) 
	TARGET_LINK_LIBRARIES( libcudapvr cutil64.lib )
ENDIF (UNIX)

CUDA_BUILD_CLEAN_TARGET()