//===========================================================================//
//                                                                           //
// @file            main.cpp                                                 //
// @author          Chirantan Ekbote (ekbote@seas.harvard.edu)               //
// @date            2012/11/05                                               //
// @version         0.2                                                      //
// @brief           Compress any given image into the PVR texture format     //
//                                                                           //
//===========================================================================//
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <FreeImage.h>

#include "../inc/Compressor.h"


/* Miscellaneous Functions */
void usage_error() {
	std::cerr << "Usage: pvrtex <input> <output>" << std::endl;
}

void exit_message(int message) {
	std::cout << std::endl << "Press ENTER to exit." << std::endl;
	getchar();
	exit(message);
}

///////////////////////////////////////////////////////////////////////////////
//	FreeImage Error Handler													 //
//																			 //
//	fif				Format / Plugin responsible for the error				 //
//	message			Error message											 //
///////////////////////////////////////////////////////////////////////////////
void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message) {
	std::cerr << std::endl << "*** ";
	if(fif != FIF_UNKNOWN) {
		std::cerr << FreeImage_GetFormatFromFIF(fif) <<	"Format" << std::endl;
	}
	std::cerr << message;
	std::cerr << " ***" << std::endl;
  
	std::cerr << "Press ENTER to exit." << std::endl;
	getchar();
	exit(EXIT_FAILURE);
}


//////////////////////////////////////////////////////////////////////////////
//	Generic Image Loader using FreeImage.  Returns the loaded image if      //
//  successful, otherwise returns NULL.                                     //
//														                    //
//	lpszPathName	Pointer to the full file name		                    //
//	flag			Optional load flag constant			                    //
//////////////////////////////////////////////////////////////////////////////
FIBITMAP* GenericLoader(const char* lpszPathName, int flag) {
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	// check the file signature and deduce its format
	// (the second argument is currently not used by FreeImage)
	fif = FreeImage_GetFileType(lpszPathName, 0);
	if(fif == FIF_UNKNOWN) {
		// no signature ?
		// try to guess the file format from the file extension
		fif = FreeImage_GetFIFFromFilename(lpszPathName);
	}
	// check that the plugin has reading capabilities ...
	if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif)) {
		// ok, let's load the file
		FIBITMAP *dib = FreeImage_Load(fif, lpszPathName, flag);
		// unless a bad file format, we are done !
		return dib;
	}
	return NULL;
}

int main(int argc, char **argv)
{
  if (argc != 3) {
		usage_error();
		exit_message(EXIT_FAILURE);
	}
  
	std::cout << "PVR Texture Compression Tool" << std::endl << std::endl;
	std::cout << FreeImage_GetCopyrightMessage() << std::endl;
	std::cout << "FreeImage v" << FreeImage_GetVersion() << std::endl;
  
  // set the FreeImage error handler
	FreeImage_SetOutputMessage(FreeImageErrorHandler);
  
	// load the image into a bitmap
	FIBITMAP *dib = GenericLoader(argv[1], 0);
	if (dib == NULL) {
		std::cerr << "Error: Invalid input image type." << std::endl;
		exit_message(EXIT_FAILURE);
	}
  
  // convert the bitmap into a 32-bit raw buffer (top-left pixel first)
  FIBITMAP *src = FreeImage_ConvertTo32Bits(dib);
	FreeImage_Unload(dib);
  
  // Allocate a raw buffer
	int width = FreeImage_GetWidth(src);
	int height = FreeImage_GetHeight(src);
	int scan_width = FreeImage_GetPitch(src);
	BYTE *bits = (BYTE*)malloc(height * scan_width);
  
  // convert the bitmap to raw bits (top-left pixel first)
	FreeImage_ConvertToRawBits(bits, src, scan_width, 32, FI_RGBA_RED_MASK,
                             FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true);
	FreeImage_Unload(src);
  
  // Compress the image
  BYTE *result = (BYTE*)malloc(height * scan_width);
  pvrtex::Compressor cmp(width,
                         height,
                         pvrtex::Compressor::A8R8G8B8,
                         reinterpret_cast<unsigned int*>(bits));
  cmp.Compress(reinterpret_cast<unsigned int*>(result),
               pvrtex::Compressor::PVRTC4);
  // Use FreeImage to write out the uncompressed 32-bit ARGB result
  // as a PNG file
	FIBITMAP *out = FreeImage_ConvertFromRawBits(result, width, height,
                                               scan_width, 32,FI_RGBA_RED_MASK,
                                               FI_RGBA_GREEN_MASK,
                                               FI_RGBA_BLUE_MASK, true);
	FreeImage_Save(FIF_PNG, out, argv[2]);
	FreeImage_Unload(out);
  
  return EXIT_SUCCESS;
}
