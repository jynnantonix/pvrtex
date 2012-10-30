//////////////////////////////////////////////////////////////////////////////////
//  PVR Texture Compressor														//
//  Author: Chirantan Ekbote													//
//	Mailto: ekbote.1@osu.edu													//
//																				//
//  Harvard School of Engineering and Applied Sciences							//
//  The Ohio State University													//
//																				//
//	main.cpp - contains code for launching the compressor						//
//																				//
//  This is a GPU implementation of a compressor for the PVR format as			//
//  described by Simon Fenney in his paper Texture Compression using			//
//  Low-Frequency Signal Modulation.											//
//																				//
//////////////////////////////////////////////////////////////////////////////////

#include <iostream>   
#include <math.h>
#include "timing.h"
#include "FreeImage.h"
#include "PVRTexLib.h"
using namespace pvrtexlib;

#define USE_GPU
#define TWO_BPP
#define GET_RMS_ERROR
//#define COMPARE_ERROR
#define GET_SNR
#define COMPRESSED_RESULT
#define SQR(x)				((x) * (x))

extern "C" int cuda_pvr_compress(int, int, int, unsigned char*, unsigned char*);

void usage_error() {
	std::cerr << "Usage: PVRCompressor <input> <output>" << std::endl;
}

void exit_message(int message) {
	std::cout << std::endl << "Press ENTER to exit." << std::endl;
	getchar();
	exit(message);
}

//////////////////////////////////////////////////////////////////////////////////
//	FreeImage Error Handler														//
//																				//
//	fif				Fromat / Plugin responsible for the error					//
//	message			Error message												//
//////////////////////////////////////////////////////////////////////////////////
void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message) {
	std::cerr << std::endl << "*** ";
	if(fif != FIF_UNKNOWN) {
		std::cerr << FreeImage_GetFormatFromFIF(fif) <<
			"Format" << std::endl;
	}
	std::cerr << message;
	std::cerr << " ***" << std::endl;

	std::cerr << "Press ENTER to exit." << std::endl;
	getchar();
	exit(EXIT_FAILURE);
}

//////////////////////////////////////////////////////////
//	Generic Image Loader using FreeImage.Returns the	//
//	loaded image if successful, returns NULL otherwise.	//		
//														//
//	lpszPathName	Pointer to the full file name		//
//	flag			Optional load flag constant			//
//////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////
//	Texture Compressor. Reads in the given image using	//
//	FreeImage and prepares it for compression. Then		//
//	uses PVRTexLib or CUDA to compress it.				//
//														//
//	argc		Should be 3								//
//	argv[1]		The path to the input image				//
//	argv[2]		The path to the output image (pvr)		//
//////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	if (argc != 3) {
		usage_error();
		exit_message(EXIT_FAILURE);
	}

	std::cout << "CUDA PVR Texture Compression Tool" << std::endl << std::endl;
	std::cout << FreeImage_GetCopyrightMessage() << std::endl;
	std::cout << "FreeImage v" << FreeImage_GetVersion() << std::endl << std::endl;

	// set the FreeImage error handler
	FreeImage_SetOutputMessage(FreeImageErrorHandler);

	// load the image into a bitmap
	FIBITMAP *dib = GenericLoader(argv[1], 0);
	if (dib == NULL) {
		std::cerr << "Error: Invalid input image type." << std::endl;
		exit_message(EXIT_FAILURE);
	}
#ifdef COMPARE_ERROR
	FIBITMAP *dib2 = GenericLoader(argv[2], 0);
	if (dib2 == NULL) {
		std::cerr << "Error: Invalid input image type." << std::endl;
		exit_message(EXIT_FAILURE);
	}
#endif

	// convert the bitmap into a 32-bit raw buffer (top-left pixel first)
	// ------------------------------------------------------------------
	FIBITMAP *src = FreeImage_ConvertTo32Bits(dib);
	FreeImage_Unload(dib);
#ifdef COMPARE_ERROR
	FIBITMAP *src2 = FreeImage_ConvertTo32Bits(dib2);
	FreeImage_Unload(dib2);
#endif

	// Allocate a raw buffer
	int width = FreeImage_GetWidth(src);
	int height = FreeImage_GetHeight(src);
	int scan_width = FreeImage_GetPitch(src);
	BYTE *bits = (BYTE*)malloc(height * scan_width);
#ifdef COMPARE_ERROR
	if (width != FreeImage_GetWidth(src2) ||
		height != FreeImage_GetHeight(src2) ||
		scan_width != FreeImage_GetPitch(src2)) {
			printf("FATAL Error: Image dimensions do not match.\n");
			printf("%d\n%d\n%d", FreeImage_GetWidth(src2), 
				FreeImage_GetHeight(src2), FreeImage_GetPitch(src2));
			exit_message(EXIT_FAILURE);
	}
	BYTE *bits2 = (BYTE*)malloc(height * scan_width);
#endif
#ifdef	COMPRESSED_RESULT
#ifdef TWO_BPP
    BYTE *result = (BYTE*)malloc(2*(width>>3)*(height>>2)*sizeof(unsigned int));
#else
	BYTE *result = (BYTE*)malloc(2*(width>>2)*(height>>2)*sizeof(unsigned int));
#endif // TWO_BPP
#else	// We are getting a full resolution 32-bit ARGB result
	BYTE *result = (BYTE*)malloc(height * scan_width);
#endif // COMPRESSED_RESULT

	// convert the bitmap to raw bits (top-left pixel first)
	FreeImage_ConvertToRawBits(bits, src, scan_width, 32,
		FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
	FreeImage_Unload(src);
#ifdef COMPARE_ERROR
	FreeImage_ConvertToRawBits(bits2, src2, scan_width, 32,
		FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);
	FreeImage_Unload(src2);
#endif

	// create the timer 
	CTimer *timer = new CTimer();
	double time;
	// do the compression, either with CUDA or PVRTexLib
	// ------------------------------------------------------------------
#ifdef USE_GPU
    timer->Reset();					// start the timer
    if (cuda_pvr_compress(width, height, scan_width, bits, result) == 1) {
        exit_message(EXIT_FAILURE);
    }
    time = timer->Query();
#ifdef COMPRESSED_RESULT
	PVRTRY {
		// create the utilities instance
		PVRTextureUtilities sPVRU = PVRTextureUtilities();

		// create the original texture from the supplied image
		CPVRTexture sCompressedTexture(
			width,					// u32Width,
			height,					// u32Height,
			0,						// u32MipMapCount
			1,						// u32NumSurfaces,
			false,					// bBorder,
			true,					// bTwiddled,
			false,					// bCubeMap,
			false,					// bVolume,
			false,					// bFalseMips,
			false,					// bHasAlpha
			false,					// bVerticallyFlipped
#ifdef TWO_BPP
            OGL_PVRTC2,
#else
			OGL_PVRTC4,				// ePixelType,
#endif  // TWO_BPP
			0.0f,					// fNormalMap,
			result					// pPixelData
			);

		// write out the compressed file
		sCompressedTexture.writeToFile(argv[2]);
	} PVRCATCH(PVRException) {
		std::cerr << "Exception occurred during compression: " << 
			PVRException.what() << std::endl;
		std::cerr << "Press ENTER to exit." << std::endl;
		getchar();
		exit(EXIT_FAILURE);
	}
#else // Use FreeImage to write out the uncompressed 32-bit ARGB result
	  // as a PNG file
	FIBITMAP *out = FreeImage_ConvertFromRawBits(result, width, height,
		scan_width, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, true);
	FreeImage_Save(FIF_PNG, out, argv[2]);
	FreeImage_Unload(out);
#endif // COMPRESSED_RESULT
#else
	timer->Query();

	// do the compression with PVRTexLib
	PVRTRY {
		// create the utilities instance
		PVRTextureUtilities sPVRU = PVRTextureUtilities();

		// create the original texture from the supplied image
		CPVRTexture sOriginalTexture(
			width,					// u32Width,
			height,					// u32Height,
			0,						// u32MipMapCount
			1,						// u32NumSurfaces,
			false,					// bBorder,
			false,					// bTwiddled,
			false,					// bCubeMap,
			false,					// bVolume,
			false,					// bFalseMips,
			true,					// bHasAlpha
			false,					// bVerticallyFlipped
			eInt8StandardPixelType,	// ePixelType,
			0.0f,					// fNormalMap,
			bits					// pPixelData
			);

		// declare the compressed texture and set the correct pixel type
		CPVRTexture sCompressedTexture(sOriginalTexture.getHeader());
		sCompressedTexture.setPixelType(OGL_PVRTC4);

		sPVRU.CompressPVR(sOriginalTexture, sCompressedTexture);

		time = timer->Pause();
#ifdef GET_RMS_ERROR
		int i, j, k, index;
		unsigned char o, c;
		float error = 0.0f, mean_error = 0.0f, total_error = 0.0f;

		// decompress the compressed data and get the pointer to the 
		// uncompressed pixel data
		CPVRTexture sDecompressedTexture(sOriginalTexture.getHeader());
		sPVRU.DecompressPVR(sCompressedTexture, sDecompressedTexture);
#ifdef COMPARE_ERROR
		unsigned char *decompressed = bits2;
#else
		unsigned char *decompressed = sDecompressedTexture.getData().getData();
#endif // COMPARE_ERROR
		for (i = 0; i < height; i++) {
			for (j = 0; j < width; j++) {
				index = 4*i*width + 4*j;
				for (k = 0; k < 4; k++) {
					o = bits[index + k];
					c = decompressed[index + k];
					total_error += SQR((unsigned int)o - (unsigned int)c);
				}
			}
		}

		mean_error = total_error / (width * height * 3);
		error = sqrtf(mean_error);
		printf("RMS Error is %.3f.\n", error);
#ifdef GET_SNR
		printf("Peak signal-to-noise ratio is %0.3f.\n", 10 * log10(SQR(255) / mean_error));
#endif // GET_SNR
#endif	// GET_RMS_ERROR

#ifdef COMPRESSED_RESULT
		// write out the compressed file
		sCompressedTexture.writeToFile(argv[2]);
#else	// Use FreeImage to write out the uncompressed 32-bit ARGB result
		// as a PNG file
		FIBITMAP *out = FreeImage_ConvertFromRawBits(decompressed, width, height,
			scan_width, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, true);
		FreeImage_Save(FIF_PNG, out, argv[2]);
		FreeImage_Unload(out);
#endif // COMPRESSED_RESULT
	} PVRCATCH(PVRException) {
		std::cerr << "Exception occurred during compression: " << 
			PVRException.what() << std::endl;
		std::cerr << "Press ENTER to exit." << std::endl;
		getchar();
		exit(EXIT_FAILURE);
	}
#endif // _USE_GPU_

	std::cout << "Compression took " << time << " seconds" << std::endl;
	exit_message(EXIT_SUCCESS);
}
