//////////////////////////////////////////////////////////////////////////////////
//  PVR Texture Compressor														//
//  Author: Chirantan Ekbote													//
//	Mailto: ekbote.1@osu.edu													//
//																				//
//  Harvard School of Engineering and Applied Sciences							//
//  The Ohio State University													//
//																				//
//	PVRTexCUDA.cu - Contains CUDA compression code								//
//																				//
//  This is a GPU implementation of a compressor for the PVR format as			//
//  described by Simon Fenney in his paper Texture Compression using			//
//  Low-Frequency Signal Modulation.											//
//																				//
//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
//									File Includes	 							//
//////////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>
#include <stdio.h>
#include <limits.h>
//////////////////////////////////////////////////////////////////////////////////
//									File Includes	 							//
//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
//							   Preprocessor Definitions	 						//
//////////////////////////////////////////////////////////////////////////////////
#define TWO_BPP
#define USE_PIXEL_UPDATE
#define USE_SVD
//#define USE_CHOLESKY
//#define USE_JAMA_SVD
//#define GET_RMS_ERROR
//#define GET_SNR
//#define DECOMPRESS_PVR
#define TWO_BY_TWO
#define BLOCK_WIDTH				16
#define BLOCK_HEIGHT			16
#define FILTER_LENGTH			2
#define HALF_FILTER_LENGTH		FILTER_LENGTH * 0.5f
#define RED_SHIFT				16
#define GREEN_SHIFT				8
#define	BLUE_SHIFT				0
#define RSHIFT_16BPP			3
#define LSHIFT_16BPP			5
#define ALPHA_MASK				0xFF000000
#define RED_MASK				0x00FF0000
#define GREEN_MASK				0x0000FF00
#define BLUE_MASK				0x000000FF
#define ONE_SIXTEENTH           0.0625f
#define ONE_EIGHTH				0.125f
#define ONE_FOURTH				0.25f
#define THREE_EIGHTHS			0.375f
#define FIVE_EIGHTHS			0.625f
#define NUM_OPTIMIZATION_PASSES	20
#define EPSILON					10e-10
//////////////////////////////////////////////////////////////////////////////////
//							   Preprocessor Definitions	 						//
//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
//								  Macro Definitions		 						//
//////////////////////////////////////////////////////////////////////////////////
#define SQR(x)					((x) * (x))
#define CLAMP(x, a, b)			(min((max((x), (a))), (b)))
#define MAKE_RED_PIXEL(x)		(((x) & RED_MASK)>>RED_SHIFT)
#define MAKE_GREEN_PIXEL(x)		(((x) & GREEN_MASK)>>GREEN_SHIFT)
#define MAKE_BLUE_PIXEL(x)		(((x) & BLUE_MASK)>>BLUE_SHIFT)
#define MAKE_ARGB(r, g, b)		(ALPHA_MASK | ((r)<<RED_SHIFT) | \
                                ((g)<<GREEN_SHIFT) | ((b)<<BLUE_SHIFT))
//////////////////////////////////////////////////////////////////////////////////
//								  Macro Definitions		 						//
//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
//									   Misc.									//
//////////////////////////////////////////////////////////////////////////////////
// CUDA texture declarations
texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> texRef;
texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> origRef;
texture<int, cudaTextureType2D, cudaReadModeElementType> redCurrentRef;
texture<int, cudaTextureType2D, cudaReadModeElementType> greenCurrentRef;
texture<int, cudaTextureType2D, cudaReadModeElementType> blueCurrentRef;
texture<float, cudaTextureType2D, cudaReadModeElementType> modRef;
texture<float, cudaTextureType1D, cudaReadModeElementType> filterRef;
#ifdef TWO_BY_TWO
texture<float, cudaTextureType1D, cudaReadModeElementType> svdMatRefTL;
texture<float, cudaTextureType1D, cudaReadModeElementType> svdMatRefTR;
texture<float, cudaTextureType1D, cudaReadModeElementType> svdMatRefBL;
texture<float, cudaTextureType1D, cudaReadModeElementType> svdMatRefBR;
#else
texture<float, cudaTextureType1D, cudaReadModeElementType> svdMatRef;
#endif // TWO_BY_TWO

// various wavelet filters. remember to update the FILTER_LENGTH macro if you
// choose to use a different one

// bior wavelet filter
//static const float wavelet_filter[] = {
//	-0.00488281250000000000,
//	0.00976562500000000000,
//	0.03320312500000000000,
//	-0.07617187500000000000,
//	-0.12011718750000000000,
//	0.31640625000000000000,
//	0.68359375000000000000,
//	0.31640625000000000000,
//	-0.12011718750000000000,
//	-0.07617187500000000000,
//	0.03320312500000000000,
//	0.00976562500000000000,
//	-0.00488281250000000000,
//};

// 2nd bior wavelet filter
//static const float wavelet_filter[] = {
//-0.0625,
//0.06250,
//0.50000,
//0.50000,
//0.06250,
//-0.0625
//};

// my wavelet filter
//static const float wavelet_filter[] = {
//-0.2500,
//0.50000,
//0.50000,
//0.50000,
//-0.2500
//};

// daubechies wavelet filter
//static const float wavelet_filter[] = {
//	0.02692517479416041400,
//	0.17241715192471294000,
//	0.42767453217028290000,
//	0.46477285717277800000,
//	0.09418477475112015100,
//	-0.20737588089628295000,
//	-0.06847677451090331000,
//	0.10503417113713563000,
//	0.02172633772990401800,
//	-0.04782363205881859400,
//	0.00017744640673182261,
//	0.01581208292613723100,
//	-0.00333981011324138060,
//	-0.00302748028715121120,
//	0.00130648364017893680,
//	0.00016290733600968354,
//	-0.00017816487954739422,
//	0.00002782275679290904
//};

// simple wavelet filter
static float wavelet_filter[] = {0.5f, 0.5f};

// weight matrix to be combined with modulation data in the optimization step
#ifdef TWO_BPP
#define ROW(n)	1*n, 2*n, 3*n, 4*n, 5*n, 6*n, 7*n, 8*n, 7*n, 6*n, 5*n, 4*n, 3*n, 2*n, 1*n
#else
#define ROW(n)	1*n,	2*n,	3*n,	4*n,	3*n,	2*n,	1*n
#endif // TWO_BPP

#ifdef TWO_BY_TWO

#ifdef TWO_BPP
#define SVD_FACTOR_X            ONE_SIXTEENTH
#define SVD_FACTOR_Y			ONE_EIGHTH
#define SVD_OFFSET_X            8
#define SVD_OFFSET_Y			4
#define SVD_MAT_WIDTH			8
#define SVD_MAT_HEIGHT			253
#define SVD_DIM_X               23
#define SVD_DIM_Y				11
static const float MwTL[] = 
{
	ROW(0.03125f), 0, 0, 0, 0, 0, 0, 0, 0,	// 1/32
	ROW(0.0625f), 0, 0, 0, 0, 0, 0, 0, 0,	// 2/32
	ROW(0.09375f), 0, 0, 0, 0, 0, 0, 0, 0,	// 3/32
	ROW(0.125f), 0, 0, 0, 0, 0, 0, 0, 0,    // 4/32
	ROW(0.09375f), 0, 0, 0, 0, 0, 0, 0, 0,	// 3/32
	ROW(0.0625f), 0, 0, 0, 0, 0, 0, 0, 0,	// 2/32
	ROW(0.03125f), 0, 0, 0, 0, 0, 0, 0, 0,	// 1/32
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0
};
static const float MwTR[] = 
{
    0, 0, 0, 0, 0, 0, 0, 0,	ROW(0.03125f),  // 1/32
	0, 0, 0, 0, 0, 0, 0, 0,	ROW(0.0625f),   // 2/32
	0, 0, 0, 0, 0, 0, 0, 0,	ROW(0.09375f),  // 3/32
	0, 0, 0, 0, 0, 0, 0, 0, ROW(0.125f),    // 4/32
	0, 0, 0, 0, 0, 0, 0, 0,	ROW(0.09375f),  // 3/32
	0, 0, 0, 0, 0, 0, 0, 0,	ROW(0.0625f),   // 2/32
	0, 0, 0, 0, 0, 0, 0, 0,	ROW(0.03125f),  // 1/32
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0
};
static const float MwBL[] = 
{
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0.03125f), 0, 0, 0, 0, 0, 0, 0, 0,	// 1/32
	ROW(0.0625f), 0, 0, 0, 0, 0, 0, 0, 0,	// 2/32
	ROW(0.09375f), 0, 0, 0, 0, 0, 0, 0, 0,	// 3/32
	ROW(0.125f), 0, 0, 0, 0, 0, 0, 0, 0,    // 4/32
	ROW(0.09375f), 0, 0, 0, 0, 0, 0, 0, 0,	// 3/32
	ROW(0.0625f), 0, 0, 0, 0, 0, 0, 0, 0,	// 2/32
	ROW(0.03125f), 0, 0, 0, 0, 0, 0, 0, 0	// 1/32
};
static const float MwBR[] = 
{
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0,	ROW(0.03125f),  // 1/32
	0, 0, 0, 0, 0, 0, 0, 0,	ROW(0.0625f),   // 2/32
	0, 0, 0, 0, 0, 0, 0, 0,	ROW(0.09375f),  // 3/32
	0, 0, 0, 0, 0, 0, 0, 0, ROW(0.125f),    // 4/32
	0, 0, 0, 0, 0, 0, 0, 0,	ROW(0.09375f),  // 3/32
	0, 0, 0, 0, 0, 0, 0, 0,	ROW(0.0625f),   // 2/32
	0, 0, 0, 0, 0, 0, 0, 0,	ROW(0.03125f)   // 1/32
};

#else

#define SVD_FACTOR_X            ONE_EIGHTH
#define SVD_FACTOR_Y			ONE_EIGHTH
#define SVD_OFFSET_X            4
#define SVD_OFFSET_Y			4
#define SVD_MAT_WIDTH			8
#define SVD_MAT_HEIGHT			121
#define SVD_DIM_X               11
#define SVD_DIM_Y				11
static const float MwTL[] = 
{
	ROW(0.0625f), 0, 0, 0, 0,	// 1/16
	ROW(0.125f), 0, 0, 0, 0,	// 2/16
	ROW(0.1875f), 0, 0, 0, 0,	// 3/16
	ROW(0.25f), 0, 0, 0, 0,		// 4/16
	ROW(0.1875f), 0, 0, 0, 0,	// 3/16
	ROW(0.125f), 0, 0, 0, 0,	// 2/16
	ROW(0.0625f), 0, 0, 0, 0,	// 1/16
	ROW(0), 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0
};
static const float MwTR[] = 
{
	0, 0, 0, 0, ROW(0.0625f),	// 1/16
	0, 0, 0, 0, ROW(0.125f),	// 2/16
	0, 0, 0, 0, ROW(0.1875f),	// 3/16
	0, 0, 0, 0, ROW(0.25f),		// 4/16
	0, 0, 0, 0, ROW(0.1875f),	// 3/16
	0, 0, 0, 0, ROW(0.125f),	// 2/16
	0, 0, 0, 0, ROW(0.0625f),	// 1/16
	ROW(0), 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0
};
static const float MwBL[] = 
{
	ROW(0), 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0,
	ROW(0.0625f), 0, 0, 0, 0,	// 1/16
	ROW(0.125f), 0, 0, 0, 0,	// 2/16
	ROW(0.1875f), 0, 0, 0, 0,	// 3/16
	ROW(0.25f), 0, 0, 0, 0,		// 4/16
	ROW(0.1875f), 0, 0, 0, 0,	// 3/16
	ROW(0.125f), 0, 0, 0, 0,	// 2/16
	ROW(0.0625f), 0, 0, 0, 0,	// 1/16
};
static const float MwBR[] = 
{
	ROW(0), 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0,
	ROW(0), 0, 0, 0, 0,
	0, 0, 0, 0, ROW(0.0625f),	// 1/16
	0, 0, 0, 0, ROW(0.125f),	// 2/16
	0, 0, 0, 0, ROW(0.1875f),	// 3/16
	0, 0, 0, 0, ROW(0.25f),		// 4/16
	0, 0, 0, 0, ROW(0.1875f),	// 3/16
	0, 0, 0, 0, ROW(0.125f),	// 2/16
	0, 0, 0, 0, ROW(0.0625f)	// 1/16
};
#endif  // TWO_BPP

#else

#ifdef TWO_BPP
#define SVD_FACTOR_X            ONE_EIGHTH
#define SVD_FACTOR_Y			ONE_FOURTH
#define SVD_OFFSET_X            8
#define SVD_OFFSET_Y			4
#define SVD_MAT_WIDTH			2
#define SVD_MAT_HEIGHT			105
#define SVD_DIM_X               15
#define SVD_DIM_Y				7
static const float Mw[] = 
{
	ROW(0.03125f),	// 1/32
	ROW(0.0625f),	// 2/32
	ROW(0.09375f),	// 3/32
	ROW(0.125f),    // 4/32
	ROW(0.09375f),	// 3/32
	ROW(0.0625f),	// 2/32
	ROW(0.03125f)	// 1/32
};
#else
#define SVD_FACTOR_X			ONE_FOURTH
#define SVD_FACTOR_Y			ONE_FOURTH
#define SVD_OFFSET_X			4
#define SVD_OFFSET_Y			4
#define SVD_MAT_WIDTH			2
#define SVD_MAT_HEIGHT			49
#define SVD_DIM_X				7
#define SVD_DIM_Y				7
static const float Mw[] = 
{
	ROW(0.0625f),	// 1/16
	ROW(0.125f),	// 2/16
	ROW(0.1875f),	// 3/16
	ROW(0.25f),		// 4/16
	ROW(0.1875f),	// 3/16
	ROW(0.125f),	// 2/16
	ROW(0.0625f)	// 1/16
};
#endif // TWO_BPP
#endif // TWO_BY_TWO
// pointer swap method
static inline void swap(void **x, void **y) {
	void *t = *x;
	*x = *y;
	*y = t;
}

//////////////////////////////////////////////////////////////////////////////////
//	Calculate the root mean squared error of the image that is decoded from the	//
//	compressed data. This kernel calculates the total error for the red, green,	//
//	and blue channels of a single pixel and stores it in an array.				//
//																				//
//	orig		Pointer to the pixel data for the original image				//
//	a			Pointer to the upscaled pixel data for the final A image		//
//	b			Pointer to the upscaled pixel data for the final B image		//
//	out			Pointer to the array where the total error for the current 		//
//				pixel is stored													//
//	width		The width, in pixels, of the original image						//
//																				//
//////////////////////////////////////////////////////////////////////////////////
#ifdef GET_RMS_ERROR
__global__ void rms_error(unsigned int *orig, unsigned int *a, unsigned int *b,
						  unsigned int *out,  int width) {
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = (width * y) + x;

	unsigned int opixel = orig[idx];
	unsigned int apixel = a[idx];
	unsigned int bpixel = b[idx];
	float modbit = tex2D(modRef, x, y);
	float r_modbit = 1.0f - modbit;
	int c_red = ((float)MAKE_RED_PIXEL(bpixel)*modbit + 
		(float)MAKE_RED_PIXEL(apixel)*r_modbit);
	int c_green = ((float)MAKE_GREEN_PIXEL(bpixel)*modbit + 
		(float)MAKE_GREEN_PIXEL(apixel)*r_modbit);
	int c_blue = ((float)MAKE_BLUE_PIXEL(bpixel)*modbit + 
		(float)MAKE_BLUE_PIXEL(apixel)*r_modbit);

	out[idx] = SQR(((float)MAKE_RED_PIXEL(opixel) - c_red)) + 
			   SQR(((float)MAKE_GREEN_PIXEL(opixel) - c_green)) + 
			   SQR(((float)MAKE_BLUE_PIXEL(opixel) - c_blue));
}
#endif // GET_RMS_ERROR

#ifdef DECOMPRESS_PVR
__global__ void decompress(unsigned int *a, unsigned int *b, 
						   unsigned int *out, float *mod, int width) {
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = (width * y) + x;

	unsigned int apixel = a[idx];
	int a_red = MAKE_RED_PIXEL(apixel);
	int a_green = MAKE_GREEN_PIXEL(apixel);
	int a_blue = MAKE_BLUE_PIXEL(apixel);

	unsigned int bpixel = b[idx];
	int b_red = MAKE_RED_PIXEL(bpixel);
	int b_green = MAKE_GREEN_PIXEL(bpixel);
	int b_blue = MAKE_BLUE_PIXEL(bpixel);

	float modbit = mod[idx];
	float r_modbit = 1.0f - modbit;

	int c_red = ((float)b_red*modbit + (float)a_red*r_modbit);
	int c_green = ((float)b_green*modbit + (float)a_green*r_modbit);
	int c_blue = ((float)b_blue*modbit + (float)a_blue*r_modbit);

	out[idx] = MAKE_ARGB(c_red, c_green, c_blue);

}
#endif // DECOMPRESS_PVR

__global__ void encode_texture(const unsigned int* A, unsigned int *B, float *mod, 
							   int width, bool hasAlpha, unsigned int *out, 
							   unsigned int *mode, int outWidth, int outHeight) {
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned outy = y, outx = x;
	//int idx = (width * CLAMP(y, 15, outHeight-15)) + (CLAMP(x, 15, outWidth-15));
	/*if (x < 10) {
		outx = 15;
		outy = 15;
	} else if (x > outWidth-10) {
		outx = 80;
		outy = 45;
	} 
	if (y < 10) {
		outy = 15;
		outx = 15;
	} else if (y > outHeight-10) {
		outy = 45;
		outx = 80;
	}*/ 
	int idx = (width*outy) + outx;

	// Interleave lower 16 bits of x and y, so the bits of y
	// are in the even positions and bits from x in the odd;
	// outIdx is the resulting 32-bit Morton Number. 
	//unsigned int a = (2*x >= outWidth) ? 2*x - outWidth : 2*x;
	unsigned int a = 2*x;
	unsigned int b = y;

	a = (a | (a << 8)) & 0x00FF00FF;
	a = (a | (a << 4)) & 0x0F0F0F0F;
	a = (a | (a << 2)) & 0x33333333;
	a = (a | (a << 1)) & 0x55555555;

	b = (b | (b << 8)) & 0x00FF00FF;
	b = (b | (b << 4)) & 0x0F0F0F0F;
	b = (b | (b << 2)) & 0x33333333;
	b = (b | (b << 1)) & 0x55555555;

	// multiply by two because we encode 64-bit blocks not 32-bit blocks
	//int outIdx = 2 * CLAMP((a | (b << 1)), 0, outWidth*outHeight);
	// TODO: Fix this, currently only works if dimnesion is 2x as big
	int outIdx = (a | (b << 1));
	//outIdx += (2*x >= outWidth) ? (outWidth*outHeight) : 0;

	unsigned int apixel = A[idx];
	unsigned int bpixel = B[idx];
	unsigned int color = 0, modbits = 0;

	if (hasAlpha == false) {
		// encode color B
		color = 1;
		color = ((color<<LSHIFT_16BPP) | (MAKE_RED_PIXEL(bpixel)>>RSHIFT_16BPP));
		color = ((color<<LSHIFT_16BPP) | (MAKE_GREEN_PIXEL(bpixel)>>RSHIFT_16BPP));
		color = ((color<<LSHIFT_16BPP) | (MAKE_BLUE_PIXEL(bpixel)>>RSHIFT_16BPP));
		// encode color A
		color = ((color<<1) | 1);
		color = ((color<<LSHIFT_16BPP) | (MAKE_RED_PIXEL(apixel)>>RSHIFT_16BPP));
		color = ((color<<LSHIFT_16BPP) | (MAKE_GREEN_PIXEL(apixel)>>RSHIFT_16BPP));
		color = ((color<<(LSHIFT_16BPP-1)) | (MAKE_BLUE_PIXEL(apixel)>>(RSHIFT_16BPP+1)));

		// modulation mode 0
#ifdef TWO_BPP
        int modulation_mode = 1;//mode[idx];
        color = ((color<<1) | modulation_mode);
#else
		color = ((color<<1) | 0);
#endif  // TWO_BPP

		// encode modulation bits
#ifdef TWO_BPP
        int i, j, checker;
		float mbit;
		for (i = 3; i >= 0; i--) {
            if (modulation_mode == 1) {
                checker = !(i & 1);
                for (j = 3; j >= 0; j--) {
                    mbit = tex2D(modRef, (8*x) + (2 * j) + checker, i + (4*y));
                    if (mbit < 0.05f) {
                        modbits = (modbits<<2);
                    } else if (mbit < 0.5f) {
                        modbits = ((modbits<<2) | (0x1));
                    } else if (mbit < 0.95f) {
                        modbits = ((modbits<<2) | (0x2));
                    } else {
                        modbits = ((modbits<<2) | (0x3));
                    }
                }
            } else {
                for (j = 7; j >= 0; j--) {
                    mbit = tex2D(modRef, (8*x) + (2 * j) + checker, i + (4*y));
                    if (mbit < 0.5f) {
                        modbits = (modbits<<1);
                    } else {
                        modbits = ((modbits<<1) | (0x1));
                    }
                }
            }
		}
#else
		int i, j;
		float mbit;
		for (i = 3; i >= 0; i--) {
			for (j = 3; j >= 0; j--) {
				mbit = tex2D(modRef, j + (4*x), i + (4*y));
				if (mbit < 0.05f) {
					modbits = (modbits<<2);
				} else if (mbit < 0.5f) {
					modbits = ((modbits<<2) | (0x1));
				} else if (mbit < 0.95f) {
					modbits = ((modbits<<2) | (0x2));
				} else {
					modbits = ((modbits<<2) | (0x3));
				}
			}
		}
#endif

		out[outIdx] = modbits;
		out[outIdx+1] = color;
	} else {
		// TODO: implement this
	}
}
//////////////////////////////////////////////////////////////////////////////////
//									   Misc.									//
//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
//								  Optimization									//
//////////////////////////////////////////////////////////////////////////////////
__device__ float pythag(float a, float b) {
	float absa = fabsf(a);
	float absb = fabsf(b);
	if (absa > absb) return absa*sqrtf(1.0f+SQR((absb/absa)));
	else return (absb < EPSILON ? 0.0f : absb*sqrtf(1.0f+SQR(absa/absb)));
}

__device__ float sign(float a, float b) {
	return b >= 0.0f ? ( a >= 0.0f ? a : -a) : ( a >= 0.0f ? -a : a);
}

#ifndef TWO_BY_TWO
//////////////////////////////////////////////////////////////////////////////////
//	Calculate the Moore-Penrose pseudo-inverse of the weight matrix and use		//
//	it to compute the updated pair of A and B representative pixels.			//
//																				//
//	candidateA	Pointer to the output location of the A image					//
//	candidateB	Pointer to the output location of the B image					//
//	oldA		Pointer to the A image computed in the previous iteration		//
//	oldB		Pointer to the B image computed in the previous iteration		//
//	width		The width, in pixels, of the image to be compressed				//
//	err			Pointer to memory where errors are reported						//
//																				//
//////////////////////////////////////////////////////////////////////////////////
__global__ void moore_penrose_optimize(unsigned int *candidateA, unsigned int *candidateB,
									   unsigned int *oldA, unsigned int *oldB, 
									   int width, int height, int *err) {
	int thready = blockDim.y * blockIdx.y + threadIdx.y;
	int threadx = blockDim.x * blockIdx.x + threadIdx.x;
	const int  tIdx = (width*thready) + threadx;
	int red,red0=0,red1=0,green,green0=0,green1=0,blue,blue0=0,blue1=0;
	int x_offset = SVD_OFFSET_X*threadx-1, y_offset = SVD_OFFSET_Y*thready-1;
	int x, y, pixelx, pixely, index;
	float A00=0,A01=0,A10=0,A11=0,InverseA00,InverseA01,InverseA10,InverseA11;
	float a0, a1, modbit, dist, det;

	// Fetch the weight matrix for the 7x7 optimization window of the current
	// pair of representative values
	for (y = 0; y < SVD_DIM_Y; y++) {
		for (x = 0; x < SVD_DIM_X; x++) {
			index = y*SVD_DIM_X + x;
			pixelx = x_offset + x;
			pixely = y_offset + y;

			// fetch all the necessary values
			modbit = tex2D(modRef, CLAMP(pixelx, 0, width - 2), CLAMP(pixely, 0, height-2));
			red = tex2D(redCurrentRef, CLAMP(pixelx, 0, width-2), CLAMP(pixely, 0, height-2));
			green = tex2D(greenCurrentRef, CLAMP(pixelx, 0, width-2), CLAMP(pixely, 0, height-2));
			blue = tex2D(blueCurrentRef, CLAMP(pixelx, 0, width-2), CLAMP(pixely, 0, height-2));
			dist = tex1Dfetch(svdMatRef, index);
			a0 = dist*(1.0f - modbit);
			a1 = dist*modbit;	

			// A = TransposeA * A
			A00 += a0*a0;
			A01 += a0*a1;
			A10 += a1*a0;
			A11 += a1*a1;

			// colors = TransposeA * color
			red0 += a0*red;
			red1 += a1*red;

			green0 += a0*green;
			green1 += a1*green;

			blue0 += a0*blue;
			blue1 += a1*blue;
		}
	}

	// since we only have a 2x2 matrix, manually calculate inverse
	det = 1.0f / (A00*A11 - A01*A10);
	InverseA00 = det * A11;
	InverseA01 = -1 * det * A01;
	InverseA10 = -1 * det * A10;
	InverseA11 = det * A00;

	// Calculate new representative colors by using a "fix the error" approach.
	// We multiply the pseudo-inverse weight matrix by the difference between the
	// current and original images to get an "update" that is applied to each
	// representative value. We also limit how big the update can be to avoid
	// colors flying out of bounds.
	unsigned int oldColor;

	// A representative
	oldColor = oldA[tIdx];
	red = InverseA00*(float)red0 + InverseA01*(float)red1;
	green = InverseA00*(float)green0 + InverseA01*(float)green1;
	blue = InverseA00*(float)blue0 + InverseA01*(float)blue1;
#ifdef USE_PIXEL_UPDATE
	candidateA[tIdx] = MAKE_ARGB(
		CLAMP((int)MAKE_RED_PIXEL(oldColor) + CLAMP((int)red, -16, 16), 0, 255),
		CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + CLAMP((int)green, -16, 16), 0, 255),
		CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + CLAMP((int)blue, -16, 16), 0, 255));
#else
	candidateA[tIdx] = MAKE_ARGB(CLAMP(red, 0, 255), CLAMP(green, 0, 255), CLAMP(blue, 0, 255));
#endif // USE_PIXEL_UPDATE
	oldA[tIdx] = candidateA[tIdx];

	// B representative
	oldColor = oldB[tIdx];
	red = InverseA10*(float)red0 + InverseA11*(float)red1;
	green = InverseA10*(float)green0 + InverseA11*(float)green1;
	blue = InverseA10*(float)blue0 + InverseA11*(float)blue1;
#ifdef USE_PIXEL_UPDATE
	candidateB[tIdx] = MAKE_ARGB(
		CLAMP((int)MAKE_RED_PIXEL(oldColor) + CLAMP((int)red, -16, 16), 0, 255),
		CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + CLAMP((int)green, -16, 16), 0, 255),
		CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + CLAMP((int)blue, -16, 16), 0, 255));
#else
	candidateB[tIdx] = MAKE_ARGB(CLAMP(red, 0, 255), CLAMP(green, 0, 255), CLAMP(blue, 0, 255));
#endif // USE_PIXEL_UPDATE
	oldB[tIdx] = candidateB[tIdx];
}
#endif // TWO_BY_TWO
//////////////////////////////////////////////////////////////////////////////////
//	Calculates the singular value decomposition of the weight matrix and uses	//
//	it to compute teh pseudo-inverse. Then multiplies the pseudo-niverse by		//
//	the current color values to get the new values.								//
//	DOCUMENTATION: Numerical Recipes in C, 2nd Edition, chapter 2.6, 11.2,		//
//	& 11.3																		//
//																				//
//	candidateA	Pointer to the output location of the A image					//
//	candidateB	Pointer to the output location of the B image					//
//	oldA		Pointer to the A image computed in the previous iteration		//
//	oldB		Pointer to the B image computed in the previous iteration		//
//	width		The width, in pixels, of the image to be compressed				//
//	err			Pointer to memory where errors are reported						//
//																				//
//////////////////////////////////////////////////////////////////////////////////
__device__ void get_optimization_window(const int threadx, const int thready, 
                                        const int width, const int height,
                                        float red[SVD_MAT_HEIGHT],
                                        float green[SVD_MAT_HEIGHT],
                                        float blue[SVD_MAT_HEIGHT],
                                        float A[SVD_MAT_HEIGHT][SVD_MAT_WIDTH]) {
    int i, j, index, pixelx, pixely;
	float dist, modbit, r, gr, b;
	for (j = 0; j < SVD_DIM_Y; j++) {
		for (i = 0; i < SVD_DIM_X; i++) {
			// j is the index of the pixel we want
			index = j*SVD_DIM_X + i;
			pixelx = SVD_OFFSET_X*threadx-1 + i;
			pixely = SVD_OFFSET_Y*thready-1 + j;
            
			// fetch all the necessary values
#ifdef TWO_BPP
//            if (((pixely & 1) == 0 && (pixelx & 1) == 0) ||
//                ((pixely & 1) == 1 && (pixelx & 1) == 1)) {
//                modbit = tex2D(modRef, CLAMP(pixelx, 0, width), CLAMP(pixely, 0, height));
//            } else {
//                // We are in one of the pixels that is not directly encoded
//                float top, bottom, left, right;
//                top = tex2D(modRef, CLAMP(pixelx, 0, width), CLAMP(pixely-1, 0, height));
//                bottom = tex2D(modRef, CLAMP(pixelx, 0, width), CLAMP(pixely+1, 0, height));
//                left = tex2D(modRef, CLAMP(pixelx-1, 0, width), CLAMP(pixely, 0, height));
//                right = tex2D(modRef, CLAMP(pixelx+1, 0, width), CLAMP(pixely, 0, height));
//                modbit = (top + bottom + right + left) * 0.25f;
//            }
            modbit = tex2D(modRef, CLAMP(pixelx, 0, width), CLAMP(pixely, 0, height));
#else
			modbit = tex2D(modRef, CLAMP(pixelx, 0, width), CLAMP(pixely, 0, height));
#endif
			r = tex2D(redCurrentRef, CLAMP(pixelx, 0, width), CLAMP(pixely, 0, height));
			gr = tex2D(greenCurrentRef, CLAMP(pixelx, 0, width), CLAMP(pixely, 0, height));
			b = tex2D(blueCurrentRef, CLAMP(pixelx, 0, width), CLAMP(pixely, 0, height));
#ifdef TWO_BY_TWO
            // top left
			dist = tex1Dfetch(svdMatRefTL, index);
			A[index][0] = dist*(1.0f - modbit);
			A[index][1] = dist*modbit;
			red[index] = dist * r;
			green[index] = dist * gr;
			blue[index] = dist * b;
            
            //top right
			dist = tex1Dfetch(svdMatRefTR, index);
			A[index][2] = dist*(1.0f - modbit);
			A[index][3] = dist*modbit;
			red[index] += dist * r;
			green[index] += dist * gr;
			blue[index] += dist * b;
            
            // bottom left
			dist = tex1Dfetch(svdMatRefBL, index);
			A[index][4] = dist*(1.0f - modbit);
			A[index][5] = dist*modbit;
			red[index] += dist * r;
			green[index] += dist * gr;
			blue[index] += dist * b;
            
            // bottom right
			dist = tex1Dfetch(svdMatRefBR, index);
			A[index][6] = dist*(1.0f - modbit);
			A[index][7] = dist*modbit;
			red[index] += dist * r;
			green[index] += dist * gr;
			blue[index] += dist * b;
#else
            // there can be ONLY one!!!
			dist = tex1Dfetch(svdMatRef, index);
			A[index][0] = dist*(1.0f - modbit);
			A[index][1] = dist*modbit;
			red[index] = dist * r;
			green[index] = dist * gr;
			blue[index] = dist * b;
#endif	// TWO_BY_TWO
		}
	}
    
}

#ifdef USE_CHOLESKY
__global__ void cholesky_optimize(unsigned int *candidateA, unsigned int *candidateB,
                                  unsigned int *oldA, unsigned int *oldB, 
                                  int width, int height, int *err) {
    int thready = blockDim.y * blockIdx.y + threadIdx.y;
	int threadx = blockDim.x * blockIdx.x + threadIdx.x;
#ifdef TWO_BY_TWO
	threadx *= 2;
	thready *= 2;
#endif // TWO_BY_TWO
    float A[SVD_MAT_HEIGHT][SVD_MAT_WIDTH];
    float red[SVD_MAT_HEIGHT], green[SVD_MAT_HEIGHT], blue[SVD_MAT_HEIGHT];
    float AtA[SVD_MAT_WIDTH][SVD_MAT_WIDTH];
    float ry[SVD_MAT_WIDTH], gy[SVD_MAT_WIDTH], by[SVD_MAT_WIDTH];
    float rz[SVD_MAT_WIDTH], gz[SVD_MAT_WIDTH], bz[SVD_MAT_WIDTH];
    float rx[SVD_MAT_WIDTH], gx[SVD_MAT_WIDTH], bx[SVD_MAT_WIDTH];
    
    get_optimization_window(threadx, thready, width, height, red, green, blue, A);
    
    // Calculate transpose(A) * A
    int i, j, k;
    for (i = 0; i < SVD_MAT_WIDTH; i++) {
        for (j = 0; j <= i; j++) {
            AtA[j][i] = 0.0f;
            for (k = 0; k < SVD_MAT_HEIGHT; k++) {
                AtA[j][i] += A[k][i] * A[k][j];
            }
        }
    }
    
    // Cholesky decomposition of transpose(A)*A
    float diag[SVD_MAT_WIDTH];
    for (int i = 0; i < SVD_MAT_WIDTH; i++) {
        for (int j = i; j < SVD_MAT_WIDTH; j++) {
            float s = AtA[i][j];
            for (int k = i-1; k >= 0; k--) {
                s -= AtA[i][k] * AtA[j][k];
            }
            if (i == j) {
                diag[i] = sqrtf(s);
            } else {
                AtA[j][i] = s / diag[i];
            }
        }
    }
    for (i = 0; i < SVD_MAT_WIDTH; i++) {
        AtA[i][i] = diag[i];
    }
    
    // Calculate z = transpose(A) * b
    for (i = 0; i < SVD_MAT_WIDTH; i++) {
        rz[i] = 0.0f;
        gz[i] = 0.0f;
        bz[i] = 0.0f;
        for (j = 0; j < SVD_MAT_HEIGHT; j++) {
            rz[i] += A[j][i] * red[j];
            gz[i] += A[j][i] * green[j];
            bz[i] += A[j][i] * blue[j];
        }
    }
    
    // Solve Ly = z
	for (i = 0; i<SVD_MAT_WIDTH; i++) {
		float sumRed = rz[i];
        float sumGreen = gz[i];
        float sumBlue = bz[i];
		for (int k=i-1; k>=0; k--) {
            sumRed -= AtA[i][k] * ry[k];
            sumGreen -= AtA[i][k] * gy[k];
            sumBlue -= AtA[i][k] * by[k];
        }
        ry[i] = sumRed / AtA[i][i];
        gy[i] = sumGreen / AtA[i][i];
        by[i] = sumBlue / AtA[i][i];
	}
    
	// Solve Ltx = y
	for (i = SVD_MAT_WIDTH-1; i>=0; i--) {
		float sumRed = ry[i];
        float sumGreen = gy[i];
        float sumBlue = by[i];
		for (int k=i+1; k < SVD_MAT_WIDTH; k++) {
            sumRed -= AtA[k][i] * rx[k];
            sumGreen -= AtA[k][i] * gx[k];
            sumBlue -= AtA[k][i] * bx[k];
        }
		rx[i] = sumRed / AtA[i][i];
        gx[i] = sumGreen / AtA[i][i];
        bx[i] = sumBlue / AtA[i][i];
	}
#ifdef USE_PIXEL_UPDATE
	// Calculate new representative colors by using a "fix the error" approach.
	// We multiply the pseudo-inverse weight matrix by the difference between the
	// current and original images to get an "update" that is applied to each
	// representative value. We also limit how big the update can be to avoid
	// colors flying out of bounds.
	unsigned int oldColor;
	// Do top left for 2x2 case or only one for 1x1 case
	int tIdx = (width * thready) + threadx;
    int outRed, outGreen, outBlue, repIdx;
	oldColor = oldA[tIdx];
	outRed = CLAMP((int)rx[0], -16, 16);
	outGreen = CLAMP((int)gx[0], -16, 16);
	outBlue = CLAMP((int)bx[0], -16, 16);
	candidateA[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldA[tIdx] = candidateA[tIdx];
    
	oldColor = oldB[tIdx];
	outRed = CLAMP((int)rx[1], -16, 16);
	outGreen = CLAMP((int)bx[1], -16, 16);
	outBlue = CLAMP((int)gx[1], -16, 16);
	candidateB[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldB[tIdx] = candidateB[tIdx];
#ifdef TWO_BY_TWO
	// Top right case for 2x2
	tIdx = (width * thready) + threadx + 1;
	oldColor = oldA[tIdx];
	outRed = CLAMP((int)rx[2], -16, 16);
	outGreen = CLAMP((int)gx[2], -16, 16);
	outBlue = CLAMP((int)bx[2], -16, 16);
	candidateA[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldA[tIdx] = candidateA[tIdx];
    
	oldColor = oldB[tIdx];
	outRed = CLAMP((int)rx[3], -16, 16);
	outGreen = CLAMP((int)gx[3], -16, 16);
	outBlue = CLAMP((int)bx[3], -16, 16);
	candidateB[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldB[tIdx] = candidateB[tIdx];
    
	// Bottom left case
	tIdx = (width * (thready+1)) + threadx;
	oldColor = oldA[tIdx];
	outRed = CLAMP((int)rx[4], -16, 16);
	outGreen = CLAMP((int)bx[4], -16, 16);
	outBlue = CLAMP((int)gx[4], -16, 16);
	candidateA[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldA[tIdx] = candidateA[tIdx];
    
	oldColor = oldB[tIdx];
	outRed = CLAMP((int)rx[5], -16, 16);
	outGreen = CLAMP((int)gx[5], -16, 16);
	outBlue = CLAMP((int)bx[5], -16, 16);
	candidateB[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldB[tIdx] = candidateB[tIdx];
    
	// Bottom right case
	tIdx = (width * (thready+1)) + threadx + 1;
	oldColor = oldA[tIdx];
	outRed = CLAMP((int)rx[6], -16, 16);
	outGreen = CLAMP((int)gx[6], -16, 16);
	outBlue = CLAMP((int)bx[6], -16, 16);
	candidateA[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldA[tIdx] = candidateA[tIdx];
    
	oldColor = oldB[tIdx];
	outRed = CLAMP((int)rx[7], -16, 16);
	outGreen = CLAMP((int)gx[7], -16, 16);
	outBlue = CLAMP((int)bx[7], -16, 16);
	candidateB[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldB[tIdx] = candidateB[tIdx];
#endif // TWO_BY_TWO
#else
	// WARNING - THIS DOES NOT WORK
	// Calculate new representative colors by computing completely new colors.
	// We multiply the pseudo-inverse weight matrix by the actual colors of 
	// the original pixels in our window to get new representative colors.
	// do A candidate. If the new computed color is out of bounds, we discard
	// it and use the old one.
	// Top left case
    tIdx = (width * thready) + threadx;
	if ((frA[0] > 255) || (fgA[0] > 255) || (fbA[0] > 255) || (frA[0] < 0) || (fgA[0] < 0) || (fbA[0] < 0)) {
		outRed = MAKE_RED_PIXEL(oldA[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldA[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldA[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frA[0]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgA[0]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbA[0]), 0, 255);
	}
	candidateA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
    
	// do B candidate
	if ((frB[0] > 255) || (fgB[0] > 255) || (fbB[0] > 255) || (frB[0] < 0) || (fgB[0] < 0) || (fbB[0] < 0)) {
		outRed = MAKE_RED_PIXEL(oldB[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldB[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldB[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frB[0]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgB[0]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbB[0]), 0, 255);
	}
	candidateB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
#ifdef TWO_BY_TWO
	// Top right case
	tIdx = (width * (thready)) + threadx + 1;
	if ((frA[1] > 255) || (fgA[1] > 255) || (fbA[1] > 255) || (frA[1] < 0) || (fgA[1] < 0) || (fbA[1] < 0)) {
		outRed = MAKE_RED_PIXEL(oldA[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldA[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldA[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frA[1]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgA[1]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbA[1]), 0, 255);
	}
	candidateA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
    
	// do B candidate
	if ((frB[1] > 255) || (fgB[1] > 255) || (fbB[1] > 255) || (frB[1] < 0) || (fgB[1] < 0) || (fbB[1] < 0)) {
		outRed = MAKE_RED_PIXEL(oldB[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldB[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldB[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frB[1]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgB[1]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbB[1]), 0, 255);
	}
	candidateB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
    
	// Bottom left case
	tIdx = (width * (thready+1)) + threadx;
	if ((frA[2] > 255) || (fgA[2] > 255) || (fbA[2] > 255) || (frA[2] < 0) || (fgA[2] < 0) || (fbA[2] < 0)) {
		outRed = MAKE_RED_PIXEL(oldA[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldA[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldA[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frA[2]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgA[2]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbA[2]), 0, 255);
	}
	candidateA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
    
	// do B candidate
	if ((frB[2] > 255) || (fgB[2] > 255) || (fbB[2] > 255) || (frB[2] < 0) || (fgB[2] < 0) || (fbB[2] < 0)) {
		outRed = MAKE_RED_PIXEL(oldB[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldB[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldB[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frB[2]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgB[2]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbB[2]), 0, 255);
	}
	candidateB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
    
	// Bottom right case
	tIdx = (width * (thready+1)) + threadx + 1;
	if ((frA[3] > 255) || (fgA[3] > 255) || (fbA[3] > 255) || (frA[3] < 0) || (fgA[3] < 0) || (fbA[3] < 0)) {
		outRed = MAKE_RED_PIXEL(oldA[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldA[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldA[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frA[3]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgA[3]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbA[3]), 0, 255);
	}
	candidateA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
    
	// do B candidate
	if ((frB[3] > 255) || (fgB[3] > 255) || (fbB[3] > 255) || (frB[3] < 0) || (fgB[3] < 0) || (fbB[3] < 0)) {
		outRed = MAKE_RED_PIXEL(oldB[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldB[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldB[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frB[3]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgB[3]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbB[3]), 0, 255);
	}
	candidateB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
#endif // TWO_BY_TWO
#endif // USE_PIXEL_UPDATE
    
}
#endif // USE_CHOLESKY

#ifdef USE_JAMA_SVD
__global__ void svd_optimize(unsigned int *candidateA, unsigned int *candidateB, 
                             unsigned int *oldA, unsigned int *oldB, 
                             int width, int height, int *err) {
    int thready = blockDim.y * blockIdx.y + threadIdx.y;
	int threadx = blockDim.x * blockIdx.x + threadIdx.x;
#ifdef TWO_BY_TWO
	threadx *= 2;
	thready *= 2;
#endif // TWO_BY_TWO
	const int maxits=30;
	int i, j, k, tIdx = (width*thready) + threadx;
    float w;
	float red[SVD_MAT_HEIGHT],green[SVD_MAT_HEIGHT],
    blue[SVD_MAT_HEIGHT];
    
	// column major order for SVD
	float A[SVD_MAT_HEIGHT][SVD_MAT_WIDTH], s[SVD_MAT_WIDTH], 
    V[SVD_MAT_WIDTH][SVD_MAT_WIDTH], U[SVD_MAT_HEIGHT][SVD_MAT_WIDTH],
    InverseA[SVD_MAT_WIDTH][SVD_MAT_HEIGHT];
    
	// Fetch the weight matrix for the optimization window of the current
	// pair of representative pixels
	get_optimization_window(threadx, thready, width, height, red, green, blue, A);
    
    // Derived from LINPACK code.
    // Initialize.
    const int m = SVD_MAT_HEIGHT;
    const int n = SVD_MAT_WIDTH;
    
    /* Apparently the failing cases are only a proper subset of (m<n), 
	 so let's not throw error.  Correct fix to come later?
     if (m<n) {
     throw new IllegalArgumentException("Jama SVD only works for m >= n"); }
     */
    const int nu = SVD_MAT_WIDTH;
    float e[SVD_MAT_WIDTH], work[SVD_MAT_HEIGHT];
    bool wantu = true;
    bool wantv = true;
    
    // Reduce A to bidiagonal form, storing the diagonal elements
    // in s and the super-diagonal elements in e.
    
    int nct = min(m-1,n);
    int nrt = max(0,min(n-2,m));
    for (int k = 0; k < max(nct,nrt); k++) {
        if (k < nct) {
            
            // Compute the transformation for the k-th column and
            // place the k-th diagonal in s[k].
            // Compute 2-norm of k-th column without under/overflow.
            s[k] = 0;
            for (int i = k; i < m; i++) {
                s[k] = pythag(s[k],A[i][k]);
            }
            if (s[k] != 0.0) {
                if (A[k][k] < 0.0) {
                    s[k] = -s[k];
                }
                for (int i = k; i < m; i++) {
                    A[i][k] /= s[k];
                }
                A[k][k] += 1.0;
            }
            s[k] = -s[k];
        }
        for (int j = k+1; j < n; j++) {
            if ((k < nct) & (s[k] != 0.0))  {
                
                // Apply the transformation.
                
                float t = 0;
                for (int i = k; i < m; i++) {
                    t += A[i][k]*A[i][j];
                }
                t = -t/A[k][k];
                for (int i = k; i < m; i++) {
                    A[i][j] += t*A[i][k];
                }
            }
            
            // Place the k-th row of A into e for the
            // subsequent calculation of the row transformation.
            
            e[j] = A[k][j];
        }
        if (wantu & (k < nct)) {
            
            // Place the transformation in U for subsequent back
            // multiplication.
            
            for (int i = k; i < m; i++) {
                U[i][k] = A[i][k];
            }
        }
        if (k < nrt) {
            
            // Compute the k-th row transformation and place the
            // k-th super-diagonal in e[k].
            // Compute 2-norm without under/overflow.
            e[k] = 0;
            for (int i = k+1; i < n; i++) {
                e[k] = pythag(e[k],e[i]);
            }
            if (e[k] != 0.0) {
                if (e[k+1] < 0.0) {
                    e[k] = -e[k];
                }
                for (int i = k+1; i < n; i++) {
                    e[i] /= e[k];
                }
                e[k+1] += 1.0;
            }
            e[k] = -e[k];
            if ((k+1 < m) & (e[k] != 0.0)) {
                
                // Apply the transformation.
                
                for (int i = k+1; i < m; i++) {
                    work[i] = 0.0;
                }
                for (int j = k+1; j < n; j++) {
                    for (int i = k+1; i < m; i++) {
                        work[i] += e[j]*A[i][j];
                    }
                }
                for (int j = k+1; j < n; j++) {
                    float t = -e[j]/e[k+1];
                    for (int i = k+1; i < m; i++) {
                        A[i][j] += t*work[i];
                    }
                }
            }
            if (wantv) {
                
                // Place the transformation in V for subsequent
                // back multiplication.
                
                for (int i = k+1; i < n; i++) {
                    V[i][k] = e[i];
                }
            }
        }
    }
    
    // Set up the final bidiagonal matrix or order p.
    
    int p = min(n,m+1);
    if (nct < n) {
        s[nct] = A[nct][nct];
    }
    if (m < p) {
        s[p-1] = 0.0;
    }
    if (nrt+1 < p) {
        e[nrt] = A[nrt][p-1];
    }
    e[p-1] = 0.0;
    
    // If required, generate U.
    
    if (wantu) {
        for (int j = nct; j < nu; j++) {
            for (int i = 0; i < m; i++) {
                U[i][j] = 0.0;
            }
            U[j][j] = 1.0;
        }
        for (int k = nct-1; k >= 0; k--) {
            if (s[k] != 0.0) {
                for (int j = k+1; j < nu; j++) {
                    float t = 0;
                    for (int i = k; i < m; i++) {
                        t += U[i][k]*U[i][j];
                    }
                    t = -t/U[k][k];
                    for (int i = k; i < m; i++) {
                        U[i][j] += t*U[i][k];
                    }
                }
                for (int i = k; i < m; i++ ) {
                    U[i][k] = -U[i][k];
                }
                U[k][k] = 1.0 + U[k][k];
                for (int i = 0; i < k-1; i++) {
                    U[i][k] = 0.0;
                }
            } else {
                for (int i = 0; i < m; i++) {
                    U[i][k] = 0.0;
                }
                U[k][k] = 1.0;
            }
        }
    }
    
    // If required, generate V.
    
    if (wantv) {
        for (int k = n-1; k >= 0; k--) {
            if ((k < nrt) & (e[k] != 0.0)) {
                for (int j = k+1; j < nu; j++) {
                    float t = 0;
                    for (int i = k+1; i < n; i++) {
                        t += V[i][k]*V[i][j];
                    }
                    t = -t/V[k+1][k];
                    for (int i = k+1; i < n; i++) {
                        V[i][j] += t*V[i][k];
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                V[i][k] = 0.0;
            }
            V[k][k] = 1.0;
        }
    }
    
    // Main iteration loop for the singular values.
    
    int pp = p-1;
    int iter = 0;
    float eps = pow(2.0,-52.0);
    float tiny = pow(2.0,-966.0);
    while (p > 0) {
        int k,kase;
        
        // Here is where a test for too many iterations would go.
        if (iter > maxits) {
            *err = 1;
            break;
        }
        // This section of the program inspects for
        // negligible elements in the s and e arrays.  On
        // completion the variables kase and k are set as follows.
        
        // kase = 1     if s(p) and e[k-1] are negligible and k<p
        // kase = 2     if s(k) is negligible and k<p
        // kase = 3     if e[k-1] is negligible, k<p, and
        //              s(k), ..., s(p) are not negligible (qr step).
        // kase = 4     if e(p-1) is negligible (convergence).
        
        for (k = p-2; k >= -1; k--) {
            if (k == -1) {
                break;
            }
            if (fabsf(e[k]) <=
                tiny + eps*(fabsf(s[k]) + fabsf(s[k+1]))) {
                e[k] = 0.0;
                break;
            }
        }
        if (k == p-2) {
            kase = 4;
        } else {
            int ks;
            for (ks = p-1; ks >= k; ks--) {
                if (ks == k) {
                    break;
                }
                float t = (ks != p ? fabsf(e[ks]) : 0.) + 
                (ks != k+1 ? fabsf(e[ks-1]) : 0.);
                if (fabsf(s[ks]) <= tiny + eps*t)  {
                    s[ks] = 0.0;
                    break;
                }
            }
            if (ks == k) {
                kase = 3;
            } else if (ks == p-1) {
                kase = 1;
            } else {
                kase = 2;
                k = ks;
            }
        }
        k++;
        
        // Perform the task indicated by kase.
        
        switch (kase) {
                
                // Deflate negligible s(p).
                
            case 1: {
                float f = e[p-2];
                e[p-2] = 0.0;
                for (int j = p-2; j >= k; j--) {
                    float t = pythag(s[j],f);
                    float cs = s[j]/t;
                    float sn = f/t;
                    s[j] = t;
                    if (j != k) {
                        f = -sn*e[j-1];
                        e[j-1] = cs*e[j-1];
                    }
                    if (wantv) {
                        for (int i = 0; i < n; i++) {
                            t = cs*V[i][j] + sn*V[i][p-1];
                            V[i][p-1] = -sn*V[i][j] + cs*V[i][p-1];
                            V[i][j] = t;
                        }
                    }
                }
            }
                break;
                
                // Split at negligible s(k).
                
            case 2: {
                float f = e[k-1];
                e[k-1] = 0.0;
                for (int j = k; j < p; j++) {
                    float t = pythag(s[j],f);
                    float cs = s[j]/t;
                    float sn = f/t;
                    s[j] = t;
                    f = -sn*e[j];
                    e[j] = cs*e[j];
                    if (wantu) {
                        for (int i = 0; i < m; i++) {
                            t = cs*U[i][j] + sn*U[i][k-1];
                            U[i][k-1] = -sn*U[i][j] + cs*U[i][k-1];
                            U[i][j] = t;
                        }
                    }
                }
            }
                break;
                
                // Perform one qr step.
                
            case 3: {
                
                // Calculate the shift.
                
                float scale = max(max(max(max(fabsf(s[p-1]),fabsf(s[p-2])),fabsf(e[p-2])), 
                                                fabsf(s[k])),fabsf(e[k]));
                float sp = s[p-1]/scale;
                float spm1 = s[p-2]/scale;
                float epm1 = e[p-2]/scale;
                float sk = s[k]/scale;
                float ek = e[k]/scale;
                float b = ((spm1 + sp)*(spm1 - sp) + epm1*epm1)/2.0;
                float c = (sp*epm1)*(sp*epm1);
                float shift = 0.0;
                if ((b != 0.0) | (c != 0.0)) {
                    shift = sqrtf(b*b + c);
                    if (b < 0.0) {
                        shift = -shift;
                    }
                    shift = c/(b + shift);
                }
                float f = (sk + sp)*(sk - sp) + shift;
                float g = sk*ek;
                
                // Chase zeros.
                
                for (int j = k; j < p-1; j++) {
                    float t = pythag(f,g);
                    float cs = f/t;
                    float sn = g/t;
                    if (j != k) {
                        e[j-1] = t;
                    }
                    f = cs*s[j] + sn*e[j];
                    e[j] = cs*e[j] - sn*s[j];
                    g = sn*s[j+1];
                    s[j+1] = cs*s[j+1];
                    if (wantv) {
                        for (int i = 0; i < n; i++) {
                            t = cs*V[i][j] + sn*V[i][j+1];
                            V[i][j+1] = -sn*V[i][j] + cs*V[i][j+1];
                            V[i][j] = t;
                        }
                    }
                    t = pythag(f,g);
                    cs = f/t;
                    sn = g/t;
                    s[j] = t;
                    f = cs*e[j] + sn*s[j+1];
                    s[j+1] = -sn*e[j] + cs*s[j+1];
                    g = sn*e[j+1];
                    e[j+1] = cs*e[j+1];
                    if (wantu && (j < m-1)) {
                        for (int i = 0; i < m; i++) {
                            t = cs*U[i][j] + sn*U[i][j+1];
                            U[i][j+1] = -sn*U[i][j] + cs*U[i][j+1];
                            U[i][j] = t;
                        }
                    }
                }
                e[p-2] = f;
                iter = iter + 1;
            }
                break;
                
                // Convergence.
                
            case 4: {
                
                // Make the singular values positive.
                
                if (s[k] <= 0.0) {
                    s[k] = (s[k] < 0.0 ? -s[k] : 0.0);
                    if (wantv) {
                        for (int i = 0; i <= pp; i++) {
                            V[i][k] = -V[i][k];
                        }
                    }
                }
                
                // Order the singular values.
                
//                while (k < pp) {
//                    if (s[k] >= s[k+1]) {
//                        break;
//                    }
//                    float t = s[k];
//                    s[k] = s[k+1];
//                    s[k+1] = t;
//                    if (wantv && (k < n-1)) {
//                        for (int i = 0; i < n; i++) {
//                            t = V[i][k+1]; V[i][k+1] = V[i][k]; V[i][k] = t;
//                        }
//                    }
//                    if (wantu && (k < m-1)) {
//                        for (int i = 0; i < m; i++) {
//                            t = U[i][k+1]; U[i][k+1] = U[i][k]; U[i][k] = t;
//                        }
//                    }
//                    k++;
//                }  
                iter = 0;
                p--;
            }
                break;
        }
    }
    
    // begin constructing inverse matrix
	// inverse(A) = V * inverse(W) * transpose(U)
    
	// Step 1: transpose(A) = inverse(W) * transpose(U)
	for (i = 0; i < SVD_MAT_WIDTH; i++) {
		w = 1.0f / s[i];
		for (j=0; j < SVD_MAT_HEIGHT; j++) {
			U[j][i] *= w;
		}
	}
    
	// Step 2: inverse(A) = V * transpose(U)
	for (i = 0; i < SVD_MAT_WIDTH; i++) {			// rows of Eigenbasis
		for (j = 0; j < SVD_MAT_HEIGHT; j++) {		// colums of U transpose
			InverseA[i][j] = 0.0f;
			for (k = 0; k < SVD_MAT_WIDTH; k++) {	// dimension of overlap
				// U is stored as a transpose so we must flip indices accordingly
				// In other words U[row][column] becomes U[column][row]
				InverseA[i][j] += V[i][k] * U[j][k];
			}
		}
	}
    
	// compute optimized A and B values
	float frA[4], fgA[4], fbA[4], frB[4], fgB[4], fbB[4];
	int outRed, outGreen, outBlue, repIdx;
	for (j = 0; j < SVD_MAT_WIDTH; j+=2) {
		repIdx = (j>>1);	// repIdx = j / 2
		frA[repIdx] = fgA[repIdx] = fbA[repIdx] = frB[repIdx] = fgB[repIdx] = fbB[repIdx] = 0.0f;
		for (i = 0; i < SVD_MAT_HEIGHT; i++) {
			frA[repIdx] += InverseA[j][i] * red[i];
			fgA[repIdx] += InverseA[j][i] * green[i];
			fbA[repIdx] += InverseA[j][i] * blue[i];
			frB[repIdx] += InverseA[j+1][i] * red[i];
			fgB[repIdx] += InverseA[j+1][i] * green[i];
			fbB[repIdx] += InverseA[j+1][i] * blue[i];
		}
	}
    
#ifdef USE_PIXEL_UPDATE
	// Calculate new representative colors by using a "fix the error" approach.
	// We multiply the pseudo-inverse weight matrix by the difference between the
	// current and original images to get an "update" that is applied to each
	// representative value. We also limit how big the update can be to avoid
	// colors flying out of bounds.
	unsigned int oldColor;
	// Do top left for 2x2 case or only one for 1x1 case
	tIdx = (width * thready) + threadx;
	oldColor = oldA[tIdx];
	outRed = CLAMP((int)frA[0], -16, 16);
	outGreen = CLAMP((int)fgA[0], -16, 16);
	outBlue = CLAMP((int)fbA[0], -16, 16);
	candidateA[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldA[tIdx] = candidateA[tIdx];
    
	oldColor = oldB[tIdx];
	outRed = CLAMP((int)frB[0], -16, 16);
	outGreen = CLAMP((int)fgB[0], -16, 16);
	outBlue = CLAMP((int)fbB[0], -16, 16);
	candidateB[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldB[tIdx] = candidateB[tIdx];
#ifdef TWO_BY_TWO
	// Top right case for 2x2
	tIdx = (width * thready) + threadx + 1;
	oldColor = oldA[tIdx];
	outRed = CLAMP((int)frA[1], -16, 16);
	outGreen = CLAMP((int)fgA[1], -16, 16);
	outBlue = CLAMP((int)fbA[1], -16, 16);
	candidateA[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldA[tIdx] = candidateA[tIdx];
    
	oldColor = oldB[tIdx];
	outRed = CLAMP((int)frB[1], -16, 16);
	outGreen = CLAMP((int)fgB[1], -16, 16);
	outBlue = CLAMP((int)fbB[1], -16, 16);
	candidateB[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldB[tIdx] = candidateB[tIdx];
    
	// Bottom left case
	tIdx = (width * (thready+1)) + threadx;
	oldColor = oldA[tIdx];
	outRed = CLAMP((int)frA[2], -16, 16);
	outGreen = CLAMP((int)fgA[2], -16, 16);
	outBlue = CLAMP((int)fbA[2], -16, 16);
	candidateA[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldA[tIdx] = candidateA[tIdx];
    
	oldColor = oldB[tIdx];
	outRed = CLAMP((int)frB[2], -16, 16);
	outGreen = CLAMP((int)fgB[2], -16, 16);
	outBlue = CLAMP((int)fbB[2], -16, 16);
	candidateB[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldB[tIdx] = candidateB[tIdx];
    
	// Bottom right case
	tIdx = (width * (thready+1)) + threadx + 1;
	oldColor = oldA[tIdx];
	outRed = CLAMP((int)frA[3], -16, 16);
	outGreen = CLAMP((int)fgA[3], -16, 16);
	outBlue = CLAMP((int)fbA[3], -16, 16);
	candidateA[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldA[tIdx] = candidateA[tIdx];
    
	oldColor = oldB[tIdx];
	outRed = CLAMP((int)frB[3], -16, 16);
	outGreen = CLAMP((int)fgB[3], -16, 16);
	outBlue = CLAMP((int)fbB[3], -16, 16);
	candidateB[tIdx] = MAKE_ARGB(
                                 CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
                                 CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
                                 CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldB[tIdx] = candidateB[tIdx];
#endif // TWO_BY_TWO
#else
	// WARNING - THIS DOES NOT WORK
	// Calculate new representative colors by computing completely new colors.
	// We multiply the pseudo-inverse weight matrix by the actual colors of 
	// the original pixels in our window to get new representative colors.
	// do A candidate. If the new computed color is out of bounds, we discard
	// it and use the old one.
	// Top left case
    tIdx = (width * thready) + threadx;
	if ((frA[0] > 255) || (fgA[0] > 255) || (fbA[0] > 255) || (frA[0] < 0) || (fgA[0] < 0) || (fbA[0] < 0)) {
		outRed = MAKE_RED_PIXEL(oldA[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldA[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldA[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frA[0]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgA[0]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbA[0]), 0, 255);
	}
	candidateA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
    
	// do B candidate
	if ((frB[0] > 255) || (fgB[0] > 255) || (fbB[0] > 255) || (frB[0] < 0) || (fgB[0] < 0) || (fbB[0] < 0)) {
		outRed = MAKE_RED_PIXEL(oldB[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldB[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldB[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frB[0]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgB[0]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbB[0]), 0, 255);
	}
	candidateB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
#ifdef TWO_BY_TWO
	// Top right case
	tIdx = (width * (thready)) + threadx + 1;
	if ((frA[1] > 255) || (fgA[1] > 255) || (fbA[1] > 255) || (frA[1] < 0) || (fgA[1] < 0) || (fbA[1] < 0)) {
		outRed = MAKE_RED_PIXEL(oldA[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldA[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldA[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frA[1]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgA[1]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbA[1]), 0, 255);
	}
	candidateA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
    
	// do B candidate
	if ((frB[1] > 255) || (fgB[1] > 255) || (fbB[1] > 255) || (frB[1] < 0) || (fgB[1] < 0) || (fbB[1] < 0)) {
		outRed = MAKE_RED_PIXEL(oldB[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldB[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldB[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frB[1]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgB[1]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbB[1]), 0, 255);
	}
	candidateB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
    
	// Bottom left case
	tIdx = (width * (thready+1)) + threadx;
	if ((frA[2] > 255) || (fgA[2] > 255) || (fbA[2] > 255) || (frA[2] < 0) || (fgA[2] < 0) || (fbA[2] < 0)) {
		outRed = MAKE_RED_PIXEL(oldA[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldA[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldA[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frA[2]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgA[2]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbA[2]), 0, 255);
	}
	candidateA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
    
	// do B candidate
	if ((frB[2] > 255) || (fgB[2] > 255) || (fbB[2] > 255) || (frB[2] < 0) || (fgB[2] < 0) || (fbB[2] < 0)) {
		outRed = MAKE_RED_PIXEL(oldB[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldB[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldB[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frB[2]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgB[2]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbB[2]), 0, 255);
	}
	candidateB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
    
	// Bottom right case
	tIdx = (width * (thready+1)) + threadx + 1;
	if ((frA[3] > 255) || (fgA[3] > 255) || (fbA[3] > 255) || (frA[3] < 0) || (fgA[3] < 0) || (fbA[3] < 0)) {
		outRed = MAKE_RED_PIXEL(oldA[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldA[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldA[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frA[3]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgA[3]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbA[3]), 0, 255);
	}
	candidateA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
    
	// do B candidate
	if ((frB[3] > 255) || (fgB[3] > 255) || (fbB[3] > 255) || (frB[3] < 0) || (fgB[3] < 0) || (fbB[3] < 0)) {
		outRed = MAKE_RED_PIXEL(oldB[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldB[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldB[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frB[3]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgB[3]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbB[3]), 0, 255);
	}
	candidateB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
#endif // TWO_BY_TWO
#endif // USE_PIXEL_UPDATE
}
#endif // USE_JAMA_SVD

#ifdef USE_SVD
__global__ void svd_optimize(unsigned int *candidateA, unsigned int *candidateB,
									unsigned int *oldA, unsigned int *oldB, 
									int width,int height,  int *err) {
	int thready = blockDim.y * blockIdx.y + threadIdx.y;
	int threadx = blockDim.x * blockIdx.x + threadIdx.x;
#ifdef TWO_BY_TWO
	threadx *= 2;
	thready *= 2;
#endif // TWO_BY_TWO
	const int maxits=30;
	int tIdx = (width*thready) + threadx;
	int i,its,j,jj,k,l=0,nm=0;
	float red[SVD_MAT_HEIGHT],green[SVD_MAT_HEIGHT],
		blue[SVD_MAT_HEIGHT];
	bool flag;
	float anorm,c,f,g,h,s,scale,w,x,y,z,rv[SVD_MAT_WIDTH]; 
	// column major order for SVD
	float A[SVD_MAT_HEIGHT][SVD_MAT_WIDTH], Eigenvalues[SVD_MAT_WIDTH], 
    Eigenbasis[SVD_MAT_WIDTH][SVD_MAT_WIDTH]; 
		//InverseA[SVD_MAT_WIDTH][SVD_MAT_HEIGHT];

	// Fetch the weight matrix for the optimization window of the current
	// pair of representative pixels
	get_optimization_window(threadx, thready, width, height, red, green, blue, A);

	// Householder reduction to bidiagonal form
	g = scale = anorm = 0.0f;
	for (i=0; i<SVD_MAT_WIDTH; i++) {
		l = i+1;
		rv[i] = scale*g;
		g = s = scale = 0.0f;
		if (i<SVD_MAT_HEIGHT) {
			for (k=i; k<SVD_MAT_HEIGHT; k++) scale+= fabsf(A[k][i]);
			if (scale!=0.0f) {
				for (k=i; k<SVD_MAT_HEIGHT; k++) {
					A[k][i]/= scale;
					s+= SQR(A[k][i]);
				}
				f = A[i][i];
				g = -sign(sqrtf(s),f);
				h = f*g-s;
				A[i][i] = f-g;
				for (j=l; j<SVD_MAT_WIDTH; j++) {
				  for (s=0.0f,k=i; k<SVD_MAT_HEIGHT; k++) s+= A[k][i]*A[k][j];
					f = s/h;
					for (k=i; k<SVD_MAT_HEIGHT; k++) A[k][j]+= f*A[k][i];
				}
				for (k=i; k<SVD_MAT_HEIGHT; k++) A[k][i]*= scale;
			}
		}
		Eigenvalues[i] = scale*g;
		g = s = scale = 0.0f;
		if (i<SVD_MAT_HEIGHT && i!=SVD_MAT_WIDTH-1) {
			for (k=l; k<SVD_MAT_WIDTH; k++) scale+= fabsf(A[i][k]);
			if (scale!=0.0f)  {
				for(k=l; k<SVD_MAT_WIDTH; k++) {
					A[i][k]/= scale;
					s+= SQR(A[i][k]);
				}
				f = A[i][l];
				g = -sign(sqrtf(s),f);
				h = f*g-s;
				A[i][l] = f-g;
				for (k=l; k<SVD_MAT_WIDTH; k++) rv[k] = A[i][k]/h;
				for (j=l; j<SVD_MAT_HEIGHT; j++) {
				  for(s=0.0f,k=l; k<SVD_MAT_WIDTH; k++) s+= A[j][k]*A[i][k];
					for(k=l; k<SVD_MAT_WIDTH; k++) A[j][k]+= s*rv[k];
				}
				for(k=l; k<SVD_MAT_WIDTH; k++) A[i][k]*= scale;
			}
		}
		anorm = fmaxf(anorm,(fabsf(Eigenvalues[i])+fabsf(rv[i])));
	}
	
	// Accumulate right-hand side updates
	for(i=SVD_MAT_WIDTH-1; i>=0; i--) {
		if (i<SVD_MAT_WIDTH-1) {
		  if (g!=0.0f) {
				for (j=l; j<SVD_MAT_WIDTH; j++) Eigenbasis[j][i] = (A[i][j]/A[i][l])/g;
				for (j=l; j<SVD_MAT_WIDTH; j++) {
				  for(s=0.0f,k=l; k<SVD_MAT_WIDTH; k++) s+= A[i][k]*Eigenbasis[k][j];
					for(k=l; k<SVD_MAT_WIDTH; k++) Eigenbasis[k][j]+= s*Eigenbasis[k][i];
				}
			}
			for (j=l; j<SVD_MAT_WIDTH; j++) Eigenbasis[i][j] = Eigenbasis[j][i] = 0.0f;
		}
		Eigenbasis[i][i] = 1.0f;
		g = rv[i];
		l = i;
	}
		
	// Accumulate left-hand side updates
	for (i=min(SVD_MAT_HEIGHT-1,SVD_MAT_WIDTH-1); i>=0; i--) {
		l = i+1;
		g = Eigenvalues[i];
		for(j=l;j<SVD_MAT_WIDTH;j++) A[i][j] = 0.0f;
		if (g!=0.0f) {
		  g = 1.0f/g;
			for (j=l; j<SVD_MAT_WIDTH; j++) {
			  for (s=0.0f,k=l; k<SVD_MAT_HEIGHT; k++) s+= A[k][i]*A[k][j];
				f = (s/A[i][i])*g;
				for (k=i; k<SVD_MAT_HEIGHT; k++) A[k][j]+= f*A[k][i];
			}
			for (j=i; j<SVD_MAT_HEIGHT; j++) A[j][i]*= g;
		}
		else for (j=i; j<SVD_MAT_HEIGHT; j++) A[j][i] = 0.0f;
		A[i][i]+= 1.0f;
	}
	
	// diagonalization of the bidiagonal form: loop over singular values, and
	// over allowed iterations
	for (k=SVD_MAT_WIDTH-1; k>=0; k--) {
		for (its=0; its<maxits; its++) {
			flag = true;
			for (l=k; l>=0; l--) {			// test for splitting
				nm = l-1;
				if ((float)(fabsf(rv[l])+anorm) == anorm) {
					flag =  false;
					break;
				}
				if ((float)(fabsf(Eigenvalues[nm])+anorm) == anorm) break;
			}
			if (flag) {
			  c = 0.0f;
			  s = 1.0f;
				for (i=l; i<=k; i++) {
					f = s*rv[i];
					rv[i]*= c;
					if ((float)(fabsf(f)+anorm) == anorm) break;
					g = Eigenvalues[i];
					h = pythag(f,g);
					Eigenvalues[i] = h;
					h = 1.0f/h;
					c = g*h;
					s = -f*h;
					for (j=0; j<SVD_MAT_HEIGHT; j++) {
						y = A[j][nm];
						z = A[j][i];
						A[j][nm] = y*c+z*s;
						A[j][i]  = z*c-y*s;
					}
				}
			}
			z = Eigenvalues[k];
			if (l==k) {					// convergence
				if(z<0.0f) {			// singular value is made non-negative
					Eigenvalues[k] = -z;
					for(j=0; j<SVD_MAT_WIDTH; j++) Eigenbasis[j][k] *= -1.0f;
				}
				break;
			}
			
			if(its>=maxits) { *err = 1; } // error check
			x = Eigenvalues[l];			// shift from bottom 2-by-2 minor
			nm = k-1;
			y = Eigenvalues[nm];
			g = rv[nm];
			h = rv[k];
			f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0f*h*y);
			g = pythag(f,1.0);
			f = ((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x;
			c = s = 1.0f;				// next QR transformation
			for (j=l; j<=nm; j++) {
				i = j+1;
				g = rv[i];
				y = Eigenvalues[i];
				h = s*g;
				g = c*g;
				z = pythag(f,h);
				rv[j] = z;
				c = f/z;
				s = h/z;
				f = x*c+g*s;
				g = g*c-x*s;
				h = y*s;
				y*= c;
				for(jj=0;jj<SVD_MAT_WIDTH;jj++) {
					x = Eigenbasis[jj][j];
					z = Eigenbasis[jj][i];
					Eigenbasis[jj][j] = x*c+z*s;
					Eigenbasis[jj][i] = z*c-x*s;
				}
				z = pythag(f,h);
				Eigenvalues[j] = z;		// rotation can be arbitrary if z = 0
				if (z!=0.0f) {
				  z = 1.0f / z;
					c = f*z;
					s = h*z;
				}
				f = c*g+s*y;
				x = c*y-s*g;
				for (jj=0; jj<SVD_MAT_HEIGHT; jj++) {
					y = A[jj][j];
					z = A[jj][i];
					A[jj][j] = y*c+z*s;
					A[jj][i] = z*c-y*s;
				}
			}
			rv[l] = 0.0f;
			rv[k] = f;
			Eigenvalues[k] = x;
		}
	}

	// begin constructing inverse matrix
	// inverse(A) = V * inverse(W) * transpose(A)

	// Step 1: transpose(A) = inverse(W) * transpose(A)
	for (i = 0; i < SVD_MAT_WIDTH; i++) {
		w = 1.0f / Eigenvalues[i];
		for (j=0; j < SVD_MAT_HEIGHT; j++) {
			A[j][i] *= w;
		}
	}

//	// Step 2: inverse(A) = V * transpose(A)
//	for (i = 0; i < SVD_MAT_WIDTH; i++) {			// rows of Eigenbasis
//		for (j = 0; j < SVD_MAT_HEIGHT; j++) {		// colums of A transpose
//			InverseA[i][j] = 0.0f;
//			for (k = 0; k < SVD_MAT_WIDTH; k++) {	// dimension of overlap
//				// A is stored as a transpose so we must flip indices accordingly
//				// In other words A[row][column] becomes A[column][row]
//				InverseA[i][j] += Eigenbasis[i][k] * A[j][k];
//			}
//		}
//	}
//    
//    // compute optimized A and B values
//	float frA[4], fgA[4], fbA[4], frB[4], fgB[4], fbB[4];
//    float tempR[SVD_MAT_WIDTH], tempG[SVD_MAT_WIDTH], tempB[SVD_MAT_WIDTH];
//	int outRed, outGreen, outBlue, repIdx;
//    for (j = 0; j < SVD_MAT_WIDTH; j+=2) {
//		repIdx = (j>>1);	// repIdx = j / 2
//		frA[repIdx] = fgA[repIdx] = fbA[repIdx] = frB[repIdx] = fgB[repIdx] = fbB[repIdx] = 0.0f;
//		for (i = 0; i < SVD_MAT_HEIGHT; i++) {
//			frA[repIdx] += InverseA[j][i] * red[i];
//			fgA[repIdx] += InverseA[j][i] * green[i];
//			fbA[repIdx] += InverseA[j][i] * blue[i];
//			frB[repIdx] += InverseA[j+1][i] * red[i];
//			fgB[repIdx] += InverseA[j+1][i] * green[i];
//			fbB[repIdx] += InverseA[j+1][i] * blue[i];
//		}
//	}

    // compute optimized A and B values
	float frA[4], fgA[4], fbA[4], frB[4], fgB[4], fbB[4];
    float tempR[SVD_MAT_WIDTH], tempG[SVD_MAT_WIDTH], tempB[SVD_MAT_WIDTH];
	int outRed, outGreen, outBlue, repIdx;
    for (i = 0; i < SVD_MAT_WIDTH; i++) {
        tempR[i] = 0.0f;
        tempG[i] = 0.0f;
        tempB[i] = 0.0f;
        for (k = 0; k < SVD_MAT_HEIGHT; k++) {
            tempR[i] += A[k][i] * red[k];
            tempG[i] += A[k][i] * green[k];
            tempB[i] += A[k][i] * blue[k];
        }
    }
	
    for (j = 0; j < SVD_MAT_WIDTH; j+=2) {
		repIdx = (j>>1);	// repIdx = j / 2
		frA[repIdx] = fgA[repIdx] = fbA[repIdx] = frB[repIdx] = fgB[repIdx] = fbB[repIdx] = 0.0f;
		for (i = 0; i < SVD_MAT_WIDTH; i++) {
			frA[repIdx] += Eigenbasis[j][i] * tempR[i];
			fgA[repIdx] += Eigenbasis[j][i] * tempG[i];
			fbA[repIdx] += Eigenbasis[j][i] * tempB[i];
			frB[repIdx] += Eigenbasis[j+1][i] * tempR[i];
			fgB[repIdx] += Eigenbasis[j+1][i] * tempG[i];
			fbB[repIdx] += Eigenbasis[j+1][i] * tempB[i];
		}
	}

#ifdef USE_PIXEL_UPDATE
	// Calculate new representative colors by using a "fix the error" approach.
	// We multiply the pseudo-inverse weight matrix by the difference between the
	// current and original images to get an "update" that is applied to each
	// representative value. We also limit how big the update can be to avoid
	// colors flying out of bounds.
	unsigned int oldColor;
	// Do top left for 2x2 case or only one for 1x1 case
	tIdx = (width * thready) + threadx;
	oldColor = oldA[tIdx];
	outRed = CLAMP((int)frA[0], -16, 16);
	outGreen = CLAMP((int)fgA[0], -16, 16);
	outBlue = CLAMP((int)fbA[0], -16, 16);
	candidateA[tIdx] = MAKE_ARGB(
		CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
		CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
		CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldA[tIdx] = candidateA[tIdx];

	oldColor = oldB[tIdx];
	outRed = CLAMP((int)frB[0], -16, 16);
	outGreen = CLAMP((int)fgB[0], -16, 16);
	outBlue = CLAMP((int)fbB[0], -16, 16);
	candidateB[tIdx] = MAKE_ARGB(
		CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
		CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
		CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldB[tIdx] = candidateB[tIdx];
#ifdef TWO_BY_TWO
	// Top right case for 2x2
	tIdx = (width * thready) + threadx + 1;
	oldColor = oldA[tIdx];
	outRed = CLAMP((int)frA[1], -16, 16);
	outGreen = CLAMP((int)fgA[1], -16, 16);
	outBlue = CLAMP((int)fbA[1], -16, 16);
	candidateA[tIdx] = MAKE_ARGB(
		CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
		CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
		CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldA[tIdx] = candidateA[tIdx];

	oldColor = oldB[tIdx];
	outRed = CLAMP((int)frB[1], -16, 16);
	outGreen = CLAMP((int)fgB[1], -16, 16);
	outBlue = CLAMP((int)fbB[1], -16, 16);
	candidateB[tIdx] = MAKE_ARGB(
		CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
		CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
		CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldB[tIdx] = candidateB[tIdx];

	// Bottom left case
	tIdx = (width * (thready+1)) + threadx;
	oldColor = oldA[tIdx];
	outRed = CLAMP((int)frA[2], -16, 16);
	outGreen = CLAMP((int)fgA[2], -16, 16);
	outBlue = CLAMP((int)fbA[2], -16, 16);
	candidateA[tIdx] = MAKE_ARGB(
		CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
		CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
		CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldA[tIdx] = candidateA[tIdx];

	oldColor = oldB[tIdx];
	outRed = CLAMP((int)frB[2], -16, 16);
	outGreen = CLAMP((int)fgB[2], -16, 16);
	outBlue = CLAMP((int)fbB[2], -16, 16);
	candidateB[tIdx] = MAKE_ARGB(
		CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
		CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
		CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldB[tIdx] = candidateB[tIdx];

	// Bottom right case
	tIdx = (width * (thready+1)) + threadx + 1;
	oldColor = oldA[tIdx];
	outRed = CLAMP((int)frA[3], -16, 16);
	outGreen = CLAMP((int)fgA[3], -16, 16);
	outBlue = CLAMP((int)fbA[3], -16, 16);
	candidateA[tIdx] = MAKE_ARGB(
		CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
		CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
		CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldA[tIdx] = candidateA[tIdx];

	oldColor = oldB[tIdx];
	outRed = CLAMP((int)frB[3], -16, 16);
	outGreen = CLAMP((int)fgB[3], -16, 16);
	outBlue = CLAMP((int)fbB[3], -16, 16);
	candidateB[tIdx] = MAKE_ARGB(
		CLAMP((int)MAKE_RED_PIXEL(oldColor) + outRed, 0, 255),
		CLAMP((int)MAKE_GREEN_PIXEL(oldColor) + outGreen, 0, 255),
		CLAMP((int)MAKE_BLUE_PIXEL(oldColor) + outBlue, 0, 255));
	oldB[tIdx] = candidateB[tIdx];
#endif // TWO_BY_TWO
#else
	// WARNING - THIS DOES NOT WORK
	// Calculate new representative colors by computing completely new colors.
	// We multiply the pseudo-inverse weight matrix by the actual colors of 
	// the original pixels in our window to get new representative colors.
	// do A candidate. If the new computed color is out of bounds, we discard
	// it and use the old one.
	// Top left case
	if ((frA[0] > 255) || (fgA[0] > 255) || (fbA[0] > 255) || (frA[0] < 0) || (fgA[0] < 0) || (fbA[0] < 0)) {
		outRed = MAKE_RED_PIXEL(oldA[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldA[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldA[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frA[0]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgA[0]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbA[0]), 0, 255);
	}
	candidateA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);

	// do B candidate
	if ((frB[0] > 255) || (fgB[0] > 255) || (fbB[0] > 255) || (frB[0] < 0) || (fgB[0] < 0) || (fbB[0] < 0)) {
		outRed = MAKE_RED_PIXEL(oldB[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldB[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldB[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frB[0]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgB[0]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbB[0]), 0, 255);
	}
	candidateB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
#ifdef TWO_BY_TWO
	// Top right case
	tIdx = (width * (thready)) + threadx + 1;
	if ((frA[1] > 255) || (fgA[1] > 255) || (fbA[1] > 255) || (frA[1] < 0) || (fgA[1] < 0) || (fbA[1] < 0)) {
		outRed = MAKE_RED_PIXEL(oldA[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldA[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldA[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frA[1]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgA[1]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbA[1]), 0, 255);
	}
	candidateA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);

	// do B candidate
	if ((frB[1] > 255) || (fgB[1] > 255) || (fbB[1] > 255) || (frB[1] < 0) || (fgB[1] < 0) || (fbB[1] < 0)) {
		outRed = MAKE_RED_PIXEL(oldB[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldB[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldB[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frB[1]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgB[1]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbB[1]), 0, 255);
	}
	candidateB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);

	// Bottom left case
	tIdx = (width * (thready+1)) + threadx;
	if ((frA[2] > 255) || (fgA[2] > 255) || (fbA[2] > 255) || (frA[2] < 0) || (fgA[2] < 0) || (fbA[2] < 0)) {
		outRed = MAKE_RED_PIXEL(oldA[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldA[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldA[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frA[2]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgA[2]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbA[2]), 0, 255);
	}
	candidateA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);

	// do B candidate
	if ((frB[2] > 255) || (fgB[2] > 255) || (fbB[2] > 255) || (frB[2] < 0) || (fgB[2] < 0) || (fbB[2] < 0)) {
		outRed = MAKE_RED_PIXEL(oldB[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldB[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldB[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frB[2]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgB[2]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbB[2]), 0, 255);
	}
	candidateB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);

	// Bottom right case
	tIdx = (width * (thready+1)) + threadx + 1;
	if ((frA[3] > 255) || (fgA[3] > 255) || (fbA[3] > 255) || (frA[3] < 0) || (fgA[3] < 0) || (fbA[3] < 0)) {
		outRed = MAKE_RED_PIXEL(oldA[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldA[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldA[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frA[3]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgA[3]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbA[3]), 0, 255);
	}
	candidateA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldA[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);

	// do B candidate
	if ((frB[3] > 255) || (fgB[3] > 255) || (fbB[3] > 255) || (frB[3] < 0) || (fgB[3] < 0) || (fbB[3] < 0)) {
		outRed = MAKE_RED_PIXEL(oldB[tIdx]);
		outGreen = MAKE_GREEN_PIXEL(oldB[tIdx]);
		outBlue = MAKE_BLUE_PIXEL(oldB[tIdx]);
	} else {
		outRed = CLAMP(__float2uint_rn(frB[3]), 0, 255);
		outGreen = CLAMP(__float2uint_rn(fgB[3]), 0, 255);
		outBlue = CLAMP(__float2uint_rn(fbB[3]), 0, 255);
	}
	candidateB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
	oldB[tIdx] = MAKE_ARGB(outRed, outGreen, outBlue);
#endif // TWO_BY_TWO
#endif // USE_PIXEL_UPDATE
}
#endif // USE_SVD
//////////////////////////////////////////////////////////////////////////////////
//								  Optimization									//
//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
//								 Modulation Bits								//
//////////////////////////////////////////////////////////////////////////////////
#ifdef TWO_BPP  
__global__ void compute_modulation_mode(unsigned int *a, unsigned int *b,
                                        unsigned int *mode, float *modbits,
                                        int *red, int *green, int *blue, int width) {
    unsigned int thready = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int threadx = blockDim.x * blockIdx.x + threadIdx.x;
    
    unsigned int x = 8 * threadx;
    unsigned int y = 4 * thready;
    
    float mode0_err = 0.0f, mode1_err = 0.0f;
    
    int i, j;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 8; j++) {
            unsigned int idx = (y+i)*width + (x+j);
            unsigned int apixel = a[idx];
            int a_red = MAKE_RED_PIXEL(apixel);
            int a_green = MAKE_GREEN_PIXEL(apixel);
            int a_blue = MAKE_BLUE_PIXEL(apixel);
            
            unsigned int bpixel = b[idx];
            int b_red = MAKE_RED_PIXEL(bpixel);
            int b_green = MAKE_GREEN_PIXEL(bpixel);
            int b_blue = MAKE_BLUE_PIXEL(bpixel);
            
            unsigned int opixel = tex2D(origRef, x + j, y + i);
            int o_red = MAKE_RED_PIXEL(opixel);
            int o_green = MAKE_GREEN_PIXEL(opixel);
            int o_blue = MAKE_BLUE_PIXEL(opixel);
            
            float mod = modbits[idx];
            float r_mod = 1.0f - mod;
            mode1_err += SQR(o_red - ((float)b_red*mod + (float)a_red*r_mod));
            mode1_err += SQR(o_green - ((float)b_green*mod + (float)a_green*r_mod));
            mode1_err += SQR(o_blue - ((float)b_blue*mod + (float)a_blue*r_mod));
            
            if (mod < 0.5f) {
                mod = 0.0f;
            } else {
                mod = 1.0f;
            }
            r_mod = 1.0f - mod;
            mode0_err += SQR(o_red - ((float)b_red*mod + (float)a_red*r_mod));
            mode0_err += SQR(o_green - ((float)b_green*mod + (float)a_green*r_mod));
            mode0_err += SQR(o_blue - ((float)b_blue*mod + (float)a_blue*r_mod));
        }
    }
    
    if (mode0_err < mode1_err) {
        mode[thready*width + threadx] = 0;
        
        for (i = 0; i < 4; i++) {
            for (j = 0; j < 8; j++) {
                unsigned int idx = (y+i)*width + (x+j);
                if (modbits[idx] < 0.5f) {
                    modbits[idx] = 0.0f;
                } else {
                    modbits[idx] = 1.0f;
                }
                
                unsigned int apixel = a[idx];
                int a_red = MAKE_RED_PIXEL(apixel);
                int a_green = MAKE_GREEN_PIXEL(apixel);
                int a_blue = MAKE_BLUE_PIXEL(apixel);
                
                unsigned int bpixel = b[idx];
                int b_red = MAKE_RED_PIXEL(bpixel);
                int b_green = MAKE_GREEN_PIXEL(bpixel);
                int b_blue = MAKE_BLUE_PIXEL(bpixel);
                
                unsigned int opixel = tex2D(origRef, x + j, y + i);
                int o_red = MAKE_RED_PIXEL(opixel);
                int o_green = MAKE_GREEN_PIXEL(opixel);
                int o_blue = MAKE_BLUE_PIXEL(opixel);
                float mod = modbits[idx];
                float r_mod = 1.0f - mod;
                red[idx] += o_red - ((float)b_red*mod + (float)a_red*r_mod);
                green[idx] += o_green - ((float)b_green*mod + (float)a_green*r_mod);
                blue[idx] += o_blue - ((float)b_blue*mod + (float)a_blue*r_mod);
            }
        }
    } else {
        mode[thready*width + threadx] = 1;
    }
}
#endif // TWO_BPP
// compute modulation bits based on reconstruction of original image
__global__ void compute_modulation_bits(unsigned int *a, unsigned int *b,
										int *redCurrent, int *greenCurrent, 
										int *blueCurrent, float *mod, int width) {
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = (width * y) + x;

	unsigned int apixel = a[idx];
	int a_red = MAKE_RED_PIXEL(apixel);
	int a_green = MAKE_GREEN_PIXEL(apixel);
	int a_blue = MAKE_BLUE_PIXEL(apixel);

	unsigned int bpixel = b[idx];
	int b_red = MAKE_RED_PIXEL(bpixel);
	int b_green = MAKE_GREEN_PIXEL(bpixel);
	int b_blue = MAKE_BLUE_PIXEL(bpixel);

	unsigned int opixel = tex2D(origRef, x, y);
	int o_red = MAKE_RED_PIXEL(opixel);
	int o_green = MAKE_GREEN_PIXEL(opixel);
	int o_blue = MAKE_BLUE_PIXEL(opixel);

	int3 deltaA = make_int3(o_red - a_red, o_green - a_green, o_blue - a_blue);
	int3 deltaB = make_int3(o_red - b_red, o_green - b_green, o_blue - b_blue);
	int3 delta38 = make_int3(o_red - (FIVE_EIGHTHS*a_red + THREE_EIGHTHS*b_red),
		o_green - (FIVE_EIGHTHS*a_green + THREE_EIGHTHS*b_green),
		o_blue - (FIVE_EIGHTHS*a_blue + THREE_EIGHTHS*b_blue));
	int3 delta58 = make_int3(o_red - (THREE_EIGHTHS*a_red + FIVE_EIGHTHS*b_red),
		o_green - (THREE_EIGHTHS*a_green + FIVE_EIGHTHS*b_green),
		o_blue - (THREE_EIGHTHS*a_blue + FIVE_EIGHTHS*b_blue));

	int dotA = SQR(deltaA.x) + SQR(deltaA.y) + SQR(deltaA.z);
	int dotB = SQR(deltaB.x) + SQR(deltaB.y) + SQR(deltaB.z);
	int dot38 = SQR(delta38.x) + SQR(delta38.y) + SQR(delta38.z);
	int dot58 = SQR(delta58.x) + SQR(delta58.y) + SQR(delta58.z);
	int dotMin = min(min(dotA, dotB), min(dot38, dot58));
	float modbit;

	if (dotMin == dotA) {
		modbit = 0.0f;
	} else if (dotMin == dot38) {
		modbit = THREE_EIGHTHS;
	} else if (dotMin == dot58) {
		modbit = FIVE_EIGHTHS;
	} else {
		modbit = 1.0f;
	}
	int3 diff = make_int3(b_red - a_red, b_green - a_green, b_blue - a_blue);

	float dot = (float)(deltaA.x*diff.x + deltaA.y*diff.y + deltaA.z*diff.z) /
				(float)(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);

	/*if (dot < 0.1875f) {
		modbit = 0.000005f;
	} else if (dot < 0.5f) {
		modbit = THREE_EIGHTHS;
	} else if (dot < 0.8125f) {
		modbit = FIVE_EIGHTHS;
	} else {
		modbit = 0.999995f;
	}*/
#ifdef USE_PIXEL_UPDATE
	float r_modbit = 1.0f - modbit;
	int c_red = ((float)b_red*modbit + (float)a_red*r_modbit);
	int c_green = ((float)b_green*modbit + (float)a_green*r_modbit);
	int c_blue = ((float)b_blue*modbit + (float)a_blue*r_modbit);

	redCurrent[idx] = o_red - c_red;
	greenCurrent[idx] = o_green - c_green;
	blueCurrent[idx] = o_blue - c_blue;
#else
	// WARNING - THIS DOES NOT WORK
	redCurrent[idx] = o_red;
	greenCurrent[idx] = o_green;
	blueCurrent[idx] = o_blue;
#endif
	mod[idx] = modbit;
}
//////////////////////////////////////////////////////////////////////////////////
//								 Modulation Bits								//
//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
//								A and B prototypes								//
//////////////////////////////////////////////////////////////////////////////////
__global__ void make_a_b_prototypes(unsigned int *a_proto, unsigned int *b_proto,
									int width, int height) {
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = (width * y) + x;

	// get original, delta, and axis pixel
	int opixel = tex2D(origRef, x, y);
	int fpixel = tex2D(texRef, x, y);

	int orig_red = MAKE_RED_PIXEL(opixel);
	int orig_green = MAKE_GREEN_PIXEL(opixel);
	int orig_blue = MAKE_BLUE_PIXEL(opixel);
	int delta_red = orig_red - MAKE_RED_PIXEL(fpixel);
	int delta_green = orig_green - MAKE_GREEN_PIXEL(fpixel);
	int delta_blue = orig_blue - MAKE_BLUE_PIXEL(fpixel);
	int axis_red = CLAMP(abs(delta_red), 0, 255);
	int axis_green = CLAMP(abs(delta_green), 0, 255);
	int axis_blue = CLAMP(abs(delta_blue), 0, 255);

	// (delta dot axis) / (axis dot axis)
	int dot = ((delta_red * axis_red) + (delta_green * axis_green) + 
			  (delta_blue * axis_blue)) / ((axis_red * axis_red) + 
			  (axis_green * axis_green) + (axis_blue * axis_blue));

	/*if (dot < 0) {
		a_proto[idx] = MAKE_ARGB((CLAMP(orig_red + axis_red * dot, 0, 255)), 
								 (CLAMP(orig_green + axis_green * dot, 0, 255)), 
								 (CLAMP(orig_blue + axis_blue * dot, 0, 255)));

		b_proto[idx] = MAKE_ARGB((CLAMP(orig_red - axis_red * dot, 0, 255)), 
								 (CLAMP(orig_green - axis_green * dot, 0, 255)), 
								 (CLAMP(orig_blue - axis_blue * dot, 0, 255)));
	} else {
		b_proto[idx] = MAKE_ARGB((CLAMP(orig_red + axis_red * dot, 0, 255)), 
								 (CLAMP(orig_green + axis_green * dot, 0, 255)), 
								 (CLAMP(orig_blue + axis_blue * dot, 0, 255)));

		a_proto[idx] = MAKE_ARGB((CLAMP(orig_red - axis_red * dot, 0, 255)), 
								 (CLAMP(orig_green - axis_green * dot, 0, 255)), 
								 (CLAMP(orig_blue - axis_blue * dot, 0, 255)));
	}*/
	b_proto[idx] = MAKE_ARGB((CLAMP(orig_red + axis_red, 0, 255)), 
		(CLAMP(orig_green + axis_green, 0, 255)), 
		(CLAMP(orig_blue + axis_blue, 0, 255)));

	a_proto[idx] = MAKE_ARGB((CLAMP(orig_red - axis_red, 0, 255)), 
		(CLAMP(orig_green - axis_green, 0, 255)), 
		(CLAMP(orig_blue - axis_blue, 0, 255)));
}
//////////////////////////////////////////////////////////////////////////////////
//								A and B prototypes								//
//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
//								bilinear upscaling								//
//////////////////////////////////////////////////////////////////////////////////

// (1-alpha)*A + alpha*B. alpha must be between 0~1.
__device__ float lerp(const float A, const float B, float alpha)
{
	return (((1-alpha)*A) + (alpha*B));
}

__global__ void bilinear_resize4x4(unsigned int *out, int width, int height, bool flag) {
	int x, x1, y, y1;	// x1 and y1 for texture wrapping purposes
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int a, b, c, d;
	float x_diff, y_diff, blue, red, green;

	// do the 4x4 bilinear scaling
	if (flag == true) { 
		x = CLAMP((j-2), 0, width-1)>>2;
		y = CLAMP((i-2), 0, height-1)>>2;
		x1 = x+1;
		y1 = y+1;
		x_diff = (CLAMP(j-2, 0, width) - (x*4)) * ONE_FOURTH; // j%4 = j - (x*4)
		y_diff = (CLAMP(i-2, 0, height) - (y*4)) * ONE_FOURTH;
	} else {
		x = (j>>2);
		y = (i>>2);
		x1 = CLAMP(x+1, 0, (width>>2)-1);
		y1 = CLAMP(y+1, 0, (width>>2)-1);
		x_diff = (j - (x*4)) * ONE_FOURTH; // j%4 = j - (x*4)
		y_diff = (i - (y*4)) * ONE_FOURTH;
	}

	// fetch the 32bit ARGB unsigned ints for the four pixels
	a = tex2D(texRef, x, y);
	b = tex2D(texRef, x1, y);
	c = tex2D(texRef, x, y1);
	d = tex2D(texRef, x1, y1);

	// blue channel
	blue = lerp(lerp((float)MAKE_BLUE_PIXEL(a), (float)MAKE_BLUE_PIXEL(b), x_diff),
		lerp((float)MAKE_BLUE_PIXEL(c), (float)MAKE_BLUE_PIXEL(d), x_diff),
		y_diff);

	// green channel
	green = lerp(lerp((float)MAKE_GREEN_PIXEL(a), (float)MAKE_GREEN_PIXEL(b), x_diff),
		lerp((float)MAKE_GREEN_PIXEL(c), (float)MAKE_GREEN_PIXEL(d), x_diff),
		y_diff);

	// red channel
	red = lerp(lerp((float)MAKE_RED_PIXEL(a), (float)MAKE_RED_PIXEL(b), x_diff),
		lerp((float)MAKE_RED_PIXEL(c), (float)MAKE_RED_PIXEL(d), x_diff),
		y_diff);

	// store the result
	out[i*width + j] = MAKE_ARGB( CLAMP(__float2uint_rn(red), 0, 255),
		CLAMP(__float2uint_rn(green), 0, 255),
		CLAMP(__float2uint_rn(blue), 0, 255) );
}

__global__ void bilinear_resize8x4(unsigned int *out, int width, int height, bool flag) {
	int x, x1, y, y1;	// x1 and y1 for texture wrapping purposes
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int a, b, c, d;
	float x_diff, y_diff, blue, red, green;

	// do the 8x4 bilinear scaling
	if (flag == true) {
		x = CLAMP((j-4), 0, width-1)>>3;
		y = CLAMP((i-2), 0, height-1)>>2;
		x1 = x+1;
		y1 = y+1;
		x_diff = (CLAMP(j-4, 0, width) - (x*8)) * ONE_EIGHTH; // j%4 = j - (x*4)
		y_diff = (CLAMP(i-2, 0, height) - (y*4)) * ONE_FOURTH;
	} else {
		x = (j>>3);
		y = (i>>2);
		x1 = CLAMP(x+1, 0, (width>>3)-1);
		y1 = CLAMP(y+1, 0, (width>>2)-1);
		x_diff = (j - (x*8)) * ONE_EIGHTH; // j%4 = j - (x*4)
		y_diff = (i - (y*4)) * ONE_FOURTH;
	}
//	x = j>>3;		// (j / 8)
//	y = i>>2;		// (i / 4)
//	x1 = (x+1) & ((width>>3) - 1);	// (x+1) % (width / 8)
//	y1 = (y+1) & ((height>>2) - 1);	// (y+1) % (height / 4)
//	x_diff = (j - (x*8)) * ONE_EIGHTH;	// j%8 = j - (x*8)
//	y_diff = (i - (y*4)) * ONE_FOURTH;	// i%4 = j - (y*4)

	// fetch the 32bit ARGB unsigned ints for the four pixels
	a = tex2D(texRef, x, y);
	b = tex2D(texRef, x1, y);
	c = tex2D(texRef, x, y1);
	d = tex2D(texRef, x1, y1);

	// blue channel
	blue = lerp(lerp((float)MAKE_BLUE_PIXEL(a), (float)MAKE_BLUE_PIXEL(b), x_diff),
		lerp((float)MAKE_BLUE_PIXEL(c), (float)MAKE_BLUE_PIXEL(d), x_diff),
		y_diff);

	// green channel
	green = lerp(lerp((float)MAKE_GREEN_PIXEL(a), (float)MAKE_GREEN_PIXEL(b), x_diff),
		lerp((float)MAKE_GREEN_PIXEL(c), (float)MAKE_GREEN_PIXEL(d), x_diff),
		y_diff);

	// red channel
	red = lerp(lerp((float)MAKE_RED_PIXEL(a), (float)MAKE_RED_PIXEL(b), x_diff),
		lerp((float)MAKE_RED_PIXEL(c), (float)MAKE_RED_PIXEL(d), x_diff),
		y_diff);

	// store the result
	out[i*width + j] = MAKE_ARGB(CLAMP(__float2uint_rn(red), 0, 255),
                                 CLAMP(__float2uint_rn(green), 0, 255),
                                 CLAMP(__float2uint_rn(blue), 0, 255));
}
//////////////////////////////////////////////////////////////////////////////////
//								 bilinear upscaling								//
//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
//							low-pass wavelet filter								//
//////////////////////////////////////////////////////////////////////////////////
__global__ void linear_wavelet_transform_rows(int width, int height, 
											  unsigned int* out) {
	int n; 
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int pixel;
	float filter;

	float3 s = make_float3(0.0f, 0.0f, 0.0f);

	// we start out negative because we want to center the filter on
	// the current pixel
	for (int m = 0 - HALF_FILTER_LENGTH; m < HALF_FILTER_LENGTH; m++) {
		n = 2 * k + m;
		if (n < 0) {
			n = 0 - n;
		}
		if (n >= width) {
			n -= 2 * (1 + n - width);
		}
		pixel = tex2D(texRef, CLAMP(n, 0, width-1), CLAMP(y, 0, height-1));
		filter = tex1Dfetch(filterRef, (int)(m + HALF_FILTER_LENGTH));

		s.x += filter * __uint2float_rn(MAKE_RED_PIXEL(pixel));
		s.y += filter * __uint2float_rn(MAKE_GREEN_PIXEL(pixel));
		s.z += filter * __uint2float_rn(MAKE_BLUE_PIXEL(pixel));
	}

	out[y*width + k] = MAKE_ARGB(CLAMP(__float2uint_rn(s.x), 0, 255),
								 CLAMP(__float2uint_rn(s.y), 0, 255),
								 CLAMP(__float2uint_rn(s.z), 0, 255));
}

__global__ void linear_wavelet_transform_cols(int width, int height, 
											  unsigned int* out) {
	int n; 
	int k = blockDim.y * blockIdx.y + threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int pixel;
	float filter;

	float3 s = make_float3(0.0f, 0.0f, 0.0f);

	// we start out negative because we want to center the filter on
	// the current pixel
	for (int m = 0 - HALF_FILTER_LENGTH; m < HALF_FILTER_LENGTH; m++) {
		n = 2 * k + m;
		if (n < 0) {
			n = 0 - n;
		}
		if (n >= height) {
			n -= 2 * (1 + n - height);
		}
		pixel = tex2D(texRef, CLAMP(x, 0, width-1), CLAMP(n, 0, height-1));
		filter = tex1Dfetch(filterRef, (int)(m + HALF_FILTER_LENGTH));

		s.x += filter * __uint2float_rn(MAKE_RED_PIXEL(pixel));
		s.y += filter * __uint2float_rn(MAKE_GREEN_PIXEL(pixel));
		s.z += filter * __uint2float_rn(MAKE_BLUE_PIXEL(pixel));
	}

	out[k*width + x] = MAKE_ARGB(CLAMP(__float2uint_rn(s.x), 0, 255),
								 CLAMP(__float2uint_rn(s.y), 0, 255),
								 CLAMP(__float2uint_rn(s.z), 0, 255));
}

//////////////////////////////////////////////////////////////////////////////////
//	Calculates the low-pass wavelet transform of the input data and stores the	//
//	result in the same location. First transforms the rows and then the colums	//
//	of the input image.															//
//																				//
//	wavelet		Initial input data. The result is also stored here				//
//	temp		Pointer to scratch space in memory that is the same size as the	//
//				input data														//
//	num			The number of times the low-pass filter is applied				//
//																				//
//////////////////////////////////////////////////////////////////////////////////
void linear_wavelet_transform(unsigned int * wavelet, unsigned int *temp, int num,
							  textureReference *texRefPtr, int width, int height, 
							  int scan_width, cudaChannelFormatDesc *channelDesc,
							  dim3 default_block){
	size_t offset;
	int i;
	for (i = 1; i <= num; i++) {
		// bind the input data to the texture
		cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)wavelet, 
			channelDesc, width, height, scan_width));

		// calulate the grid dimensions and call the kernel
		dim3 grid0(width / (BLOCK_WIDTH * 2 * i), height / (BLOCK_HEIGHT * i));
		linear_wavelet_transform_rows<<<grid0, default_block>>> (width, height, temp);

		// bind the new data to the texture
		cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)temp, 
			channelDesc, width, height, scan_width));
		
		// calulate the grid dimensions and call the kernel
		dim3 grid1(width / (BLOCK_WIDTH * i), height / (BLOCK_HEIGHT * 2 * i));
		linear_wavelet_transform_cols<<<grid1, default_block>>> (width, height, wavelet);
	}
}
//////////////////////////////////////////////////////////////////////////////////
//							low-pass wavelet filter								//
//////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////
//	External method to call CUDA kernels for compression. Places the compressed	//
//	data in	h_out.																//
//																				//
//	width		Width of the input image										//
//	height		Height of the input image										//
//	scan_width	Size in bytes of a single row of pixels							//
//	h_in		Pointer to the input buffer on the host							//
//	h_out		Pointer to the output buffer on the	host						//
//																				//
//////////////////////////////////////////////////////////////////////////////////
extern "C" int cuda_pvr_compress(int width, int height, int scan_width, 
								  unsigned char* h_in, unsigned char* h_out) {
	// image dimensions must be powers of 2
	if ((width & (width - 1)) != 0 || (height & (height - 1)) != 0) {
		printf("Error: Image dimensions must be powers of 2.\n");
		printf("Aborting compression...\n");

		return 1;
	}

	// declare device memory pointers and block dimensions
	unsigned int *d_bits,*d_out,*d_temp,*d_wavelet,*d_axis,*d_aproto,*d_bproto;
	int *d_redCurrent, *d_greenCurrent, *d_blueCurrent, *d_err;
	float *d_filter, *d_mod;
#ifdef TWO_BY_TWO
    float *d_svdMatrixTL, *d_svdMatrixTR, *d_svdMatrixBL, *d_svdMatrixBR;
#else
    float *d_svdMatrix;
#endif // TWO_BY_TWO
	int i, tex_size = height * scan_width;
	dim3 default_block(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 default_grid(width / (BLOCK_WIDTH), height / (BLOCK_HEIGHT));
	dim3 quarterGrid(default_grid.x>>2, default_grid.y>>2);
    dim3 svdGrid(default_grid.x * SVD_FACTOR_X, default_grid.y * SVD_FACTOR_Y);
	size_t offset;

	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////

	// initialize cuda and allocate device memory
	CUT_DEVICE_INIT(1, "");
	cutilSafeCall(cudaMalloc((void **)&d_bits, tex_size));
	cutilSafeCall(cudaMalloc((void **)&d_temp, tex_size));
	cutilSafeCall(cudaMalloc((void **)&d_wavelet, tex_size));
	cutilSafeCall(cudaMalloc((void **)&d_axis, tex_size));
	cutilSafeCall(cudaMalloc((void **)&d_aproto, tex_size));
	cutilSafeCall(cudaMalloc((void **)&d_bproto, tex_size));
	cutilSafeCall(cudaMalloc((void **)&d_redCurrent, width * height * sizeof(int)));
	cutilSafeCall(cudaMalloc((void **)&d_greenCurrent, width * height * sizeof(int)));
	cutilSafeCall(cudaMalloc((void **)&d_blueCurrent, width * height * sizeof(int)));
	cutilSafeCall(cudaMalloc((void **)&d_err, sizeof(int)));
	cutilSafeCall(cudaMalloc((void **)&d_filter, FILTER_LENGTH * sizeof(float)));
	cutilSafeCall(cudaMalloc((void **)&d_mod, width * height * sizeof(float)));
#ifdef TWO_BPP
    cutilSafeCall(cudaMalloc((void **)&d_out, 2 * (width>>3) * (height>>2) * 
                             sizeof(unsigned int)));
#else
	cutilSafeCall(cudaMalloc((void **)&d_out, 2 * (width>>2) * (height>>2) * 
		sizeof(unsigned int)));
#endif // TWO_BPP

	cutilSafeCall(cudaMemcpy(d_bits, h_in, tex_size, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_wavelet, d_bits, tex_size, cudaMemcpyDeviceToDevice));
	cutilSafeCall(cudaMemcpy(d_filter, wavelet_filter, FILTER_LENGTH * sizeof(float), 
							 cudaMemcpyHostToDevice));
#ifdef TWO_BY_TWO
    cutilSafeCall(cudaMalloc((void **)&d_svdMatrixTL, SVD_MAT_HEIGHT * sizeof(float)));
    cutilSafeCall(cudaMalloc((void **)&d_svdMatrixTR, SVD_MAT_HEIGHT * sizeof(float)));
    cutilSafeCall(cudaMalloc((void **)&d_svdMatrixBL, SVD_MAT_HEIGHT * sizeof(float)));
    cutilSafeCall(cudaMalloc((void **)&d_svdMatrixBR, SVD_MAT_HEIGHT * sizeof(float)));
    cutilSafeCall(cudaMemcpy(d_svdMatrixTL, MwTL, SVD_MAT_HEIGHT * sizeof(float), 
							 cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_svdMatrixTR, MwTR, SVD_MAT_HEIGHT * sizeof(float), 
							 cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_svdMatrixBL, MwBL, SVD_MAT_HEIGHT * sizeof(float), 
							 cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(d_svdMatrixBR, MwBR, SVD_MAT_HEIGHT * sizeof(float), 
							 cudaMemcpyHostToDevice));
#else
    cutilSafeCall(cudaMalloc((void **)&d_svdMatrix, SVD_MAT_HEIGHT * sizeof(float)));
	cutilSafeCall(cudaMemcpy(d_svdMatrix, Mw, SVD_MAT_HEIGHT * sizeof(float), 
							 cudaMemcpyHostToDevice));
#endif // TWO_BY_TWO

	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////

	// map the input buffer as texture memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned int>();
	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc channelDesc3 = cudaCreateChannelDesc<int>();

	textureReference *texRefPtr,*origRefPtr,*filterRefPtr,*modRefPtr,
		*redRefPtr,*greenRefPtr,*blueRefPtr; 

	cudaGetTextureReference((const textureReference**)&texRefPtr, "texRef");
	cudaGetTextureReference((const textureReference**)&origRefPtr, "origRef");
	cudaGetTextureReference((const textureReference**)&filterRefPtr, "filterRef");
	cudaGetTextureReference((const textureReference**)&modRefPtr, "modRef");
	cudaGetTextureReference((const textureReference**)&redRefPtr, "redCurrentRef");
	cudaGetTextureReference((const textureReference**)&greenRefPtr, "greenCurrentRef");
	cudaGetTextureReference((const textureReference**)&blueRefPtr, "blueCurrentRef");
    
#ifdef TWO_BY_TWO
    textureReference *svdRefPtrTL, *svdRefPtrTR, *svdRefPtrBL, *svdRefPtrBR;
    cudaGetTextureReference((const textureReference**)&svdRefPtrTL, "svdMatRefTL");
    cudaGetTextureReference((const textureReference**)&svdRefPtrTR, "svdMatRefTR");
    cudaGetTextureReference((const textureReference**)&svdRefPtrBL, "svdMatRefBL");
    cudaGetTextureReference((const textureReference**)&svdRefPtrBR, "svdMatRefBR");
#else
    textureReference *svdRefPtr;
    cudaGetTextureReference((const textureReference**)&svdRefPtr, "svdMatRef");
#endif // TWO_BY_TWO

	// set the addressing and filter modes for all 2D textures
	texRefPtr->addressMode[0] = cudaAddressModeClamp;
	texRefPtr->addressMode[1] = cudaAddressModeClamp;
	texRefPtr->addressMode[2] = cudaAddressModeClamp;
	texRefPtr->filterMode = cudaFilterModePoint;

	origRefPtr->addressMode[0] = cudaAddressModeClamp;
	origRefPtr->addressMode[1] = cudaAddressModeClamp;
	origRefPtr->addressMode[2] = cudaAddressModeClamp;
	origRefPtr->filterMode = cudaFilterModePoint;

	modRefPtr->addressMode[0] = cudaAddressModeClamp;
	modRefPtr->addressMode[1] = cudaAddressModeClamp;
	modRefPtr->addressMode[2] = cudaAddressModeClamp;
	modRefPtr->filterMode = cudaFilterModePoint;

	redRefPtr->addressMode[0] = cudaAddressModeClamp;
	redRefPtr->addressMode[1] = cudaAddressModeClamp;
	redRefPtr->addressMode[2] = cudaAddressModeClamp;
	redRefPtr->filterMode = cudaFilterModePoint;

	blueRefPtr->addressMode[0] = cudaAddressModeClamp;
	blueRefPtr->addressMode[1] = cudaAddressModeClamp;
	blueRefPtr->addressMode[2] = cudaAddressModeClamp;
	blueRefPtr->filterMode = cudaFilterModePoint;

	greenRefPtr->addressMode[0] = cudaAddressModeClamp;
	greenRefPtr->addressMode[1] = cudaAddressModeClamp;
	greenRefPtr->addressMode[2] = cudaAddressModeClamp;
	greenRefPtr->filterMode = cudaFilterModePoint;

	// bind textures
	cutilSafeCall(cudaBindTexture2D(&offset, origRefPtr, (const void*)d_bits, 
		&channelDesc, width, height, scan_width));
	cutilSafeCall(cudaBindTexture(&offset, filterRefPtr, (const void*)d_filter, 
		&channelDesc2, FILTER_LENGTH * sizeof(float)));
	cutilSafeCall(cudaBindTexture2D(&offset, modRefPtr, (const void*)d_mod, 
		&channelDesc2, width, height, width * sizeof(float)));
	cutilSafeCall(cudaBindTexture2D(&offset, redRefPtr, (const void*)d_redCurrent, 
		&channelDesc3, width, height, width * sizeof(int)));
	cutilSafeCall(cudaBindTexture2D(&offset, greenRefPtr, (const void*)d_greenCurrent, 
		&channelDesc3, width, height, width * sizeof(int)));
	cutilSafeCall(cudaBindTexture2D(&offset, blueRefPtr, (const void*)d_blueCurrent, 
		&channelDesc3, width, height, width * sizeof(int)));
#ifdef TWO_BY_TWO
    cutilSafeCall(cudaBindTexture(&offset, svdRefPtrTL, (const void*)d_svdMatrixTL, 
                                  &channelDesc2, SVD_MAT_HEIGHT * sizeof(float)));
    cutilSafeCall(cudaBindTexture(&offset, svdRefPtrTR, (const void*)d_svdMatrixTR, 
                                  &channelDesc2, SVD_MAT_HEIGHT * sizeof(float)));
    cutilSafeCall(cudaBindTexture(&offset, svdRefPtrBL, (const void*)d_svdMatrixBL, 
                                  &channelDesc2, SVD_MAT_HEIGHT * sizeof(float)));
    cutilSafeCall(cudaBindTexture(&offset, svdRefPtrBR, (const void*)d_svdMatrixBR, 
                                  &channelDesc2, SVD_MAT_HEIGHT * sizeof(float)));
#else
    cutilSafeCall(cudaBindTexture(&offset, svdRefPtr, (const void*)d_svdMatrix, 
                                  &channelDesc2, SVD_MAT_HEIGHT * sizeof(float)));
#endif // TWO_BY_TWO

	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////

	// apply low-pass filter to initial image
	linear_wavelet_transform(d_wavelet, d_temp, 2, texRefPtr, width, height, 
							 scan_width, &channelDesc, default_block);
	
	// bilinear upscale
	cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)d_wavelet, 
				  &channelDesc, width, height, scan_width));
#ifdef TWO_BPP
    dim3 grid0(default_grid.x>>3, default_grid.y>>2);
    // one more time for 8x4
    linear_wavelet_transform_rows<<<grid0, default_block>>> (width, height, d_temp);
    
    cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)d_temp, 
                                    &channelDesc, width, height, scan_width));
    bilinear_resize8x4<<<default_grid, default_block>>> (d_wavelet, width, height, false);
    //printf("Finished low-pass filter...\n");
#else
	bilinear_resize4x4<<<default_grid, default_block>>> (d_temp, width, height, false);
	swap((void**)&d_temp, (void**)&d_wavelet);
#endif // TWO_BPP
	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////

	// make the inital A and B prototypes
	cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)d_wavelet, 
				  &channelDesc, width, height, scan_width));
	make_a_b_prototypes<<<default_grid, default_block>>> (d_aproto, d_bproto, 
		width, height);

    //printf("Made initial A and B prototypes...\n");
    
	// apply low-pass filter to A and B prototypes to get initial candidates
	linear_wavelet_transform(d_aproto, d_temp, 2, texRefPtr, width, height, 
		scan_width, &channelDesc, default_block);
    linear_wavelet_transform(d_bproto, d_temp, 2, texRefPtr, width, height, 
                             scan_width, &channelDesc, default_block);
    
#ifdef TWO_BPP
    // one more time for 8x4
    cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)d_aproto, 
                                    &channelDesc, width, height, scan_width));
    linear_wavelet_transform_rows<<<grid0, default_block>>> (width, height, d_temp);
    swap((void**)&d_temp, (void**)&d_aproto);
    
    cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)d_bproto, 
                                    &channelDesc, width, height, scan_width));
    linear_wavelet_transform_rows<<<grid0, default_block>>> (width, height, d_temp);
    swap((void**)&d_temp, (void**)&d_bproto);
    //printf("Finished low-pass filter for A and B...\n");
#endif   // TWO_BPP
    
	// copy the prototypes into temporary storage (re-using d_axis and d_wavelet
	// for temporary storage)
	cudaMemcpy(d_axis, d_aproto, tex_size, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_wavelet, d_bproto, tex_size, cudaMemcpyDeviceToDevice);

	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////

	// optimize A and B candidates and encode final image
	for (i = 0; i < NUM_OPTIMIZATION_PASSES; i++) {
		// expand the A and B candidates to full resolution
		cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)d_aproto, 
			&channelDesc, width, height, scan_width));
#ifdef TWO_BPP
        bilinear_resize8x4<<<default_grid, default_block>>> (d_temp, width, height, true);
#else
		bilinear_resize4x4<<<default_grid, default_block>>> (d_temp, width, height, true);
#endif // TWO_BPP
		swap((void**)&d_temp, (void**)&d_aproto);

		cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)d_bproto, 
			&channelDesc, width, height, scan_width));
#ifdef TWO_BPP
        bilinear_resize8x4<<<default_grid, default_block>>> (d_temp, width, height, true);
#else
		bilinear_resize4x4<<<default_grid, default_block>>> (d_temp, width, height, true);
#endif // TWO_BPP
		swap((void**)&d_temp, (void**)&d_bproto);

		// get the modulation bits 
		compute_modulation_bits<<<default_grid, default_block>>> (d_aproto, d_bproto, 
			d_redCurrent, d_greenCurrent, d_blueCurrent, d_mod, width);

#ifdef TWO_BPP
        compute_modulation_mode<<<grid0, default_block>>> (d_aproto, d_bproto, d_temp,
            d_mod, d_redCurrent, d_greenCurrent, d_blueCurrent, width);
#endif // TWO_BPP
		// make optimization call
		cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)d_bits, 
										&channelDesc, width, height, scan_width));
#ifdef USE_SVD
		svd_optimize<<<svdGrid, default_block>>>(d_aproto, d_bproto, d_axis,
			d_wavelet, width, height, d_err);
#else
#ifdef USE_JAMA_SVD
        svd_optimize<<<svdGrid, default_block>>>(d_aproto, d_bproto, d_axis,
			d_wavelet, width, height, d_err);
#else
#ifdef USE_CHOLESKY
        cholesky_optimize<<<svdGrid, default_block>>>(d_aproto, d_bproto, d_axis,
            d_wavelet, width, height, d_err);
#else
		moore_penrose_optimize<<<quarterGrid, default_block>>>(d_aproto, d_bproto, d_axis,
			d_wavelet, width, height, d_err);
#endif // USE_CHOLESKY  
#endif // USE_JAMA_SVD
#endif // USE_SVD
	}

	// check for SVD error
	int kernelError;
	cudaMemcpy(&kernelError, d_err, sizeof(int), cudaMemcpyDeviceToHost);
	if (kernelError == 1) {
//		printf("One or more SVD matrices did not converge in 30 iterations.\n");
//		printf("Compression failed.\n");
//		printf("Please try again.\n\n");
//		printf("Press ENTER to exit.\n");
//		getchar();
//		exit(EXIT_FAILURE);
        return EXIT_FAILURE;
	}

    //printf("Finished optimization passes...\n");
	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////

#ifdef DECOMPRESS_PVR
	// expand the A and B candidates to full resolution
	cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)d_aproto, 
		&channelDesc, width, height, scan_width));
#ifdef TWO_BPP
    bilinear_resize8x4<<<default_grid, default_block>>> (d_temp, width, height, true);
#else
    bilinear_resize4x4<<<default_grid, default_block>>> (d_temp, width, height, true);
#endif // TWO_BPP
	swap((void**)&d_temp, (void**)&d_aproto);

	cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)d_bproto, 
		&channelDesc, width, height, scan_width));
#ifdef TWO_BPP
    bilinear_resize8x4<<<default_grid, default_block>>> (d_temp, width, height, true);
#else
    bilinear_resize4x4<<<default_grid, default_block>>> (d_temp, width, height, true);
#endif // TWO_BPP
	swap((void**)&d_temp, (void**)&d_bproto);

	decompress<<<default_grid, default_block>>>(d_aproto, d_bproto, d_wavelet, d_mod, width);
	cutilSafeCall(cudaMemcpy(h_out, d_wavelet, tex_size, cudaMemcpyDeviceToHost));
#else

	// return compressed data
    //printf("Encoding Texture...\n");
#ifdef TWO_BPP
    encode_texture<<<grid0, default_block>>>(d_aproto, d_bproto, d_mod, width, 
        false, d_out, d_temp, width>>3, height>>2);
	cutilSafeCall(cudaMemcpy(h_out, d_out, 2 * (width>>3) * (height>>2) * 
                             sizeof(unsigned int), cudaMemcpyDeviceToHost));
#else
	encode_texture<<<quarterGrid, default_block>>>(d_aproto, d_bproto, d_mod, width, 
		false, d_out, d_temp, width>>2, height>>2);
	cutilSafeCall(cudaMemcpy(h_out, d_out, 2 * (width>>2) * (height>>2) * 
		sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(h_out, d_aproto, tex_size, cudaMemcpyDeviceToHost));
#endif  // TWO_BPP
    //printf("Finished copying back texture...\n");
    
#ifdef GET_RMS_ERROR
	// expand the A and B candidates to full resolution
	cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)d_aproto, 
		&channelDesc, width, height, scan_width));
#ifdef TWO_BPP
    bilinear_resize8x4<<<default_grid, default_block>>> (d_temp, width, height, true);
#else
    bilinear_resize4x4<<<default_grid, default_block>>> (d_temp, width, height, true);
#endif // TWO_BPP
	swap((void**)&d_temp, (void**)&d_aproto);

	cutilSafeCall(cudaBindTexture2D(&offset, texRefPtr, (const void*)d_bproto, 
		&channelDesc, width, height, scan_width));
#ifdef TWO_BPP
    bilinear_resize8x4<<<default_grid, default_block>>> (d_temp, width, height, true);
#else
    bilinear_resize4x4<<<default_grid, default_block>>> (d_temp, width, height, true);
#endif // TWO_BPP
	swap((void**)&d_temp, (void**)&d_bproto);

	// get the per pixel error
	unsigned int *h_rms, *d_rms, j;
	float error, mean_error, total_error = 0.0f;
	h_rms = (unsigned int*)malloc(width * height * sizeof(int));
	cutilSafeCall(cudaMalloc((void **)&d_rms, width * height * sizeof(int)));

	rms_error<<<default_grid, default_block>>>(d_bits, d_aproto, d_bproto, d_rms, width);
	cutilSafeCall(cudaMemcpy(h_rms, d_rms, width * height * sizeof(int), 
		cudaMemcpyDeviceToHost));
	cudaThreadSynchronize();
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			total_error += h_rms[i*width + j];
		}
	}
	mean_error = total_error / (width * height * 3);
	error = sqrtf(mean_error);
	printf("RMS Error is %.3f.\n", error);
#ifdef GET_SNR
	printf("Peak signal-to-noise ratio is %0.3fdB.\n", 10 * log10(SQR(255) / mean_error));
#endif // GET_SNR
#endif // GET_RMS_ERROR
#endif // DECOMPRESS_PVR

	// cleanup
	cudaDeviceReset();

	return EXIT_SUCCESS;
}