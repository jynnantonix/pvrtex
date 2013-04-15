INTRODUCTION

This is a compressor to convert images to the PVR texture format
commonly used in Android and iOS devices.  The compressor folder
contains the current working implementation.  It will convert any
input image to the PVR texture format, then decompress it and write it
out as a png.  The current implementation also converts the image
using experimental color space transformations and will write out the
results as pngs.  For more information, please read FinalReport.pdf.
This currently only supports 4bpp compression.

The cudapvr folder contains an older, badly out of date implementation
written in CUDA.  This supports writing out in the actual PVR format
for both 4 bits per pixel and 2 bits per pixel modes.  It was written
very early in my career and has become quite unmaintainable but it was
useful as a reference for the current implementation.

DEPENDENCIES

CMake (http://www.cmake.org/)
FreeImage (http://freeimage.sourceforge.net/)
Eigen3 (http://eigen.tuxfamily.org/)
OpenMP (http://openmp.org/) - OPTIONAL

USAGE

Compile the program using CMake and your favorite build toolchain.
Make sure you compile it in release mode by passing
-DCMAKE_BUILD_TYPE=Release as a parameter to CMake.  Otherwise, it
will be *very* slow.  To run the compressor type

./pvrtex <input> <output_filename_prefix>

This will write out 4 files.  So for example, if I ran

./pvrtex lena.png lena

I would get lena_pvrtc4bpp.png, lena_yuv4bpp.png, lena_yuvext4bpp.png,
and lena_yuvopt4bpp.png.  The program will also print the root
mean-squared (RMS) error for each compression method to standard out.