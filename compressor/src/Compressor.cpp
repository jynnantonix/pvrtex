//==========================================================================//
//                                                                          //
// @file            Compressor.cpp                                          //
// @author          Chirantan Ekbote (ekbote@seas.harvard.edu)              //
// @date            2012/11/05                                              //
// @version         0.2                                                     //
// @brief           Class implementation for pvr texture compressor         //
//                                                                          //
//==========================================================================//

#include "../inc/Compressor.h"

Compressor::Compressor(int w, int h, int sw, PVRTEX::IMAGE_FORMAT f, BYTE *d)
    :
    m_width(w),
    m_height(h),
    m_scanWidth(sw),
    m_format(f),
    m_data(d)
{
}

Compressor::~Compressor()
{
}

void Compressor::setWidth(int w)
{
    m_width = w;
}

void Compressor::setHeight(int h)
{
    m_height = h;
}

void Compressor::setScanWidth(int sw)
{
    m_scanWidth = sw;
}

void Compressor::setData(BYTE *d)
{
    m_data = d;
}

int Compressor::getWidth()
{
    return m_width;
}

int Compressor::getHeight()
{
    return m_height;
}

int Compressor::getScanWidth()
{
    return m_scanWidth;
}

PVRTEX::IMAGE_FORMAT Compressor::getFormat()
{
    return m_format;
}

BYTE* Compressor::getData()
{
    return m_data;
}

inline BYTE Compressor::make_alpha(unsigned int p)
{
    return ((p & FI_RGBA_ALPHA_MASK)>>FI_RGBA_ALPHA_SHIFT);
}

inline BYTE Compressor::make_red(unsigned int p)
{
    return ((p & FI_RGBA_RED_MASK)>>FI_RGBA_RED_SHIFT);
}

inline BYTE Compressor::make_green(unsigned int p)
{
    return ((p & FI_RGBA_GREEN_MASK)>>FI_RGBA_GREEN_SHIFT);
}

inline BYTE Compressor::make_blue(unsigned int p)
{
    return ((p & FI_RGBA_BLUE_MASK)>>FI_RGBA_BLUE_SHIFT);
}

void Compressor::compress(BYTE *out, PVRTEX::IMAGE_FORMAT format) 
{
    //unsigned int *bits = (unsigned int*) malloc(m_height * m_scanWidth);
    //bits = (unsigned int*)m_data;
    //memcpy((void*)out, (void*)m_data, m_height * m_scanWidth);
    
    // Create the matrix with the original data
    Eigen::MatrixXi bits(m_width, m_height);
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            unsigned int pixel = 0;
            int idx = 4 * (y*m_width + x);
            for (int k = 0; k < 4; ++k) {
                pixel = (pixel << 8) | m_data[idx + k];
            }
            bits(y, x) = pixel;
        }
    }
    
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            int idx = 4 * (y*m_width + x);
            unsigned int pixel = bits(y, x);
            out[idx] = make_alpha(pixel);
            out[idx+1] = make_red(pixel);
            out[idx+2] = make_green(pixel);
            out[idx+3] = make_blue(pixel);
        }
    }
    
}

void Compressor::writeToFile(const char *filename)
{
    
}
    
