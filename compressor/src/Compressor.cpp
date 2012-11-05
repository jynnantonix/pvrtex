//==========================================================================//
//                                                                          //
// @file            Compressor.cpp                                          //
// @author          Chirantan Ekbote (ekbote@seas.harvard.edu)              //
// @date            2012/11/05                                              //
// @version         0.1                                                     //
// @brief           Class implementation for pvr texture compressor         //
//                                                                          //
//==========================================================================//

#include "../inc/Compressor.h"

Compressor::Compressor(int w, int h, PVRTEX::IMAGE_FORMAT f, BYTE *d)
    :
    m_width(w),
    m_height(h),
    m_format(f),
    m_data(d)
{
    m_scanWidth = m_width * sizeof(unsigned int);
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

void Compressor::compress(BYTE *out, PVRTEX::IMAGE_FORMAT format) 
{
    
}
    
