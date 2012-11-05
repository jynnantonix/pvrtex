//=========================================================================//
//                                                                         //
// @file            Compressor.h                                           //
// @author          Chirantan Ekbote (ekbote@seas.harvard.edu)             //
// @date            2012/11/05                                             //
// @version         0.1                                                    //
// @brief           Class declaration for pvr texture compressor           //
//                                                                         //
//=========================================================================//

#include <FreeImage.h>
#include <Eigen/Dense>

namespace PVRTEX
{
    enum IMAGE_FORMAT { A8R8G8B8, R8G8B8, PVRTC4, PVRTC2 };
};

class Compressor
{
public:
    
    Compressor(int w, int h, PVRTEX::IMAGE_FORMAT f, BYTE *d);

    void setWidth(int w);
    void setHeight(int h);
    void setScanWidth(int sw);
    void setFormat(PVRTEX::IMAGE_FORMAT f);
    void setData(BYTE *d);
    
    int getWidth();
    int getHeight();
    int getScanWidth();
    PVRTEX::IMAGE_FORMAT getFormat();
    BYTE* getData();

    void compress(BYTE *out, PVRTEX::IMAGE_FORMAT format = PVRTEX::PVRTC4); 
    
private:
    Compressor();
    ~Compressor();
    
    int m_width;
    int m_height;
    int m_scanWidth;
    PVRTEX::IMAGE_FORMAT m_format;
    BYTE *m_data;
};
    
    
