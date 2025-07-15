#pragma once
#include <vector>

#include "rk_matrix.hpp"

#ifdef HAVE_COMPOSYX
#include "composyx.hpp"
#include "composyx/interfaces/basic_concepts.hpp"
#include "composyx/utils/Arithmetic.hpp"
#include "composyx/utils/ZFP_compressor.hpp"
#include "composyx/utils/SZ_compressor.hpp"
#include "composyx/utils/SZ3_compressor.hpp"

namespace hmat 
{


template<typename T>
class SZ2compressor : public FPCompressorInterface<T> {
private:
    composyx::SZ_compressor<T, composyx::SZ_CompressionMode::POINTWISE>* _compressor;
    size_t _size;

public:
    SZ2compressor() {};

    void compress(T* data, size_t size, double epsilon) override;

    T* decompress() override;

    double get_ratio() override;

    
};


template<typename T>
class SZ3compressor : public FPCompressorInterface<T> {
private:
    composyx::SZ3_compressor<T, SZ3::EB::EB_REL>* _compressor;
    size_t _size;

public:
    SZ3compressor() {};

    void compress(T* data, size_t size, double epsilon) override;

    T* decompress() override;

    double get_ratio() override;

    
};

template<typename T>
class ZFPcompressor : public FPCompressorInterface<T> {
private:
    composyx::ZFP_compressor<T, composyx::ZFP_CompressionMode::ACCURACY>* _compressor;
    size_t _size;

public:
    ZFPcompressor() {};

    void compress(T* data, size_t size, double epsilon) override;

    T* decompress() override;

    double get_ratio() override;

    
};


}
#else // HAVE_COMPOSYX

namespace hmat 
{

/* Default compressor if composyx is not installed
*/
template<typename T>
class Defaultcompressor : public FPCompressorInterface<T> {
private:
    T* _data;

public:
    Defaultcompressor() {};

    void compress(T* data, size_t size, double epsilon) override;

    T* decompress() override;

    double get_ratio() override;

    
};

}


#endif // HAVE_COMPOSYX
