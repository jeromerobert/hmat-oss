#pragma once
#include <vector>
#include <mutex>

#include "rk_matrix.hpp"





#ifdef HAVE_COMPOSYX

#include "composyx.hpp"
#include "composyx/interfaces/basic_concepts.hpp"
#include "composyx/utils/Arithmetic.hpp"

#ifdef COMPOSYX_USE_ZFP_COMPRESSOR
#include "composyx/utils/ZFP_compressor.hpp"
#endif //COMPOSYX_USE_ZFP_COMPRESSOR

#ifdef COMPOSYX_USE_SZ_COMPRESSOR
#include "composyx/utils/SZ_compressor.hpp"
#endif //COMPOSYX_USE_SZ_COMPRESSOR

#ifdef COMPOSYX_USE_SZ3_COMPRESSOR
#include "composyx/utils/SZ3_compressor.hpp"
#endif //COMPOSYX_USE_SZ3_COMPRESSOR


#endif //HAVE_COMPOSYX


namespace hmat 
{


#ifdef HAVE_COMPOSYX

#ifdef COMPOSYX_USE_SZ_COMPRESSOR
template<typename T>
class SZcompressor : public FPCompressorInterface<T> {
private:
    composyx::SZ_compressor<T, composyx::SZ_CompressionMode::POINTWISE>* _compressor;
    size_t _size;

    mutable std::recursive_mutex _mutex;

public:
    SZcompressor() {};

    ~SZcompressor();

    void compress(std::vector<T> data, size_t size, double epsilon) override;

    std::vector<T> decompress() override;

    double get_ratio() override;

    
};
#endif //COMPOSYX_USE_SZ_COMPRESSOR

#ifdef COMPOSYX_USE_SZ3_COMPRESSOR

template<typename T>
class SZ3compressor : public FPCompressorInterface<T> {
private:
    composyx::SZ3_compressor<T, SZ3::EB::EB_REL>* _compressor;
    size_t _size;

    mutable std::recursive_mutex _mutex;

public:
    SZ3compressor() {};

    ~SZ3compressor();

    void compress(std::vector<T> data, size_t size, double epsilon) override;

    std::vector<T> decompress() override;

    double get_ratio() override;

    
};

#endif //COMPOSYX_USE_SZ3_COMPRESSOR

#ifdef COMPOSYX_USE_ZFP_COMPRESSOR

template<typename T>
class ZFPcompressor : public FPCompressorInterface<T> {
private:
    composyx::ZFP_compressor<T, composyx::ZFP_CompressionMode::ACCURACY>* _compressor;
    size_t _size;

    mutable std::recursive_mutex _mutex;

public:
    ZFPcompressor() {};

    ~ZFPcompressor();

    void compress(std::vector<T> data, size_t size, double epsilon) override;

    std::vector<T> decompress() override;

    double get_ratio() override;

    
};

#endif //COMPOSYX_USE_ZFP_COMPRESSOR


#endif // HAVE_COMPOSYX


/* Default compressor if composyx is not installed
*/
template<typename T>
class Defaultcompressor : public FPCompressorInterface<T> {
private:
    std::vector<T> _data;

public:
    Defaultcompressor() {};

    void compress(std::vector<T> data, size_t size, double epsilon) override;

    std::vector<T> decompress() override;

    double get_ratio() override;

    
};

}



