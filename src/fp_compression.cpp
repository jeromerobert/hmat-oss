#include "config.h"
#include "fp_compression.hpp"
#include "data_types.hpp"


namespace hmat{

/**
 * Instanciate FP compressors depending on the chosen method. If the method is unknown, undefined or not included, return a default compressor (which does not compress)
 */
template<typename T>
FPCompressorInterface<T>* initCompressor(hmat_FPcompress_t method)
{
    FPCompressorInterface<T>* res;

    switch (method)
        {
    #ifdef HAVE_COMPOSYX

    #ifdef COMPOSYX_USE_ZFP_COMPRESSOR

            
        case ZFP_COMPRESSOR:
            res = new ZFPcompressor<T>();

            break;
    #endif //COMPOSYX_USE_ZFP_COMPRESSOR

    #ifdef COMPOSYX_USE_SZ3_COMPRESSOR
        case SZ3_COMPRESSOR:
            res = new SZ3compressor<T>();

            break;

    #endif //COMPOSYX_USE_SZ3_COMPRESSOR

    #ifdef COMPOSYX_USE_SZ_COMPRESSOR

        case SZ_COMPRESSOR:
            res = new SZcompressor<T>();
            break;
    #endif //COMPOSYX_USE_SZ_COMPRESSOR
        
    #endif //HAVE_COMPOSYX


            
        case DEFAULT_COMPRESSOR:          
        default:
            res = new Defaultcompressor<T>();
        }

    return res;
}


template<typename T>
void Defaultcompressor<T>::compress(T* data, size_t size, double epsilon)
{
    this->_data = std::vector<T>(data, data + size);
}

template<typename T>
std::vector<T> Defaultcompressor<T>::decompress()
{
    std::vector<T> out = _data;
    this->_data.clear();
    return out;
}

template <typename T>
void Defaultcompressor<T>::decompress(T *dest)
{
    decompressCopy(dest);
    this->_data.clear();
}

template <typename T>
std::vector<T> Defaultcompressor<T>::decompressCopy()
{
    return _data;
}

template <typename T>
void Defaultcompressor<T>::decompressCopy(T *dest)
{
    std::copy(_data.begin(), _data.end(), dest);
}

template <typename T>
double Defaultcompressor<T>::get_ratio()
{
    return 1;
}


#ifdef HAVE_COMPOSYX


#ifdef COMPOSYX_USE_SZ_COMPRESSOR

template <typename T>
SZcompressor<T>::~SZcompressor()
{
    if(_compressor)
    {
        delete _compressor;
        _compressor = nullptr;
    }
}

template <typename T>
void SZcompressor<T>::compress(T* data, size_t size, double epsilon)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    //epsilon if absolute value but composyx::SZ_compressor only accept relative error bound at the moment

    this->_size = size;

    double max_abs = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        max_abs = std::max(max_abs, (double)std::abs(data[i]));
    }
    double zeta_rel = epsilon / std::max(max_abs, 1e-300);

    this->_compressor = new composyx::SZ_compressor<T, composyx::SZ_CompressionMode::POINTWISE>(data, size, zeta_rel);
}

template<typename T>
std::vector<T> SZcompressor<T>::decompress()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    std::vector<T> out = decompressCopy();
    delete _compressor;
    _compressor = nullptr;
    return out;
}

template <typename T>
void SZcompressor<T>::decompress(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    decompressCopy(dest);
    delete _compressor;
    _compressor = nullptr;
}

template <typename T>
std::vector<T> SZcompressor<T>::decompressCopy()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    return _compressor->decompress();
}

template <typename T>
void SZcompressor<T>::decompressCopy(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    _compressor->decompress(dest);
}

template <typename T>
double SZcompressor<T>::get_ratio()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n",sizeof(T)* this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    
    return this->_compressor->get_ratio();
}

#endif //COMPOSYX_USE_SZ_COMPRESSOR


#ifdef COMPOSYX_USE_SZ3_COMPRESSOR

template <typename T>
SZ3compressor<T>::~SZ3compressor()
{
     if(_compressor)
    {
         delete _compressor;
        _compressor = nullptr;
    }
}

template <typename T>
void SZ3compressor<T>::compress(T* data, size_t size, double epsilon)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    this->_size = size;
    this->_compressor = new composyx::SZ3_compressor<T, SZ3::EB::EB_ABS>(data, size, epsilon);
}

template<typename T>
std::vector<T> SZ3compressor<T>::decompress()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    std::vector<T> out = decompressCopy();
    delete _compressor;
    _compressor = nullptr;
    return out;
}

template <typename T>
void SZ3compressor<T>::decompress(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    decompressCopy(dest);
    delete _compressor;
    _compressor = nullptr;
}

template <typename T>
std::vector<T> SZ3compressor<T>::decompressCopy()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    return _compressor->decompress();
}

template <typename T>
void SZ3compressor<T>::decompressCopy(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    _compressor->decompress(dest);
}

template <typename T>
double SZ3compressor<T>::get_ratio()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n", sizeof(T)*this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    
    return this->_compressor->get_ratio();
}


#endif //COMPOSYX_USE_SZ3_COMPRESSOR

#ifdef COMPOSYX_USE_ZFP_COMPRESSOR

template <typename T>
ZFPcompressor<T>::~ZFPcompressor()
{
     if(_compressor)
    {
         delete _compressor;
        _compressor = nullptr;
    }
}

template <typename T>
void ZFPcompressor<T>::compress(T* data, size_t size, double epsilon)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    this->_size = size;
    this->_compressor = new composyx::ZFP_compressor<T, composyx::ZFP_CompressionMode::ACCURACY>(data, size, epsilon);
}

template<typename T>
std::vector<T> ZFPcompressor<T>::decompress()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    std::vector<T> out = decompressCopy();
    delete _compressor;
    _compressor = nullptr;
    return out;
}

template <typename T>
void ZFPcompressor<T>::decompress(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    decompressCopy(dest);
    delete _compressor;
    _compressor = nullptr;
}

template <typename T>
std::vector<T> ZFPcompressor<T>::decompressCopy()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    return _compressor->decompress();
}

template <typename T>
void ZFPcompressor<T>::decompressCopy(T *dest)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    _compressor->decompress(dest);
}

template <typename T>
double ZFPcompressor<T>::get_ratio()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n", sizeof(T)*this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    return this->_compressor->get_ratio();
}

// Templates declaration

#endif //COMPOSYX_USE_ZFP_COMPRESSOR

#endif // HAVE_COMPOSYX



template FPCompressorInterface<S_t>* initCompressor(hmat_FPcompress_t method);
template FPCompressorInterface<D_t>* initCompressor(hmat_FPcompress_t method);
template FPCompressorInterface<C_t>* initCompressor(hmat_FPcompress_t method);
template FPCompressorInterface<Z_t>* initCompressor(hmat_FPcompress_t method);

}

