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



            
        case ZFP_COMPRESSOR:
            res = new ZFPcompressor<T>();

            break;
        case SZ3_COMPRESSOR:
            res = new SZ3compressor<T>();

            break;


        case SZ_COMPRESSOR:
            res = new SZcompressor<T>();
            break;
        
        #endif //HAVE_COMPOSYX


            
        case DEFAULT_COMPRESSOR:          
        default:
            res = new Defaultcompressor<T>();
        }

    return res;
}


template<typename T>
void Defaultcompressor<T>::compress(std::vector<T> data, size_t size, double epsilon)
{
    this->_data = data;
}

template<typename T>
std::vector<T> Defaultcompressor<T>::decompress()
{
    return _data;
}

template <typename T>
double Defaultcompressor<T>::get_ratio()
{
    return 1;
}


#ifdef HAVE_COMPOSYX


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
void SZcompressor<T>::compress(std::vector<T> data, size_t size, double epsilon)
{
    this->_size = size;
    this->_compressor = new composyx::SZ_compressor<T, composyx::SZ_CompressionMode::POINTWISE>(data.data(), size, epsilon);
}

template<typename T>
std::vector<T> SZcompressor<T>::decompress()
{
    std::vector<T> out = _compressor->decompress();
    delete _compressor;
    _compressor = nullptr;
    return out;
}

template <typename T>
double SZcompressor<T>::get_ratio()
{
    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n",sizeof(T)* this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    
    return this->_compressor->get_ratio();
}

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
void SZ3compressor<T>::compress(std::vector<T> data, size_t size, double epsilon)
{
    this->_size = size;
    this->_compressor = new composyx::SZ3_compressor<T, SZ3::EB::EB_REL>(data.data(), size, epsilon);
    //this->_compressor->print_config();
}

template<typename T>
std::vector<T> SZ3compressor<T>::decompress()
{
    std::vector<T> out = _compressor->decompress();
    delete _compressor;
    _compressor = nullptr;
    return out;
}

template <typename T>
double SZ3compressor<T>::get_ratio()
{
    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n", sizeof(T)*this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    
    return this->_compressor->get_ratio();
}

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
void ZFPcompressor<T>::compress(std::vector<T> data, size_t size, double epsilon)
{
    this->_size = size;
    this->_compressor = new composyx::ZFP_compressor<T, composyx::ZFP_CompressionMode::ACCURACY>(data.data(), size, epsilon);
}

template<typename T>
std::vector<T> ZFPcompressor<T>::decompress()
{
    std::vector<T> out = _compressor->decompress();
    delete _compressor;
    _compressor = nullptr;
    return out;
}

template <typename T>
double ZFPcompressor<T>::get_ratio()
{
    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n", sizeof(T)*this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    return this->_compressor->get_ratio();
}


// Templates declaration
template class SZcompressor<S_t>;
template class SZcompressor<D_t>;
template class SZcompressor<C_t>;
template class SZcompressor<Z_t>;

template class SZ3compressor<S_t>;
template class SZ3compressor<D_t>;
template class SZ3compressor<C_t>;
template class SZ3compressor<Z_t>;

template class ZFPcompressor<S_t>;
template class ZFPcompressor<D_t>;
template class ZFPcompressor<C_t>;
template class ZFPcompressor<Z_t>;

#endif // HAVE_COMPOSYX



template FPCompressorInterface<S_t>* initCompressor(hmat_FPcompress_t method);
template FPCompressorInterface<D_t>* initCompressor(hmat_FPcompress_t method);
template FPCompressorInterface<C_t>* initCompressor(hmat_FPcompress_t method);
template FPCompressorInterface<Z_t>* initCompressor(hmat_FPcompress_t method);

}

