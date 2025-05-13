#include "fp_compression.hpp"
#include "data_types.hpp"



namespace hmat{

template<typename T>
FPCompressorInterface<T>* initCompressor(hmat_FPcompress_t method)
{
    FPCompressorInterface<T>* res;

    switch (method)
        {
        case ZFP_COMPRESSOR:
            res = new ZFPcompressor<T>();

            break;
        case SZ3_COMPRESSOR:
            res = new SZ3compressor<T>();

            break;


        case SZ2_COMPRESSOR:
        case DEFAULT_COMPRESSOR:          
        default:
            res = new SZ2compressor<T>();
            break;
        }

    return res;
}

template<typename T>
void SZ2compressor<T>::compress(T* data, size_t size, double epsilon)
{
    this->_size = size;
    //this->_compressor = new composyx::SZ_compressor<T, composyx::SZ_CompressionMode::POINTWISE>(data, size, epsilon);
}

template<typename T>
T* SZ2compressor<T>::decompress()
{
    T* out = new T[_size];
    //_compressor->decompress(out);
    //delete _compressor;
    return out;
}

template <typename T>
double SZ2compressor<T>::get_ratio()
{
    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n",sizeof(T)* this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    
    return 0;//this->_compressor->get_ratio();
}


template<typename T>
void SZ3compressor<T>::compress(T* data, size_t size, double epsilon)
{
    this->_size = size;
    //this->_compressor = new composyx::SZ3_compressor<T, SZ3::EB::EB_REL>(data, size, epsilon);
    //this->_compressor->print_config();
}

template<typename T>
T* SZ3compressor<T>::decompress()
{
    T* out = new T[_size];
    //_compressor->decompress(out);
    //delete _compressor;
    return out;
}

template <typename T>
double SZ3compressor<T>::get_ratio()
{
    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n", sizeof(T)*this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    
    return 0;//this->_compressor->get_ratio();
}


template<typename T>
void ZFPcompressor<T>::compress(T* data, size_t size, double epsilon)
{
    this->_size = size;
    //this->_compressor = new composyx::ZFP_compressor<T, composyx::ZFP_CompressionMode::ACCURACY>(data, size, epsilon);
}

template<typename T>
T* ZFPcompressor<T>::decompress()
{
    T* out = new T[_size];
    //_compressor->decompress(out);
    //delete _compressor;
    return out;
}

template <typename T>
double ZFPcompressor<T>::get_ratio()
{
    //printf("Number of bytes : %ld, Compressed bytes : %ld, ratio = %f\n", sizeof(T)*this->_compressor->get_n_elts(), this->_compressor->get_compressed_bytes(), this->_compressor->get_ratio());
    return 0;//this->_compressor->get_ratio();
}


// Templates declaration
template class SZ2compressor<S_t>;
template class SZ2compressor<D_t>;
template class SZ2compressor<C_t>;
template class SZ2compressor<Z_t>;

template class SZ3compressor<S_t>;
template class SZ3compressor<D_t>;
template class SZ3compressor<C_t>;
template class SZ3compressor<Z_t>;

template class ZFPcompressor<S_t>;
template class ZFPcompressor<D_t>;
template class ZFPcompressor<C_t>;
template class ZFPcompressor<Z_t>;

template FPCompressorInterface<S_t>* initCompressor(hmat_FPcompress_t method);
template FPCompressorInterface<D_t>* initCompressor(hmat_FPcompress_t method);
template FPCompressorInterface<C_t>* initCompressor(hmat_FPcompress_t method);
template FPCompressorInterface<Z_t>* initCompressor(hmat_FPcompress_t method);

}


