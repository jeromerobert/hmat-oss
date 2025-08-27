#include <cuda_runtime.h>
#include "rk_matrix.hpp"
#include <cassert>
#include <cmath>

/* Compilation avec NVCC */

template<typename T>
    __global__ void Sqrt_SingularValues_Kernel(T *S, int k) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        if (idx == 0 && gridDim.x * blockDim.x < k) {
            // gridDim.x * blockDim.x correspond au nombre total de threads lancés pour ce kernel
            printf("Attention : nombre total de threads (%d) insuffisant pour traiter %d éléments.\n"
                "Suggestion : augmentez le nombre de blocs ou de threads par bloc.\n",
                gridDim.x * blockDim.x, k);
                assert(false);
        }
        __syncthreads();
        if (idx < k) {
            using ::sqrt;
            S[idx] = sqrt(S[idx]);
        }

    }

template <typename T>
    void launch_Sqrt_SingularVals_Kernel(T* deviceData, int k) {
        int blockSize = 256;
        int gridSize = (k + blockSize - 1) / blockSize;
        Sqrt_SingularValues_Kernel<T><<<gridSize, blockSize>>>(deviceData, k);
        cudaDeviceSynchronize();
    }

extern "C" void launch_Sqrt_SingularVals_Kernel_float(float* S_gpu, int k) {
    launch_Sqrt_SingularVals_Kernel<float>(S_gpu, k);
}

extern "C" void launch_Sqrt_SingularVals_Kernel_double(double* S_gpu, int k) {
    launch_Sqrt_SingularVals_Kernel<double>(S_gpu, k);
}


template<typename T>
    __global__ void FindK(T* d_singulaur_values, double epsilon, int old_rank, int *new_rank) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ T threshold_eigenvalue;
        
        if(threadIdx.x == 0){
            // Tous les threads partagent une valeur threshold_eigenvalue, initialisée par threadIdx.x == 0
            threshold_eigenvalue = d_singulaur_values[0] * epsilon;
        }
        __syncthreads();
        if (idx < old_rank) {
            // Check if the singular value at this index is below the threshold
            if (d_singulaur_values[idx] <= threshold_eigenvalue) {
                // If it is, atomically update the result with the minimum index.
                // This is a race condition handled by hardware to ensure correctness.
                // All threads that find a value below the threshold will try to write their
                // index, but only the smallest index will ultimately be stored.
                atomicMin(new_rank, idx);
            }
        }

    }

__global__ void double_to_cuDoubleComplex_kernel(double* in, cuDoubleComplex* out, int k) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < k) {
        out[idx] = make_cuDoubleComplex(in[idx], 0.0f); // imag = 0
    }
}

__global__ void float_to_cuComplex_kernel(float* in, cuComplex* out, int k) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < k) {
        out[idx] = make_cuComplex(in[idx], 0.0f); // imag = 0
    }
}


template<typename T>
    void launch_FindK(T* S_gpu, double epsilon, int old_rank, int* newK_gpu) {
        int blockSize = 128;
        int numBlocks = (old_rank + blockSize - 1) / blockSize;
        FindK<T><<<numBlocks, blockSize>>>(S_gpu, epsilon, old_rank, newK_gpu);
        cudaDeviceSynchronize();
    }

extern "C" void launch_FindK_float(float* S_gpu, double epsilon, int size, int* newK_gpu) {
    launch_FindK<float>(S_gpu, epsilon, size, newK_gpu);
}

extern "C" void launch_FindK_double(double* S_gpu, double epsilon, int size, int* newK_gpu) {
    launch_FindK<double>(S_gpu, epsilon, size, newK_gpu);
}

extern "C" void convert_double_to_cuDoubleComplex(double* in, cuDoubleComplex* out, int k) {
    int blockSize = 128;
    int numBlocks = (k + blockSize - 1) / blockSize;
    double_to_cuDoubleComplex_kernel<<<numBlocks, blockSize>>>(in, out, k);
    //cudaDeviceSynchronize();
}

extern "C" void convert_float_to_cuComplex(float* in, cuComplex* out, int k) {
    int blockSize = 128;
    int numBlocks = (k + blockSize - 1) / blockSize;
    float_to_cuComplex_kernel<<<numBlocks, blockSize>>>(in, out, k);
    //cudaDeviceSynchronize();
}    