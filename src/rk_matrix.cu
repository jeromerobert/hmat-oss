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
    __global__ void FindK(T* data, double epsilon, int size, int *new_size) {
        extern __shared__ int shared_array[]; // taille donnée au lancement du kernel
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ T threshold_eigenvalue;
        
        if(threadIdx.x == 0){
            threshold_eigenvalue = data[0] * epsilon;
        }
        __syncthreads();


        if (idx < size){
            if (data[idx] > threshold_eigenvalue){
                shared_array[threadIdx.x] = idx;
            } 
            else {
                shared_array[threadIdx.x] = -1;
            }
        }
        __syncthreads();
        int i = size / 2;
        while(i > 0){
            if(threadIdx.x < i){
                shared_array[threadIdx.x] = max(shared_array[threadIdx.x], shared_array[threadIdx.x + i]);
            }
            __syncthreads();
            i /= 2;
        }

        if(threadIdx.x == 0){
            printf("Max index satisfying condition: %d\n", shared_array[0]);
            atomicMax(new_size, shared_array[0] + 1);
        }
    }
// atomicMax garantit que la mise à jour sur new_size est atomique, donc que la variable ne sera pas corrompue si plusieurs blocs écrivent en même temps.

template<typename T>
    void launch_FindK(T* S_gpu, double epsilon, int size, int* newK_gpu) {
        int blockSize = 128;
        int numBlocks = (size + blockSize - 1) / blockSize;
        size_t sharedMemSize = blockSize * sizeof(int);
        FindK<T><<<numBlocks, blockSize, sharedMemSize>>>(S_gpu, epsilon, size, newK_gpu);
        cudaDeviceSynchronize();
    }

extern "C" void launch_FindK_float(float* S_gpu, double epsilon, int size, int* newK_gpu) {
    launch_FindK<float>(S_gpu, epsilon, size, newK_gpu);
}

extern "C" void launch_FindK_double(double* S_gpu, double epsilon, int size, int* newK_gpu) {
    launch_FindK<double>(S_gpu, epsilon, size, newK_gpu);
}
    
/*
extern "C" void launch_FindK_cuComplex(cuComplex* S_gpu, double epsilon, int size, int* newK_gpu) {
    int blockSize = 128;
    int numBlocks = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(int);
    FindK<cuComplex><<<numBlocks, blockSize, sharedMemSize>>>(S_gpu, epsilon, size, newK_gpu);
}

extern "C" void launch_FindK_cuDoubleComplex(cuDoubleComplex* S_gpu, double epsilon, int size, int* newK_gpu) {
    int blockSize = 128;
    int numBlocks = (size + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(int);
    FindK<cuDoubleComplex><<<numBlocks, blockSize, sharedMemSize>>>(S_gpu, epsilon, size, newK_gpu);
}

*/