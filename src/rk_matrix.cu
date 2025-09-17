#include <cuda_runtime.h>
#include <cassert>
#include <cmath>

/* Compilation avec NVCC */
template<typename T>
__global__ void FindKAndSqrtAll_Kernel(T* S, double epsilon, int old_rank, int *new_rank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // --- Step 1: compute the new rank ---
    __shared__ T threshold_eigenvalue;
    if(threadIdx.x == 0){
        threshold_eigenvalue = S[0] * epsilon;
    }
    __syncthreads(); // Intra-bloc synchronization
    
    if (idx < old_rank) {
        if (S[idx] <= threshold_eigenvalue) {
            atomicMin(new_rank, idx);
        }

        // Step 2: compute the square roots of ALL singular values, since we don't know new_rank yet
        using ::sqrt;
        S[idx] = sqrt(S[idx]);
    }
}

template<typename T>
void launch_FindKAndSqrtAll(T* S_gpu, double epsilon, int old_rank, int* newK_gpu) {
    int blockSize = 256;
    int numBlocks = (old_rank + blockSize - 1) / blockSize;
    FindKAndSqrtAll_Kernel<T><<<numBlocks, blockSize>>>(S_gpu, epsilon, old_rank, newK_gpu);
    cudaDeviceSynchronize();
}

template void launch_FindKAndSqrtAll<float>(float* S_gpu, double epsilon, int old_rank, int* newK_gpu);
template void launch_FindKAndSqrtAll<double>(double* S_gpu, double epsilon, int old_rank, int* newK_gpu);
