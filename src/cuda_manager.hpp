#pragma once

#include <iostream>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdlib> // For exit() and EXIT_FAILURE

// Helper macro to check for library status errors
#define CUDA_CHECK(call)				\
  do {									\
    auto status = call;							\
    if ((int)status != (int)cudaSuccess) { /* This is 0 for both libs */ \
      fprintf(stderr, "CUDA Error %d (%s: %s) at %s:%d\n",				\
	      status, cudaGetErrorName(status), cudaGetErrorString(status), __FILE__, __LINE__); \
      /* In a real app, you might throw an exception */                 \
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)
// Helper macro to check for library status errors
#define CUBLAS_CHECK(call)				\
  do {									\
    auto status = call;							\
    if ((int)status != (int)CUBLAS_STATUS_SUCCESS) { /* This is 0 for both libs */ \
      fprintf(stderr, "cuBLAS Error %d (%s: %s) at %s:%d\n",				\
	      status, cublasGetStatusName(status), cublasGetStatusString(status), __FILE__, __LINE__); \
      /* In a real app, you might throw an exception */                 \
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)
// Helper macro to check for library status errors
#define CUSOLVER_CHECK(call)				\
  do {									\
    auto status = call;							\
    if ((int)status != (int)CUSOLVER_STATUS_SUCCESS) { /* This is 0 for both libs */ \
      fprintf(stderr, "cuSOLVER Error %d at %s:%d\n",				\
	      status, __FILE__, __LINE__);				\
      /* In a real app, you might throw an exception */                 \
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)

namespace hmat {

  /**
   * @class CudaManager
   * @brief Manages the single, global instance of cuBLAS and cuSOLVER handles.
   *
   * This singleton provides a central point of access for CUDA library handles,
   * ensuring they are initialized only once and cleaned up automatically.
   */
  class CudaManager {
  public:
    /**
     * @brief Returns a reference to the single instance of the CudaManager.
     *
     * The first time this is called, it will construct and initialize the manager,
     * creating the CUDA handles. Subsequent calls return the existing instance.
     * @return A reference to the CudaManager singleton.
     */
    static CudaManager& getInstance() {
      // C++11 guarantees this static local variable is initialized only once
      // in a thread-safe manner.
      static CudaManager instance;
      return instance;
    }

    // --- Public Getters for Handles ---

    cublasHandle_t getCublasHandle() const {
      return cublas_handle_;
    }

    cusolverDnHandle_t getCusolverHandle() const {
      return cusolver_handle_;
    }

    // --- Rule of Five: Prevent Duplication ---
    // Delete copy constructor and copy assignment to ensure only one instance exists.
    CudaManager(const CudaManager&) = delete;
    CudaManager& operator=(const CudaManager&) = delete;
    // Move operations are also deleted for simplicity.
    CudaManager(CudaManager&&) = delete;
    CudaManager& operator=(CudaManager&&) = delete;


  private:
    /**
     * @brief Private constructor to enforce the singleton pattern.
     *
     * This is where the expensive, one-time initialization of CUDA
     * library handles occurs.
     */
    CudaManager() {
      std::cout << "--> Initializing global CUDA handles (one-time operation)..." << std::endl;
      CUBLAS_CHECK(cublasCreate(&cublas_handle_));
      CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle_));
      std::cout << "--> Handles initialized successfully." << std::endl;
    }

    /**
     * @brief Private destructor for automatic cleanup.
     *
     * This is called automatically when the program exits, ensuring that
     * all acquired resources are released properly.
     */
    ~CudaManager() {
      std::cout << "--> Destroying global CUDA handles (at program exit)..." << std::endl;
      if (cusolver_handle_) {
	CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle_));
      }
      if (cublas_handle_) {
	CUBLAS_CHECK(cublasDestroy(cublas_handle_));
      }
      std::cout << "--> Handles destroyed successfully." << std::endl;
    }

    // Member variables holding the unique handles
    cublasHandle_t   cublas_handle_   = nullptr;
    cusolverDnHandle_t cusolver_handle_ = nullptr;
  };

}  // end namespace hmat

namespace proxy_cuda {

  // GEQRF computes a QR factorization of a m-by-n matrix A = Q * R.
  // A is overwritten by Q, Ra and tau are returned in arrays allocated in this routine.
  template <typename T> int geqrf(int m, int n, T *a_gpu, int lda, T **tauA_gpu, T **Ra_gpu);
  
  template <>
  inline int geqrf<hmat::S_t>(int m, int n, hmat::S_t *a_gpu, int lda, hmat::S_t **tauA_gpu, hmat::S_t **Ra_gpu) {
    cusolverDnHandle_t cusolver_handle = hmat::CudaManager::getInstance().getCusolverHandle();
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));
    // Compute the size of the workspace and allocate on the GPU
    int size_workspace_geqrf_a = 0;
    CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(cusolver_handle, m, n, a_gpu, lda, &size_workspace_geqrf_a));  
    float* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, size_workspace_geqrf_a * sizeof(float)));
    CUDA_CHECK(cudaMalloc(tauA_gpu, n * sizeof(float)));
    // compute the QR facto
    CUSOLVER_CHECK(cusolverDnSgeqrf(cusolver_handle, m, n, a_gpu, lda, *tauA_gpu, workspace, size_workspace_geqrf_a, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {printf("Erreur dans la factorisation QR de a. Code : %d\n", info);}
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    // Copy R factor in a separate array
    CUDA_CHECK(cudaMalloc(Ra_gpu, n * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(*Ra_gpu, 0, n * n * sizeof(float)));
    for (int j = 0; j < n; ++j) {
      CUBLAS_CHECK(cublasScopy(cublas_handle, j + 1, &a_gpu[j * m], 1, *Ra_gpu + j * n, 1));
    }
    return 0;
  }

  template <>
  inline int geqrf<hmat::D_t>(int m, int n, hmat::D_t *a_gpu, int lda, hmat::D_t **tauA_gpu, hmat::D_t **Ra_gpu) {
    cusolverDnHandle_t cusolver_handle = hmat::CudaManager::getInstance().getCusolverHandle();
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));
    // Compute the size of the workspace and allocate on the GPU
    int size_workspace_geqrf_a = 0;
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolver_handle, m, n, a_gpu, lda, &size_workspace_geqrf_a));  
    double* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, size_workspace_geqrf_a * sizeof(double)));
    CUDA_CHECK(cudaMalloc(tauA_gpu, n * sizeof(double)));
    // compute the QR facto
    CUSOLVER_CHECK(cusolverDnDgeqrf(cusolver_handle, m, n, a_gpu, lda, *tauA_gpu, workspace, size_workspace_geqrf_a, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {printf("Erreur dans la factorisation QR de a. Code : %d\n", info);}
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    // Copy R factor in a separate array
    CUDA_CHECK(cudaMalloc(Ra_gpu, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemset(*Ra_gpu, 0, n * n * sizeof(double)));
    for (int j = 0; j < n; ++j) {
      CUBLAS_CHECK(cublasDcopy(cublas_handle, j + 1, &a_gpu[j * m], 1, *Ra_gpu + j * n, 1));
    }
    return 0;
  }

  template <>
  inline int geqrf<hmat::C_t>(int m, int n, hmat::C_t *a_gpu, int lda, hmat::C_t **tauA_gpu, hmat::C_t **Ra_gpu) {
    cusolverDnHandle_t cusolver_handle = hmat::CudaManager::getInstance().getCusolverHandle();
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));
    // Compute the size of the workspace and allocate on the GPU
    int size_workspace_geqrf_a = 0;
    CUSOLVER_CHECK(cusolverDnCgeqrf_bufferSize(cusolver_handle, m, n, reinterpret_cast<cuComplex*>(a_gpu), lda, &size_workspace_geqrf_a));  
    hmat::C_t* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, size_workspace_geqrf_a * sizeof(hmat::C_t)));
    CUDA_CHECK(cudaMalloc(tauA_gpu, n * sizeof(hmat::C_t)));
    // compute the QR facto
    CUSOLVER_CHECK(cusolverDnCgeqrf(cusolver_handle, m, n, reinterpret_cast<cuComplex*>(a_gpu), lda, reinterpret_cast<cuComplex*>(*tauA_gpu), reinterpret_cast<cuComplex*>(workspace), size_workspace_geqrf_a, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {printf("Erreur dans la factorisation QR de a. Code : %d\n", info);}
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    // Copy R factor in a separate array
    CUDA_CHECK(cudaMalloc(Ra_gpu, n * n * sizeof(hmat::C_t)));
    CUDA_CHECK(cudaMemset(*Ra_gpu, 0, n * n * sizeof(hmat::C_t)));
    for (int j = 0; j < n; ++j) {
      CUBLAS_CHECK(cublasCcopy(cublas_handle, j + 1, reinterpret_cast<cuComplex*>(a_gpu) + j * m, 1, reinterpret_cast<cuComplex*>(*Ra_gpu) + j * n, 1));
    }
    return 0;
  }

  template <>
  inline int geqrf<hmat::Z_t>(int m, int n, hmat::Z_t *a_gpu, int lda, hmat::Z_t **tauA_gpu, hmat::Z_t **Ra_gpu) {
    cusolverDnHandle_t cusolver_handle = hmat::CudaManager::getInstance().getCusolverHandle();
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));
    // Compute the size of the workspace and allocate on the GPU
    int size_workspace_geqrf_a = 0;
    CUSOLVER_CHECK(cusolverDnZgeqrf_bufferSize(cusolver_handle, m, n, reinterpret_cast<cuDoubleComplex*>(a_gpu), lda, &size_workspace_geqrf_a));  
    hmat::Z_t* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, size_workspace_geqrf_a * sizeof(hmat::Z_t)));
    CUDA_CHECK(cudaMalloc(tauA_gpu, n * sizeof(hmat::Z_t)));
    // compute the QR facto
    CUSOLVER_CHECK(cusolverDnZgeqrf(cusolver_handle, m, n, reinterpret_cast<cuDoubleComplex*>(a_gpu), lda, reinterpret_cast<cuDoubleComplex*>(*tauA_gpu), reinterpret_cast<cuDoubleComplex*>(workspace), size_workspace_geqrf_a, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {printf("Erreur dans la factorisation QR de a. Code : %d\n", info);}
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    // Copy R factor in a separate array
    CUDA_CHECK(cudaMalloc(Ra_gpu, n * n * sizeof(hmat::Z_t)));
    CUDA_CHECK(cudaMemset(*Ra_gpu, 0, n * n * sizeof(hmat::Z_t)));
    for (int j = 0; j < n; ++j) {
      CUBLAS_CHECK(cublasZcopy(cublas_handle, j + 1, reinterpret_cast<cuDoubleComplex*>(a_gpu) + j * m, 1, reinterpret_cast<cuDoubleComplex*>(*Ra_gpu) + j * n, 1));
    }
    return 0;
  }
}
