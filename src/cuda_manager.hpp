#pragma once

#include <iostream>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cstdlib> // For exit() and EXIT_FAILURE
#include "data_types.hpp"

// Helper macro to check for library status errors
#define CUDA_CHECK(call)						\
  do {									\
    auto status = call;							\
    if ((int)status != (int)cudaSuccess) { /* This is 0 for both libs */ \
      std::cerr << "CUDA Error " << status << " (" << cudaGetErrorName(status) << ": " << cudaGetErrorString(status) << ") at " << __FILE__ << ":" << __LINE__ << std::endl;   \
     /* In a real app, you might throw an exception */                 \
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)
// Helper macro to check for library status errors
#define CUBLAS_CHECK(call)						\
  do {									\
    auto status = call;							\
    if ((int)status != (int)CUBLAS_STATUS_SUCCESS) { /* This is 0 for both libs */ \
      std::cerr << "cuBLAS Error " << status << " (" << cublasGetStatusName(status) << ": " << cublasGetStatusString(status) << ") at " << __FILE__ << ":" << __LINE__ << std::endl;   \
      /* In a real app, you might throw an exception */                 \
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)
// Helper macro to check for library status errors
#define CUSOLVER_CHECK(call)						\
  do {									\
    auto status = call;							\
    if ((int)status != (int)CUSOLVER_STATUS_SUCCESS) { /* This is 0 for both libs */ \
      std::cerr << "cuSOLVER Error " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
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

    int getCudaDeviceCount() const {
      return cuda_device_count_;
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
      cudaError_t error_code = cudaGetDeviceCount(&cuda_device_count_);
      if (error_code != cudaSuccess) cuda_device_count_=0;
      if (cuda_device_count_) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle_));
        std::cout << "--> CUDA handles initialized successfully (#GPU=" << cuda_device_count_ << ")." << std::endl;
      } else {
        std::cout << "--> No CUDA devices..." << std::endl;
      }
    }

    /**
     * @brief Private destructor for automatic cleanup.
     *
     * This is called automatically when the program exits, ensuring that
     * all acquired resources are released properly.
     */
    ~CudaManager() {
      if (cuda_device_count_) {
        if (cusolver_handle_) {
	        CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle_));
        }
        if (cublas_handle_) {
	        CUBLAS_CHECK(cublasDestroy(cublas_handle_));
        }
        std::cout << "--> CUDA handles destroyed successfully." << std::endl;
      }
    }

    int cuda_device_count_ = 0;
    // Member variables holding the unique handles
    cublasHandle_t   cublas_handle_   = nullptr;
    cusolverDnHandle_t cusolver_handle_ = nullptr;
  };

}  // end namespace hmat

namespace proxy_cuda {

  // GEQRF computes a QR factorization of a m-by-n matrix A = Q * R.
  // A is overwritten by Q, Ra and tau are returned in arrays allocated in this routine.
  // https://docs.nvidia.com/cuda/archive/12.6.0/cusolver/index.html#cusolverdn-t-geqrf
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
    hmat::S_t* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, sizeof(hmat::S_t) * size_workspace_geqrf_a));
    CUDA_CHECK(cudaMalloc(tauA_gpu, sizeof(hmat::S_t) * n));
    // compute the QR facto
    CUSOLVER_CHECK(cusolverDnSgeqrf(cusolver_handle, m, n, a_gpu, lda, *tauA_gpu, workspace, size_workspace_geqrf_a, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0)
      std::cerr << "Error in QR factorization of A. Code : " << info << std::endl;
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    // Copy R factor in a separate array
    CUDA_CHECK(cudaMalloc(Ra_gpu, sizeof(hmat::S_t) * n * n));
    CUDA_CHECK(cudaMemset(*Ra_gpu, 0, sizeof(hmat::S_t) * n * n));
    for (int j = 0; j < n; ++j) {
      CUBLAS_CHECK(cublasScopy(cublas_handle, j + 1, &a_gpu[j * m], 1, *Ra_gpu + j * n, 1));
    }
    return info;
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
    hmat::D_t* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, sizeof(hmat::D_t) * size_workspace_geqrf_a));
    CUDA_CHECK(cudaMalloc(tauA_gpu, sizeof(hmat::D_t) * n));
    // compute the QR facto
    CUSOLVER_CHECK(cusolverDnDgeqrf(cusolver_handle, m, n, a_gpu, lda, *tauA_gpu, workspace, size_workspace_geqrf_a, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0)
      std::cerr << "Error in QR factorization of A. Code : " << info << std::endl;
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    // Copy R factor in a separate array
    CUDA_CHECK(cudaMalloc(Ra_gpu, sizeof(hmat::D_t) * n * n));
    CUDA_CHECK(cudaMemset(*Ra_gpu, 0, sizeof(hmat::D_t) * n * n));
    for (int j = 0; j < n; ++j) {
      CUBLAS_CHECK(cublasDcopy(cublas_handle, j + 1, &a_gpu[j * m], 1, *Ra_gpu + j * n, 1));
    }
    return info;
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
    CUDA_CHECK(cudaMalloc(&workspace, sizeof(hmat::C_t) * size_workspace_geqrf_a));
    CUDA_CHECK(cudaMalloc(tauA_gpu, sizeof(hmat::C_t) * n));
    // compute the QR facto
    CUSOLVER_CHECK(cusolverDnCgeqrf(cusolver_handle, m, n, reinterpret_cast<cuComplex*>(a_gpu), lda, reinterpret_cast<cuComplex*>(*tauA_gpu), reinterpret_cast<cuComplex*>(workspace), size_workspace_geqrf_a, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0)
      std::cerr << "Error in QR factorization of A. Code : " << info << std::endl;
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    // Copy R factor in a separate array
    CUDA_CHECK(cudaMalloc(Ra_gpu, sizeof(hmat::C_t) * n * n));
    CUDA_CHECK(cudaMemset(*Ra_gpu, 0, sizeof(hmat::C_t) * n * n));
    for (int j = 0; j < n; ++j) {
      CUBLAS_CHECK(cublasCcopy(cublas_handle, j + 1, reinterpret_cast<cuComplex*>(a_gpu) + j * m, 1, reinterpret_cast<cuComplex*>(*Ra_gpu) + j * n, 1));
    }
    return info;
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
    CUDA_CHECK(cudaMalloc(&workspace, sizeof(hmat::Z_t) * size_workspace_geqrf_a));
    CUDA_CHECK(cudaMalloc(tauA_gpu, sizeof(hmat::Z_t) * n));
    // compute the QR facto
    CUSOLVER_CHECK(cusolverDnZgeqrf(cusolver_handle, m, n, reinterpret_cast<cuDoubleComplex*>(a_gpu), lda, reinterpret_cast<cuDoubleComplex*>(*tauA_gpu), reinterpret_cast<cuDoubleComplex*>(workspace), size_workspace_geqrf_a, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0)
      std::cerr << "Error in QR factorization of A. Code : " << info << std::endl;
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    // Copy R factor in a separate array
    CUDA_CHECK(cudaMalloc(Ra_gpu, sizeof(hmat::Z_t) * n * n));
    CUDA_CHECK(cudaMemset(*Ra_gpu, 0, sizeof(hmat::Z_t) * n * n));
    for (int j = 0; j < n; ++j) {
      CUBLAS_CHECK(cublasZcopy(cublas_handle, j + 1, reinterpret_cast<cuDoubleComplex*>(a_gpu) + j * m, 1, reinterpret_cast<cuDoubleComplex*>(*Ra_gpu) + j * n, 1));
    }
    return info;
  }

  // TRMM computes a matrix-matrix multiplication between a triangular matrix and a regular matrix
  // https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html#cublas-t-trmm
  template<typename T>
  void trmm(const char side, const char uplo, const char trans, const char diag,
	    const int m, const int n, const T& alpha, const T* a, const int lda,
	    const T* b, int ldb, T* c, const int ldc);

  template <>
  inline void trmm<hmat::S_t>(const char side, const char uplo, const char trans, const char diag,
	    const int m, const int n, const hmat::S_t& alpha, const hmat::S_t* a, const int lda,
	    const hmat::S_t* b, const int ldb, hmat::S_t* c, const int ldc) {
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    const cublasSideMode_t s = (side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    const cublasFillMode_t u = (uplo == 'U' ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER);
    const cublasOperation_t t = (trans == 'C' ? CUBLAS_OP_C : (trans == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    const cublasDiagType_t d = (diag == 'N' ?  CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT );
    cublasStrmm(cublas_handle, s, u, t, d, m, n, &alpha, a, lda, b, ldb, c, ldc);
  }
  template <>
  inline void trmm<hmat::D_t>(const char side, const char uplo, const char trans, const char diag,
	    const int m, const int n, const hmat::D_t& alpha, const hmat::D_t* a, const int lda,
	    const hmat::D_t* b, const int ldb, hmat::D_t* c, const int ldc) {
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    const cublasSideMode_t s = (side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    const cublasFillMode_t u = (uplo == 'U' ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER);
    const cublasOperation_t t = (trans == 'C' ? CUBLAS_OP_C : (trans == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    const cublasDiagType_t d = (diag == 'N' ?  CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT );
    cublasDtrmm(cublas_handle, s, u, t, d, m, n, &alpha, a, lda, b, ldb, c, ldc);
  }
  template <>
  inline void trmm<hmat::C_t>(const char side, const char uplo, const char trans, const char diag,
	    const int m, const int n, const hmat::C_t& alpha, const hmat::C_t* a, const int lda,
	    const hmat::C_t* b, const int ldb, hmat::C_t* c, const int ldc) {
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    const cublasSideMode_t s = (side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    const cublasFillMode_t u = (uplo == 'U' ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER);
    const cublasOperation_t t = (trans == 'C' ? CUBLAS_OP_C : (trans == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    const cublasDiagType_t d = (diag == 'N' ?  CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT );
    cublasCtrmm(cublas_handle, s, u, t, d, m, n, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a), lda, reinterpret_cast<const cuComplex*>(b), ldb, reinterpret_cast<cuComplex*>(c), ldc);
  }
  template <>
  inline void trmm<hmat::Z_t>(const char side, const char uplo, const char trans, const char diag,
	    const int m, const int n, const hmat::Z_t& alpha, const hmat::Z_t* a, const int lda,
	    const hmat::Z_t* b, const int ldb, hmat::Z_t* c, const int ldc) {
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    const cublasSideMode_t s = (side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    const cublasFillMode_t u = (uplo == 'U' ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER);
    const cublasOperation_t t = (trans == 'C' ? CUBLAS_OP_C : (trans == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    const cublasDiagType_t d = (diag == 'N' ?  CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT );
    cublasZtrmm(cublas_handle, s, u, t, d, m, n, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a), lda, reinterpret_cast<const cuDoubleComplex*>(b), ldb, reinterpret_cast<cuDoubleComplex*>(c), ldc);
  }

  // GESVD computes the Singular Value Decomposition of a matrix A = U.S.VT
  // A is overwritten. U,S,VT are returned in arrays allocated in this routine.
  // Note that gesvd returns V^T in real and V^H in complex, not V.
  // https://docs.nvidia.com/cuda/archive/12.6.0/cusolver/index.html#cusolverdn-t-gesvd

  template <typename T>
  int gesvd(char jobu, char jobvt, int m, int n, T *a, int lda,
	    typename hmat::Types<T>::real **s, T **u, T **vt);

  template <>
  inline int gesvd<hmat::S_t>(char jobu, char jobvt, int m, int n, hmat::S_t *a,
			      int lda, hmat::S_t **s, hmat::S_t **u, hmat::S_t **vt) {
    cusolverDnHandle_t cusolver_handle = hmat::CudaManager::getInstance().getCusolverHandle();
    int size_workspace_svd = 0;
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(cusolver_handle, m, n, &size_workspace_svd));
    hmat::S_t* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, sizeof(hmat::S_t) * size_workspace_svd));
    *u = nullptr;
    CUDA_CHECK(cudaMalloc(u, sizeof(hmat::S_t) * m * m));
    *s = nullptr;
    CUDA_CHECK(cudaMalloc(s, sizeof(hmat::S_t) * std::min(m,n)));
    *vt = nullptr;
    CUDA_CHECK(cudaMalloc(vt, sizeof(hmat::S_t) * n * n));
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnSgesvd(cusolver_handle, jobu, jobvt, m, n, a, m, *s, *u, m, *vt, n, workspace, size_workspace_svd, nullptr, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info < 0)
      std::cerr << "Error: parameter " << -info << "is invalid." << std::endl;
    else if (info > 0)
      std::cerr << "Warning: SVD has not converged. Return code: " << info << std::endl;
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    return info;
  }
  template <>
  inline int gesvd<hmat::D_t>(char jobu, char jobvt, int m, int n, hmat::D_t *a,
			      int lda, hmat::D_t **s, hmat::D_t **u, hmat::D_t **vt) {
    cusolverDnHandle_t cusolver_handle = hmat::CudaManager::getInstance().getCusolverHandle();
    int size_workspace_svd = 0;
    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolver_handle, m, n, &size_workspace_svd));
    hmat::D_t* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, sizeof(hmat::D_t) * size_workspace_svd));
    *u = nullptr;
    CUDA_CHECK(cudaMalloc(u, sizeof(hmat::D_t) * m * m));
    *s = nullptr;
    CUDA_CHECK(cudaMalloc(s, sizeof(hmat::D_t) * std::min(m,n)));
    *vt = nullptr;
    CUDA_CHECK(cudaMalloc(vt, sizeof(hmat::D_t) * n * n));
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDgesvd(cusolver_handle, jobu, jobvt, m, n, a, m, *s, *u, m, *vt, n, workspace, size_workspace_svd, nullptr, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info < 0)
      std::cerr << "Error: parameter " << -info << "is invalid." << std::endl;
    else if (info > 0)
      std::cerr << "Warning: SVD has not converged. Return code: " << info << std::endl;
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    return info;
  }
  template <>
  inline int gesvd<hmat::C_t>(char jobu, char jobvt, int m, int n, hmat::C_t *a,
			      int lda, hmat::S_t **s, hmat::C_t **u, hmat::C_t **vt) {
    cusolverDnHandle_t cusolver_handle = hmat::CudaManager::getInstance().getCusolverHandle();
    int size_workspace_svd = 0;
    CUSOLVER_CHECK(cusolverDnCgesvd_bufferSize(cusolver_handle, m, n, &size_workspace_svd));
    hmat::C_t* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, sizeof(hmat::C_t) * size_workspace_svd));
    *u = nullptr;
    CUDA_CHECK(cudaMalloc(u, sizeof(hmat::C_t) * m * m));
    *s = nullptr;
    CUDA_CHECK(cudaMalloc(s, sizeof(hmat::S_t) * std::min(m,n)));
    *vt = nullptr;
    CUDA_CHECK(cudaMalloc(vt, sizeof(hmat::C_t) * n * n));
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnCgesvd(cusolver_handle, jobu, jobvt, m, n, reinterpret_cast<cuComplex*>(a), m, *s, reinterpret_cast<cuComplex*>(*u), m, reinterpret_cast<cuComplex*>(*vt), n, reinterpret_cast<cuComplex*>(workspace), size_workspace_svd, nullptr, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info < 0)
      std::cerr << "Error: parameter " << -info << "is invalid." << std::endl;
    else if (info > 0)
      std::cerr << "Warning: SVD has not converged. Return code: " << info << std::endl;
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    return info;
  }
  template <>
  inline int gesvd<hmat::Z_t>(char jobu, char jobvt, int m, int n, hmat::Z_t *a,
			      int lda, hmat::D_t **s, hmat::Z_t **u, hmat::Z_t **vt) {
    cusolverDnHandle_t cusolver_handle = hmat::CudaManager::getInstance().getCusolverHandle();
    int size_workspace_svd = 0;
    CUSOLVER_CHECK(cusolverDnZgesvd_bufferSize(cusolver_handle, m, n, &size_workspace_svd));
    hmat::Z_t* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, sizeof(hmat::Z_t) * size_workspace_svd));
    *u = nullptr;
    CUDA_CHECK(cudaMalloc(u, sizeof(hmat::Z_t) * m * m));
    *s = nullptr;
    CUDA_CHECK(cudaMalloc(s, sizeof(hmat::D_t) * std::min(m,n)));
    *vt = nullptr;
    CUDA_CHECK(cudaMalloc(vt, sizeof(hmat::Z_t) * n * n));
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnZgesvd(cusolver_handle, jobu, jobvt, m, n, reinterpret_cast<cuDoubleComplex*>(a), m, *s, reinterpret_cast<cuDoubleComplex*>(*u), m, reinterpret_cast<cuDoubleComplex*>(*vt), n, reinterpret_cast<cuDoubleComplex*>(workspace), size_workspace_svd, nullptr, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info < 0)
      std::cerr << "Error: parameter " << -info << "is invalid." << std::endl;
    else if (info > 0)
      std::cerr << "Warning: SVD has not converged. Return code: " << info << std::endl;
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    return info;
  }

  // GEAM computes a matrix-matrix addition/transposition
  // C is allocated and returned by this routine (unless *c=a or *c=b, for in-place variants)
  // https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html#cublas-t-geam
  template<typename T>
  void geam(const char transa, const char transb,
	    const int m, const int n, const T& alpha, const T* a, const int lda,
	    const T& beta, const T* b, int ldb, T** c, int ldc);

  template <>
  inline void geam<hmat::S_t>(const char transa, const char transb,
	    const int m, const int n, const hmat::S_t& alpha, const hmat::S_t* a, const int lda,
	    const hmat::S_t& beta, const hmat::S_t* b, int ldb, hmat::S_t** c, int ldc) {
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    if ( (*c != a && *c != b) || (*c == nullptr) )
      CUDA_CHECK(cudaMalloc(c, sizeof(hmat::S_t) * m * n));
    const cublasOperation_t ta = (transa == 'C' ? CUBLAS_OP_C : (transa == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    const cublasOperation_t tb = (transb == 'C' ? CUBLAS_OP_C : (transb == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    CUBLAS_CHECK(cublasSgeam(cublas_handle, ta, tb, m, n, &alpha, a, lda, &beta, b, ldb, *c, ldc));
  }
  template <>
  inline void geam<hmat::D_t>(const char transa, const char transb,
	    const int m, const int n, const hmat::D_t& alpha, const hmat::D_t* a, const int lda,
	    const hmat::D_t& beta, const hmat::D_t* b, int ldb, hmat::D_t** c, int ldc) {
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    if ( (*c != a && *c != b) || (*c == nullptr) )
      CUDA_CHECK(cudaMalloc(c, sizeof(hmat::D_t) * m * n));
    const cublasOperation_t ta = (transa == 'C' ? CUBLAS_OP_C : (transa == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    const cublasOperation_t tb = (transb == 'C' ? CUBLAS_OP_C : (transb == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    CUBLAS_CHECK(cublasDgeam(cublas_handle, ta, tb, m, n, &alpha, a, lda, &beta, b, ldb, *c, ldc));
  }
  template <>
  inline void geam<hmat::C_t>(const char transa, const char transb,
	    const int m, const int n, const hmat::C_t& alpha, const hmat::C_t* a, const int lda,
	    const hmat::C_t& beta, const hmat::C_t* b, int ldb, hmat::C_t** c, int ldc) {
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    if ( (*c != a && *c != b) || (*c == nullptr) )
      CUDA_CHECK(cudaMalloc(c, sizeof(hmat::C_t) * m * n));
    const cublasOperation_t ta = (transa == 'C' ? CUBLAS_OP_C : (transa == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    const cublasOperation_t tb = (transb == 'C' ? CUBLAS_OP_C : (transb == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    CUBLAS_CHECK(cublasCgeam(cublas_handle, ta, tb, m, n, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a), lda, reinterpret_cast<const cuComplex*>(&beta), reinterpret_cast<const cuComplex*>(b), ldb, reinterpret_cast<cuComplex*>(*c), ldc));
  }
  template <>
  inline void geam<hmat::Z_t>(const char transa, const char transb,
	    const int m, const int n, const hmat::Z_t& alpha, const hmat::Z_t* a, const int lda,
	    const hmat::Z_t& beta, const hmat::Z_t* b, int ldb, hmat::Z_t** c, int ldc) {
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    if ( (*c != a && *c != b) || (*c == nullptr) )
      CUDA_CHECK(cudaMalloc(c, sizeof(hmat::Z_t) * m * n));
    const cublasOperation_t ta = (transa == 'C' ? CUBLAS_OP_C : (transa == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    const cublasOperation_t tb = (transb == 'C' ? CUBLAS_OP_C : (transb == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    CUBLAS_CHECK(cublasZgeam(cublas_handle, ta, tb, m, n, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a), lda, reinterpret_cast<const cuDoubleComplex*>(&beta), reinterpret_cast<const cuDoubleComplex*>(b), ldb, reinterpret_cast<cuDoubleComplex*>(*c), ldc));
  }

  // DGMM computes a matrix-matrix multiplication with a diagonal matrix
  // https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html#id10
template<typename T>
  void dgmm(const char side, const int m, const int n, const T* a, const int lda,
	    const T* x, int incx, T* c, int ldc);
  template <>
  inline void dgmm<hmat::S_t>(const char side, const int m, const int n, const hmat::S_t* a, const int lda,
	    const hmat::S_t* x, int incx, hmat::S_t* c, int ldc) {
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    const cublasSideMode_t s = (side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    CUBLAS_CHECK(cublasSdgmm(cublas_handle, s, m, n, a, lda, x, incx, c, ldc));
  }
  template <>
  inline void dgmm<hmat::D_t>(const char side, const int m, const int n, const hmat::D_t* a, const int lda,
	    const hmat::D_t* x, int incx, hmat::D_t* c, int ldc) {
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    const cublasSideMode_t s = (side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    CUBLAS_CHECK(cublasDdgmm(cublas_handle, s, m, n, a, lda, x, incx, c, ldc));
  }
  template <>
  inline void dgmm<hmat::C_t>(const char side, const int m, const int n, const hmat::C_t* a, const int lda,
	    const hmat::C_t* x, int incx, hmat::C_t* c, int ldc) {
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    const cublasSideMode_t s = (side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    CUBLAS_CHECK(cublasCdgmm(cublas_handle, s, m, n, reinterpret_cast<const cuComplex*>(a), lda, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<cuComplex*>(c), ldc));
  }
  template <>
  inline void dgmm<hmat::Z_t>(const char side, const int m, const int n, const hmat::Z_t* a, const int lda,
	    const hmat::Z_t* x, int incx, hmat::Z_t* c, int ldc) {
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    const cublasSideMode_t s = (side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    CUBLAS_CHECK(cublasZdgmm(cublas_handle, s, m, n, reinterpret_cast<const cuDoubleComplex*>(a), lda, reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<cuDoubleComplex*>(c), ldc));
  }

  // ORMQR/UNMQR compute the product of a Q matrix (from QR decomposition) with another matrix u (k x n)
  // The output matrix qu (m x n) is allocated and returned by this routine.
  // ORMQR for real orthogonal matrices, UNMQR from complex unitary matrices
  // https://docs.nvidia.com/cuda/archive/12.6.0/cusolver/index.html#cusolverdn-t-ormqr
  template <typename T>
  int or_un_mqr(char side, char trans, int m, int n, int k,
      const T *q, int ldq,
	    const T *tau,
      T *u, int ldu,
      T **qu);

  template <>
  inline int or_un_mqr<hmat::S_t>(char side, char trans, int m, int n, int k,
    const hmat::S_t *q, int ldq, const hmat::S_t *tau, hmat::S_t *u, int ldu, hmat::S_t **qu) {
    cusolverDnHandle_t cusolver_handle = hmat::CudaManager::getInstance().getCusolverHandle();
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    const cublasSideMode_t s = (side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    const cublasOperation_t t = (trans == 'C' ? CUBLAS_OP_C : (trans == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));

    // Copy u at the top of qu
    CUDA_CHECK(cudaMalloc(qu, sizeof(hmat::S_t) * m * n));
    CUDA_CHECK(cudaMemset(*qu, 0, sizeof(hmat::S_t) * m * n));
    for (int j = 0; j < n; ++j) {
      // u is k x k (leading dimension k) *qu is m x k (leading dimension m)
      CUBLAS_CHECK(cublasScopy(cublas_handle, k, &u[j * k], 1, *qu + j * m, 1));                      
    }
    // Construction of the panel qu (m x n) <- q (m x m) * u (m x n) with k reflectors in q
    hmat::S_t* work_sormqr = nullptr;
    int worksize_sormqr = 0;
    CUSOLVER_CHECK(cusolverDnSormqr_bufferSize(cusolver_handle, s, t, m, n, k, q, m, tau, *qu, m, &worksize_sormqr));
    CUDA_CHECK(cudaMalloc(&work_sormqr, sizeof(hmat::S_t) * worksize_sormqr));
    // Multiply q by u
    CUSOLVER_CHECK(cusolverDnSormqr(cusolver_handle, s, t, m, n, k, q, m, tau, *qu, m, work_sormqr, worksize_sormqr, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0)
      std::cerr << "Erreur cusolverDnSormqr : info = " << info << std::endl;
    CUDA_CHECK(cudaFree(work_sormqr));
    work_sormqr = nullptr;
    CUDA_CHECK(cudaFree(info_GPU));
    
    return 0;
  }
  template <>
  inline int or_un_mqr<hmat::D_t>(char side, char trans, int m, int n, int k,
    const hmat::D_t *q, int ldq, const hmat::D_t *tau, hmat::D_t *u, int ldu, hmat::D_t **qu) {
    cusolverDnHandle_t cusolver_handle = hmat::CudaManager::getInstance().getCusolverHandle();
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    const cublasSideMode_t s = (side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    const cublasOperation_t t = (trans == 'C' ? CUBLAS_OP_C : (trans == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));

    // Copy u at the top of qu
    CUDA_CHECK(cudaMalloc(qu, sizeof(hmat::D_t) * m * n));
    CUDA_CHECK(cudaMemset(*qu, 0, sizeof(hmat::D_t) * m * n));
    for (int j = 0; j < n; ++j) {
      // u is k x k (leading dimension k) *qu is m x k (leading dimension m)
      CUBLAS_CHECK(cublasDcopy(cublas_handle, k, &u[j * k], 1, *qu + j * m, 1));                      
    }
    // Construction of the panel qu (m x n) <- q (m x m) * u (m x n) with k reflectors in q
    hmat::D_t* work_sormqr = nullptr;
    int worksize_sormqr = 0;
    CUSOLVER_CHECK(cusolverDnDormqr_bufferSize(cusolver_handle, s, t, m, n, k, q, m, tau, *qu, m, &worksize_sormqr));
    CUDA_CHECK(cudaMalloc(&work_sormqr, sizeof(hmat::D_t) * worksize_sormqr));
    // Multiply q by u
    CUSOLVER_CHECK(cusolverDnDormqr(cusolver_handle, s, t, m, n, k, q, m, tau, *qu, m, work_sormqr, worksize_sormqr, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0)
      std::cerr << "Erreur cusolverDnDormqr : info = " << info << std::endl;
    CUDA_CHECK(cudaFree(work_sormqr));
    work_sormqr = nullptr;
    CUDA_CHECK(cudaFree(info_GPU));
    
    return 0;
  }
  template <>
  inline int or_un_mqr<hmat::C_t>(char side, char trans, int m, int n, int k,
    const hmat::C_t *q, int ldq, const hmat::C_t *tau, hmat::C_t *u, int ldu, hmat::C_t **qu) {
    cusolverDnHandle_t cusolver_handle = hmat::CudaManager::getInstance().getCusolverHandle();
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    const cublasSideMode_t s = (side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    const cublasOperation_t t = (trans == 'C' ? CUBLAS_OP_C : (trans == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));

    // Copy u at the top of qu
    CUDA_CHECK(cudaMalloc(qu, sizeof(hmat::C_t) * m * n ));
    CUDA_CHECK(cudaMemset(*qu, 0, sizeof(hmat::C_t) * m * n));
    for (int j = 0; j < n; ++j) {
      // u is k x k (leading dimension k) *qu is m x k (leading dimension m)
      CUBLAS_CHECK(cublasCcopy(cublas_handle, k, &reinterpret_cast<const cuComplex*>(u)[j * k], 1, reinterpret_cast<cuComplex*>(*qu) + j * m, 1));                      
    }
    // Construction of the panel qu (m x n) <- q (m x m) * u (m x n) with k reflectors in q
    hmat::C_t* work_sormqr = nullptr;
    int worksize_sormqr = 0;
    CUSOLVER_CHECK(cusolverDnCunmqr_bufferSize(cusolver_handle, s, t, m, n, k, reinterpret_cast<const cuComplex*>(q), m, reinterpret_cast<const cuComplex*>(tau), reinterpret_cast<cuComplex*>(*qu), m, &worksize_sormqr));
    CUDA_CHECK(cudaMalloc(&work_sormqr, sizeof(hmat::C_t) * worksize_sormqr));
    // Multiply q by u
    CUSOLVER_CHECK(cusolverDnCunmqr(cusolver_handle, s, t, m, n, k, reinterpret_cast<const cuComplex*>(q), m, reinterpret_cast<const cuComplex*>(tau), reinterpret_cast<cuComplex*>(*qu), m, reinterpret_cast<cuComplex*>(work_sormqr), worksize_sormqr, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0)
      std::cerr << "Erreur cusolverDnCunmqr : info = " << info << std::endl;
    CUDA_CHECK(cudaFree(work_sormqr));
    work_sormqr = nullptr;
    CUDA_CHECK(cudaFree(info_GPU));
    
    return 0;
  }
  template <>
  inline int or_un_mqr<hmat::Z_t>(char side, char trans, int m, int n, int k,
    const hmat::Z_t *q, int ldq, const hmat::Z_t *tau, hmat::Z_t *u, int ldu, hmat::Z_t **qu) {
    cusolverDnHandle_t cusolver_handle = hmat::CudaManager::getInstance().getCusolverHandle();
    cublasHandle_t cublas_handle = hmat::CudaManager::getInstance().getCublasHandle();
    const cublasSideMode_t s = (side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
    const cublasOperation_t t = (trans == 'C' ? CUBLAS_OP_C : (trans == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));

    // Copy u at the top of qu
    CUDA_CHECK(cudaMalloc(qu, sizeof(hmat::Z_t) * m * n));
    CUDA_CHECK(cudaMemset(*qu, 0, sizeof(hmat::Z_t) * m * n));
    for (int j = 0; j < n; ++j) {
      // u is k x k (leading dimension k) *qu is m x k (leading dimension m)
      CUBLAS_CHECK(cublasZcopy(cublas_handle, k, &reinterpret_cast<const cuDoubleComplex*>(u)[j * k], 1, reinterpret_cast<cuDoubleComplex*>(*qu) + j * m, 1));                      
    }
    // Construction of the panel qu (m x n) <- q (m x m) * u (m x n) with k reflectors in q
    hmat::Z_t* work_sormqr = nullptr;
    int worksize_sormqr = 0;
    CUSOLVER_CHECK(cusolverDnZunmqr_bufferSize(cusolver_handle, s, t, m, n, k, reinterpret_cast<const cuDoubleComplex*>(q), m, reinterpret_cast<const cuDoubleComplex*>(tau), reinterpret_cast<cuDoubleComplex*>(*qu), m, &worksize_sormqr));
    CUDA_CHECK(cudaMalloc(&work_sormqr, sizeof(hmat::Z_t) * worksize_sormqr));
    // Multiply q by u
    CUSOLVER_CHECK(cusolverDnZunmqr(cusolver_handle, s, t, m, n, k, reinterpret_cast<const cuDoubleComplex*>(q), m, reinterpret_cast<const cuDoubleComplex*>(tau), reinterpret_cast<cuDoubleComplex*>(*qu), m, reinterpret_cast<cuDoubleComplex*>(work_sormqr), worksize_sormqr, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0)
      std::cerr << "Erreur cusolverDnZunmqr : info = " << info << std::endl;
    CUDA_CHECK(cudaFree(work_sormqr));
    work_sormqr = nullptr;
    CUDA_CHECK(cudaFree(info_GPU));
    
    return 0;
  }
}  // end namespace proxy_cuda

template<typename T> void launch_FindKAndSqrtAll(T* S_gpu, double epsilon, int old_rank, int* newK_gpu);
