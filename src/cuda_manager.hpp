#pragma once

#include <iostream>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdlib> // For exit() and EXIT_FAILURE
#include "data_types.hpp"

// Helper macro to check for library status errors
#define CUDA_CHECK(call)						\
  do {									\
    auto status = call;							\
    if ((int)status != (int)cudaSuccess) { /* This is 0 for both libs */ \
      fprintf(stderr, "CUDA Error %d (%s: %s) at %s:%d\n",		\
	      status, cudaGetErrorName(status), cudaGetErrorString(status), __FILE__, __LINE__); \
      /* In a real app, you might throw an exception */                 \
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)
// Helper macro to check for library status errors
#define CUBLAS_CHECK(call)						\
  do {									\
    auto status = call;							\
    if ((int)status != (int)CUBLAS_STATUS_SUCCESS) { /* This is 0 for both libs */ \
      fprintf(stderr, "cuBLAS Error %d (%s: %s) at %s:%d\n",		\
	      status, cublasGetStatusName(status), cublasGetStatusString(status), __FILE__, __LINE__); \
      /* In a real app, you might throw an exception */                 \
      exit(EXIT_FAILURE);						\
    }									\
  } while (0)
// Helper macro to check for library status errors
#define CUSOLVER_CHECK(call)						\
  do {									\
    auto status = call;							\
    if ((int)status != (int)CUSOLVER_STATUS_SUCCESS) { /* This is 0 for both libs */ \
      fprintf(stderr, "cuSOLVER Error %d at %s:%d\n",			\
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
    float* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, size_workspace_geqrf_a * sizeof(float)));
    CUDA_CHECK(cudaMalloc(tauA_gpu, n * sizeof(float)));
    // compute the QR facto
    CUSOLVER_CHECK(cusolverDnSgeqrf(cusolver_handle, m, n, a_gpu, lda, *tauA_gpu, workspace, size_workspace_geqrf_a, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {printf("Error in QR factorization of A. Code : %d\n", info);}
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    // Copy R factor in a separate array
    CUDA_CHECK(cudaMalloc(Ra_gpu, n * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(*Ra_gpu, 0, n * n * sizeof(float)));
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
    double* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, size_workspace_geqrf_a * sizeof(double)));
    CUDA_CHECK(cudaMalloc(tauA_gpu, n * sizeof(double)));
    // compute the QR facto
    CUSOLVER_CHECK(cusolverDnDgeqrf(cusolver_handle, m, n, a_gpu, lda, *tauA_gpu, workspace, size_workspace_geqrf_a, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {printf("Error in QR factorization of A. Code : %d\n", info);}
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    // Copy R factor in a separate array
    CUDA_CHECK(cudaMalloc(Ra_gpu, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemset(*Ra_gpu, 0, n * n * sizeof(double)));
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
    CUDA_CHECK(cudaMalloc(&workspace, size_workspace_geqrf_a * sizeof(hmat::C_t)));
    CUDA_CHECK(cudaMalloc(tauA_gpu, n * sizeof(hmat::C_t)));
    // compute the QR facto
    CUSOLVER_CHECK(cusolverDnCgeqrf(cusolver_handle, m, n, reinterpret_cast<cuComplex*>(a_gpu), lda, reinterpret_cast<cuComplex*>(*tauA_gpu), reinterpret_cast<cuComplex*>(workspace), size_workspace_geqrf_a, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {printf("Error in QR factorization of A. Code : %d\n", info);}
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    // Copy R factor in a separate array
    CUDA_CHECK(cudaMalloc(Ra_gpu, n * n * sizeof(hmat::C_t)));
    CUDA_CHECK(cudaMemset(*Ra_gpu, 0, n * n * sizeof(hmat::C_t)));
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
    CUDA_CHECK(cudaMalloc(&workspace, size_workspace_geqrf_a * sizeof(hmat::Z_t)));
    CUDA_CHECK(cudaMalloc(tauA_gpu, n * sizeof(hmat::Z_t)));
    // compute the QR facto
    CUSOLVER_CHECK(cusolverDnZgeqrf(cusolver_handle, m, n, reinterpret_cast<cuDoubleComplex*>(a_gpu), lda, reinterpret_cast<cuDoubleComplex*>(*tauA_gpu), reinterpret_cast<cuDoubleComplex*>(workspace), size_workspace_geqrf_a, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {printf("Error in QR factorization of A. Code : %d\n", info);}
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info_GPU));
    // Copy R factor in a separate array
    CUDA_CHECK(cudaMalloc(Ra_gpu, n * n * sizeof(hmat::Z_t)));
    CUDA_CHECK(cudaMemset(*Ra_gpu, 0, n * n * sizeof(hmat::Z_t)));
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
    // WARNING: &alpha instead of alpha for complex values
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
    // WARNING: &alpha instead of alpha for complex values
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
    float* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, size_workspace_svd * sizeof(float)));
    *u = nullptr;
    CUDA_CHECK(cudaMalloc(u, m * m * sizeof(float)));
    *s = nullptr;
    CUDA_CHECK(cudaMalloc(s, std::min(m,n) * sizeof(float)));
    *vt = nullptr;
    CUDA_CHECK(cudaMalloc(vt, n * n * sizeof(float)));
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnSgesvd(cusolver_handle, jobu, jobvt, m, n, a, m, *s, *u, m, *vt, n, workspace, size_workspace_svd, nullptr, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info < 0) {
      printf("Error: parameter %d is invalid.\n", -info);
    } else if (info>0) {
      printf("Warning: SVD has not converged. Return code: %d\n", info);
    }
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
    double* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, size_workspace_svd * sizeof(double)));
    *u = nullptr;
    CUDA_CHECK(cudaMalloc(u, m * m * sizeof(double)));
    *s = nullptr;
    CUDA_CHECK(cudaMalloc(s, std::min(m,n) * sizeof(double)));
    *vt = nullptr;
    CUDA_CHECK(cudaMalloc(vt, n * n * sizeof(double)));
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDgesvd(cusolver_handle, jobu, jobvt, m, n, a, m, *s, *u, m, *vt, n, workspace, size_workspace_svd, nullptr, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info < 0) {
      printf("Error: parameter %d is invalid.\n", -info);
    } else if (info>0) {
      printf("Warning: SVD has not converged. Return code: %d\n", info);
    }
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
    double* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, size_workspace_svd * sizeof(hmat::C_t)));
    *u = nullptr;
    CUDA_CHECK(cudaMalloc(u, m * m * sizeof(hmat::C_t)));
    *s = nullptr;
    CUDA_CHECK(cudaMalloc(s, std::min(m,n) * sizeof(hmat::S_t)));
    *vt = nullptr;
    CUDA_CHECK(cudaMalloc(vt, n * n * sizeof(hmat::C_t)));
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnCgesvd(cusolver_handle, jobu, jobvt, m, n, reinterpret_cast<cuComplex*>(a), m, *s, reinterpret_cast<cuComplex*>(*u), m, reinterpret_cast<cuComplex*>(*vt), n, reinterpret_cast<cuComplex*>(workspace), size_workspace_svd, nullptr, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info < 0) {
      printf("Error: parameter %d is invalid.\n", -info);
    } else if (info>0) {
      printf("Warning: SVD has not converged. Return code: %d\n", info);
    }
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
    double* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, size_workspace_svd * sizeof(hmat::Z_t)));
    *u = nullptr;
    CUDA_CHECK(cudaMalloc(u, m * m * sizeof(hmat::Z_t)));
    *s = nullptr;
    CUDA_CHECK(cudaMalloc(s, std::min(m,n) * sizeof(hmat::D_t)));
    *vt = nullptr;
    CUDA_CHECK(cudaMalloc(vt, n * n * sizeof(hmat::Z_t)));
    int *info_GPU = NULL;
    CUDA_CHECK(cudaMalloc(&info_GPU, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnZgesvd(cusolver_handle, jobu, jobvt, m, n, reinterpret_cast<cuDoubleComplex*>(a), m, *s, reinterpret_cast<cuDoubleComplex*>(*u), m, reinterpret_cast<cuDoubleComplex*>(*vt), n, reinterpret_cast<cuDoubleComplex*>(workspace), size_workspace_svd, nullptr, info_GPU));
    int info=0;
    CUDA_CHECK(cudaMemcpy(&info, info_GPU, sizeof(int), cudaMemcpyDeviceToHost));
    if (info < 0) {
      printf("Error: parameter %d is invalid.\n", -info);
    } else if (info>0) {
      printf("Warning: SVD has not converged. Return code: %d\n", info);
    }
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
      CUDA_CHECK(cudaMalloc(c, m * n * sizeof(hmat::S_t)));
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
      CUDA_CHECK(cudaMalloc(c, m * n * sizeof(hmat::D_t)));
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
      CUDA_CHECK(cudaMalloc(c, m * n * sizeof(hmat::C_t)));
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
      CUDA_CHECK(cudaMalloc(c, m * n * sizeof(hmat::Z_t)));
    const cublasOperation_t ta = (transa == 'C' ? CUBLAS_OP_C : (transa == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    const cublasOperation_t tb = (transb == 'C' ? CUBLAS_OP_C : (transb == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N));
    CUBLAS_CHECK(cublasZgeam(cublas_handle, ta, tb, m, n, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a), lda, reinterpret_cast<const cuDoubleComplex*>(&beta), reinterpret_cast<const cuDoubleComplex*>(b), ldb, reinterpret_cast<cuDoubleComplex*>(*c), ldc));
  }

}  // end namespace proxy_cuda
