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
