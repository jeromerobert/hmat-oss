#pragma once

#include <iostream>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdlib> // For exit() and EXIT_FAILURE

// Helper macro to check for library status errors
#define CHECK_CUDA_STATUS(call, lib_name)				\
  do {									\
    auto status = call;							\
    if ((int)status != (int)CUBLAS_STATUS_SUCCESS) { /* This is 0 for both libs */ \
      fprintf(stderr, "Error in %s at %s:%d\n",				\
	      lib_name, __FILE__, __LINE__);				\
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
      CHECK_CUDA_STATUS(cublasCreate(&cublas_handle_), "cuBLAS");
      CHECK_CUDA_STATUS(cusolverDnCreate(&cusolver_handle_), "cuSOLVER");
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
	CHECK_CUDA_STATUS(cusolverDnDestroy(cusolver_handle_), "cuSOLVER");
      }
      if (cublas_handle_) {
	CHECK_CUDA_STATUS(cublasDestroy(cublas_handle_), "cuBLAS");
      }
      std::cout << "--> Handles destroyed successfully." << std::endl;
    }

    // Member variables holding the unique handles
    cublasHandle_t   cublas_handle_   = nullptr;
    cusolverDnHandle_t cusolver_handle_ = nullptr;
  };

}  // end namespace hmat
