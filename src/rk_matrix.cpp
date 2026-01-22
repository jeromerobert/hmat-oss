#include "config.h"
/*
  HMat-OSS (HMatrix library, open source software)

  Copyright (C) 2014-2015 Airbus Group SAS

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

  http://github.com/jeromerobert/hmat-oss
*/

#include "rk_matrix.hpp"
#include "h_matrix.hpp"
#include "cluster_tree.hpp"
#include <cstring> // memcpy
#include <cfloat> // DBL_MAX
#include "data_types.hpp"
#include "lapack_operations.hpp"
#include "blas_overloads.hpp"
#include "lapack_overloads.hpp"
#include "common/context.hpp"
#include "common/my_assert.h"
#include "common/timeline.hpp"
#include "lapack_exception.hpp"
#include <iomanip>

#include <algorithm>
#include <chrono>

#ifdef HAVE_CUDA
#include "cuda_manager.hpp"
#endif

namespace hmat {

/** RkApproximationControl */
template<typename T> RkApproximationControl RkMatrix<T>::approx;

/** RkMatrix */
template<typename T> RkMatrix<T>::RkMatrix(ScalarArray<T>* _a, const IndexSet* _rows,
                                           ScalarArray<T>* _b, const IndexSet* _cols)
  : rows(_rows),
    cols(_cols),
    a(_a),
    b(_b)
{

  // We make a special case for empty matrices.
  if ((!a) && (!b)) {
    return;
  }
  assert(a->rows == rows->size());
  assert(b->rows == cols->size());
}

template<typename T> RkMatrix<T>::~RkMatrix() {
  clear();
}


template<typename T> ScalarArray<T>* RkMatrix<T>::evalArray(ScalarArray<T>* result) const {
  if(result==nullptr)
    result = new ScalarArray<T>(rows->size(), cols->size());
  if (rank())
    result->gemm('N', 'T', 1, a, b, 0);
  else
    result->clear();
  return result;
}

template<typename T> FullMatrix<T>* RkMatrix<T>::eval() const {
  FullMatrix<T>* result = new FullMatrix<T>(rows, cols, false);
  evalArray(&result->data);
  return result;
}

// Compute squared Frobenius norm
template<typename T> double RkMatrix<T>::normSqr() const {
  return a->norm_abt_Sqr(*b);
}

template<typename T> void RkMatrix<T>::scale(T alpha) {
  // We need just to scale the first matrix, A.
  if (a) {
    a->scale(alpha);
  }
}

template<typename T> void RkMatrix<T>::transpose() {
  std::swap(a, b);
  std::swap(rows, cols);
}

template<typename T> void RkMatrix<T>::clear() {
  delete a;
  delete b;
  a = nullptr;
  b = nullptr;
}

template<typename T>
void RkMatrix<T>::gemv(char trans, T alpha, const ScalarArray<T>* x, T beta, ScalarArray<T>* y, Side side) const {
  if (rank() == 0) {
    if (beta != T(1)) {
      y->scale(beta);
    }
    return;
  }
  if (side == Side::LEFT) {
    if (trans == 'N') {
      // Compute Y <- Y + alpha * A * B^T * X
      ScalarArray<T> z(b->cols, x->cols);
      z.gemm('T', 'N', 1, b, x, 0);
      y->gemm('N', 'N', alpha, a, &z, beta);
    } else if (trans == 'T') {
      // Compute Y <- Y + alpha * B * A^T * X
      ScalarArray<T> z(a->cols, x->cols);
      z.gemm('T', 'N', 1, a, x, 0);
      y->gemm('N', 'N', alpha, b, &z, beta);
    } else {
      assert(trans == 'C');
      // Compute Y <- Y + alpha * (A*B^T)^H * X = Y + alpha * conj(B) * A^H * X
      ScalarArray<T> z(a->cols, x->cols);
      z.gemm('C', 'N', 1, a, x, 0);
      ScalarArray<T> * newB = b->copy();
      newB->conjugate();
      y->gemm('N', 'N', alpha, newB, &z, beta);
      delete newB;
    }
  } else {
    if (trans == 'N') {
      // Compute Y <- Y + alpha * X * A * B^T
      ScalarArray<T> z(x->rows, a->cols);
      z.gemm('N', 'N', 1, x, a, 0);
      y->gemm('N', 'T', alpha, &z, b, beta);
    } else if (trans == 'T') {
      // Compute Y <- Y + alpha * X * B * A^T
      ScalarArray<T> z(x->rows, b->cols);
      z.gemm('N', 'N', 1, x, b, 0);
      y->gemm('N', 'T', alpha, &z, a, beta);
    } else {
      assert(trans == 'C');
      // Compute Y <- Y + alpha * X * (A*B^T)^H = Y + alpha * X * conj(B) * A^H
      ScalarArray<T> * newB = b->copy();
      newB->conjugate();
      ScalarArray<T> z(x->rows, b->cols);
      z.gemm('N', 'N', 1, x, newB, 0);
      delete newB;
      y->gemm('N', 'C', alpha, &z, a, beta);
    }
  }
}

template<typename T> const RkMatrix<T>* RkMatrix<T>::subset(const IndexSet* subRows,
                                                            const IndexSet* subCols) const {
  assert(subRows->isSubset(*rows));
  assert(subCols->isSubset(*cols));
  ScalarArray<T>* subA = nullptr;
  ScalarArray<T>* subB = nullptr;
  if(rank() > 0) {
    // The offset in the matrix, and not in all the indices
    int rowsOffset = subRows->offset() - rows->offset();
    int colsOffset = subCols->offset() - cols->offset();
    subA = new ScalarArray<T>(*a, rowsOffset, subRows->size(), 0, rank());
    subB = new ScalarArray<T>(*b, colsOffset, subCols->size(), 0, rank());
  }
  return new RkMatrix<T>(subA, subRows, subB, subCols);
}

template<typename T> RkMatrix<T>* RkMatrix<T>::truncatedSubset(const IndexSet* subRows,
                                                               const IndexSet* subCols,
                                                               double epsilon) const {
  assert(subRows->isSubset(*rows));
  assert(subCols->isSubset(*cols));
  RkMatrix<T> * r = new RkMatrix<T>(nullptr, subRows, nullptr, subCols);
  if(rank() > 0) {
    r->a = ScalarArray<T>(*a, subRows->offset() - rows->offset(),
                          subRows->size(), 0, rank()).copy();
    r->b = ScalarArray<T>(*b, subCols->offset() - cols->offset(),
                          subCols->size(), 0, rank()).copy();
    if(epsilon >= 0)
      r->truncate(epsilon);
  }
  return r;
}

template<typename T> size_t RkMatrix<T>::compressedSize() {
    return ((size_t)rows->size()) * rank() + ((size_t)cols->size()) * rank();
}

template<typename T> size_t RkMatrix<T>::uncompressedSize() {
    return ((size_t)rows->size()) * cols->size();
}

template<typename T> void RkMatrix<T>::addRand(double epsilon) {
  DECLARE_CONTEXT;
  a->addRand(epsilon);
  b->addRand(epsilon);
  return;
}

/**
 * @brief Truncate the A or B block of a RkMatrix.
 * This is a utilitary function for RkMatrix<T>::truncated and RkMatrix<T>::truncate
 * @param ab The A or B block to truncate
 * @param indexSet The index set of the A or B block to truncate (rows for A and cols for B)
 * @param newK The rank to truncate to (output of the SVD)
 * @param uv The U or V matrix (output of the SVD)
 * @return the truncated A or B block
 */
template <typename T>
ScalarArray<T> *truncatedAB(ScalarArray<T> *ab, const IndexSet *indexSet,
                            int newK, ScalarArray<T> *uv,
                            bool useInitPivot = false, int initialPivot = 0) {
  // We need to calculate Qa * u
  ScalarArray<T>* newAB = new ScalarArray<T>(indexSet->size(), newK);
  if (useInitPivot && initialPivot) {
    // If there is an initial pivot, we must compute the product by Q in two parts
    // first the column >= initialPivotA, obtained from lapack GETRF, will overwrite newA when calling UNMQR
    // then the first initialPivotA columns, with a classical GEMM, will add the result in newA

    // create subset of a (columns>=initialPivotA) and u (rows>=initialPivotA)
    ScalarArray<T> sub_ab(*ab, 0, ab->rows, initialPivot, ab->cols-initialPivot);
    ScalarArray<T> sub_uv(* uv, initialPivot,  uv->rows-initialPivot, 0,  uv->cols);
    newAB->copyMatrixAtOffset(&sub_uv, 0, 0);
    // newA <- Qa * newA (with newA = u)
    sub_ab.productQ('L', 'N', newAB);

    // then add the regular part of the product by Q
    ScalarArray<T> sub_ab2(*ab, 0, ab->rows, 0, initialPivot);
    ScalarArray<T> sub_uv2(* uv, 0, initialPivot, 0,  uv->cols);
    newAB->gemm('N', 'N', 1, &sub_ab2, &sub_uv2, 1);
  } else {
    // If no initialPivotA, then no gemm, just a productQ()
    newAB->copyMatrixAtOffset( uv, 0, 0);
    // newA <- Qa * newA
    ab->productQ('L', 'N', newAB);
  }

  newAB->setOrtho( uv->getOrtho());
  delete  uv;
  return newAB;
}

template<typename T> void RkMatrix<T>::truncate(double epsilon, int initialPivotA, int initialPivotB) {
  DECLARE_CONTEXT;
  if (rank() == 0) {
    assert(!(a || b));
    return;
  }
  
  assert(rows->size() >= rank());
  // Case: more columns than one dimension of the matrix.
  // In this case, the calculation of the SVD of the matrix "R_a R_b^t" is more
  // expensive than computing the full SVD matrix. We make then a full matrix conversion,
  // and compress it with RkMatrix::fromMatrix().
  if (rank() > std::min(rows->size(), cols->size())) {
    FullMatrix<T>* tmp = eval();
    RkMatrix<T>* rk = truncatedSvd(tmp, epsilon); // TODO compress with something else than SVD (rank() can still be quite large) ?
    delete tmp;
    // "Move" rk into this, and delete the old "this".
    swap(*rk);
    delete rk;
    return;
  }
  
  static bool usedRecomp = getenv("HMAT_RECOMPRESS") && strcmp(getenv("HMAT_RECOMPRESS"), "MGS") == 0 ;
  if (usedRecomp){
    mGSTruncate(epsilon, initialPivotA, initialPivotB);
    return;
  }
  
  /* To recompress an Rk-matrix to Rk-matrix, we need :
      - A = Q_a R_A (QR decomposition)
      - B = Q_b R_b (QR decomposition)
      - Calculate the SVD of R_a R_b^t  = U S V^t
      - Make truncation U, S, and V in the same way as for
      compression of a full rank matrix, ie:
      - Restrict U to its newK first columns U_tilde
      - Restrict S to its newK first values (diagonal matrix) S_tilde
      - Restrict V to its newK first columns V_tilde
      - Put A = Q_a U_tilde SQRT (S_tilde)
      B = Q_b V_tilde SQRT(S_tilde)

     The sizes of the matrices are:
      - Qa : rows x k
      - Ra : k x k
      - Qb : cols x k
      - Rb : k x k
     So:
      - Ra Rb^t: k x k
      - U  : k * k
      - S  : k (diagonal)
      - V^t: k * k
      After truncation:
      - U_tilde  : k * newK
      - S_tilde  : newK (diagonal)
      - V_tilde^t: newK * k
      Hence:
      - newA: rows x newK
      - newB: cols x newK
  */
  
#ifdef HAVE_CUDA
  if (hmat::CudaManager::getInstance().getCudaDeviceCount()) {

    // Set the right GPU as active to allocate and free memory there
    hmat::CudaManager::getInstance().setCudaDevice();
    T *a_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&a_gpu, sizeof(T) * a->rows * a->cols));
    CUDA_CHECK(cudaMemcpy(a_gpu, a->ptr(), sizeof(T) * a->rows * a->cols, cudaMemcpyHostToDevice));
    T *b_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&b_gpu, sizeof(T) * b->rows * b->cols));
    CUDA_CHECK(cudaMemcpy(b_gpu, b->ptr(), sizeof(T) * b->rows * b->cols, cudaMemcpyHostToDevice));
    
    int info = 0;
    
    // Facto QR de a = Qa * Ra
    T *tauA_gpu = nullptr, *Ra_gpu = nullptr;
    info = proxy_cuda::geqrf(a->rows, a->cols, a_gpu, a->rows, &tauA_gpu, &Ra_gpu);
    HMAT_ASSERT(!info);
    
    // Facto QR de b = Qb * Rb
    T *tauB_gpu = nullptr, *Rb_gpu = nullptr;
    info = proxy_cuda::geqrf(b->rows, b->cols, b_gpu, b->rows, &tauB_gpu, &Rb_gpu);
    HMAT_ASSERT(!info);

    // Ra_gpu <- Ra_gpu * T^Rb_gpu        
    proxy_cuda::trmm('R', 'U', 'T', 'N', a->cols, a->cols, (T)1., Rb_gpu, a->cols, Ra_gpu, a->cols, Ra_gpu, a->cols);
    CUDA_CHECK(cudaFree(Rb_gpu));

    // Décomposition SVD de Ra = U.S.VT/VH
    typename Types<T>::real *S_gpu = nullptr;
    T *U_gpu = nullptr, *VT_gpu = nullptr;
    info = proxy_cuda::gesvd('A', 'A', a->cols, a->cols, Ra_gpu, a->cols, &S_gpu, &U_gpu, &VT_gpu);
    HMAT_ASSERT(!info);
    CUDA_CHECK(cudaFree(Ra_gpu));

    // Compute the new truncated rank and the squareroots of all singular values
    int *newK_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&newK_gpu, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(newK_gpu, &a->cols, sizeof(int), cudaMemcpyHostToDevice));  
    launch_FindKAndSqrtAll<typename Types<T>::real>(S_gpu, epsilon, a->cols, newK_gpu);
    int newK = 0;
    CUDA_CHECK(cudaMemcpy(&newK, newK_gpu, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(newK_gpu));

    // Transposition de VT/VH (VT^T = V : k x newK)
    T* V_gpu=nullptr;
    proxy_cuda::geam('T', 'N', a->cols, newK, (T)1., VT_gpu, a->cols, (T)0., (T*)nullptr, a->cols, &V_gpu, a->cols);
    CUDA_CHECK(cudaFree(VT_gpu));

    // Apply the squareroots of singular values to the newK columns of U and V
    // Since S_gpu is a real array, we use S/D dgmm even with U,V complex, with twice the number of rows
    int mult = hmat::Types<T>::IS_REAL::value ? 1 : 2;
    proxy_cuda::dgmm<typename Types<T>::real>('R', mult*a->cols, newK, reinterpret_cast<typename Types<T>::real*>(U_gpu), mult*a->cols, S_gpu, 1, reinterpret_cast<typename Types<T>::real*>(U_gpu), mult*a->cols);
    proxy_cuda::dgmm<typename Types<T>::real>('R', mult*a->cols, newK, reinterpret_cast<typename Types<T>::real*>(V_gpu), mult*a->cols, S_gpu, 1, reinterpret_cast<typename Types<T>::real*>(V_gpu), mult*a->cols);
    CUDA_CHECK(cudaFree(S_gpu));

    // Construction du panneau QaU_gpu (a->rows x newK) <- Qa (a->rows x a->rows) * U (a->rows x newK)
    T *QaU_gpu = nullptr; 
    info = proxy_cuda::or_un_mqr('L', 'N', a->rows, newK, a->cols, a_gpu, a->rows, tauA_gpu, U_gpu, a->cols, &QaU_gpu);  
    HMAT_ASSERT(!info);
    CUDA_CHECK(cudaFree(U_gpu));
    CUDA_CHECK(cudaFree(tauA_gpu));
    CUDA_CHECK(cudaFree(a_gpu));

    // Construction du panneau QbV_gpu (b->rows x newK) <- Qb (b->rows x b->rows) * U (b->rows x newK)
    T *QbV_gpu = nullptr; 
    info = proxy_cuda::or_un_mqr('L', 'N', b->rows, newK, b->cols, b_gpu, b->rows, tauB_gpu, V_gpu, b->cols, &QbV_gpu);  
    HMAT_ASSERT(!info);
    CUDA_CHECK(cudaFree(V_gpu));
    CUDA_CHECK(cudaFree(tauB_gpu));
    CUDA_CHECK(cudaFree(b_gpu));

    T* a_data_copy = (T*)malloc(sizeof(T)* a->rows * newK);
    CUDA_CHECK(cudaMemcpy(a_data_copy, QaU_gpu, sizeof(T)* a->rows * newK, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(QaU_gpu));

    T* b_data_copy = (T*)malloc(sizeof(T)* b->rows * newK);
    CUDA_CHECK(cudaMemcpy(b_data_copy, QbV_gpu, sizeof(T)* b->rows * newK, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(QbV_gpu));

    ScalarArray<T> *newA_CUDA = new ScalarArray<T>(a_data_copy, a->rows, newK, a->rows);
    ScalarArray<T> *newB_CUDA = new ScalarArray<T>(b_data_copy, b->rows, newK, b->rows);
    delete a;
    a = newA_CUDA;
    delete b;
    b = newB_CUDA;
    // NOTE: The panels a and b can usually not be directly compared between CPU and GPU. If the singular values are all distincts,
    // the columns of a resp. b on CPU and GPU are equal up to a norm 1 constant (+-1 in real, exp(i.theta) in complex): if I
    // multiply a column of a by exp(i.theta) and the same column of b by exp(-i.theta), my Rk matrix is unchanged.

  } else 
#endif // HAVE_CUDA
  
  {
    /* --------------------------------- CPU CODE --------------------------------- */  
    
    // QR decomposition of A and B
    ScalarArray<T> ra(rank(), rank());
    a->qrDecomposition(&ra, initialPivotA); // A contains Qa and tau_a
    ScalarArray<T> rb(rank(), rank());
    b->qrDecomposition(&rb, initialPivotB); // B contains Qb and tau_b
    
    // R <- Ra Rb^t
    ScalarArray<T> r(rank(), rank());
    r.gemm('N','T', 1, &ra, &rb , 0);
    
    // truncated SVD of Ra Rb^t (allows failure)
    ScalarArray<T> *u = nullptr, *v = nullptr;
    int newK_cpu = r.truncatedSvdDecomposition(&u, &v, epsilon, true); // TODO use something else than SVD ?
    
    if (newK_cpu == 0) {
      clear();
      return;
    }
    // We need to know if qrDecomposition has used initPivot...
    // (Not so great, because HMAT_TRUNC_INITPIV is checked at 2 different locations)
    static char *useInitPivot = getenv("HMAT_TRUNC_INITPIV");
    ScalarArray<T>* newA = truncatedAB(a, rows, newK_cpu, u, useInitPivot, initialPivotA);
    ScalarArray<T>* newB = truncatedAB(b, cols, newK_cpu, v, useInitPivot, initialPivotB);
    delete a;
    a = newA;
    delete b;
    b = newB;
  }

#ifdef HAVE_CUDA
  /*if(newK_cpu != newK)
    std::cout << "New rank sur CPU : " << newK_cpu << " sur GPU " << newK << "\n";
  double frob_a = newA->norm();
  double frob_b = newB->norm();
  double frob_a_gpu = newA_CUDA->norm();
  double frob_b_gpu = newB_CUDA->norm();
  newA_CUDA->axpy(-1, a);
  newB_CUDA->axpy(-1, b);
  double frob_a_diff = newA_CUDA->norm();
  double frob_b_diff = newB_CUDA->norm();
  newA_CUDA->axpy(2, a);
  newB_CUDA->axpy(2, b);
  double frob_a_diff2 = newA_CUDA->norm();
  double frob_b_diff2 = newB_CUDA->norm();
  if(frob_a_diff / frob_a > 1e-4 && frob_a_diff2 / frob_a > 1e-4)
    std::cout << "Norme de Froebenius de l'erreur relative entre CPU et GPU sur a : " << frob_a_diff / frob_a << " ou "  << frob_a_diff2 / frob_a << " frob_a " << frob_a << "frob_a_gpu " << frob_a_gpu << "\n";
  if(frob_b_diff / frob_b > 1e-4 && frob_b_diff2 / frob_b > 1e-4)
    std::cout << "Norme de Froebenius de l'erreur relative entre CPU et GPU sur a : " << frob_b_diff / frob_b << " ou " << frob_b_diff2 / frob_b << " frob_b " << frob_b << "frob_b_gpu " << frob_b_gpu << "\n";
  delete newA_CUDA;
  delete newB_CUDA;*/
#endif // HAVE_CUDA

}

template<typename T> 
void RkMatrix<T>::truncateAlter(double epsilon)
{
  int *sigma_a=nullptr;
  int *sigma_b=nullptr;
  double *tau_a=nullptr;
  double *tau_b=nullptr;
  int rank_a;
  int rank_b;
  char transA='T';
  if (std::is_same<T,C_t>::value || std::is_same<T,Z_t>::value) transA='C';
  a->cpqrDecomposition(sigma_a, tau_a, &rank_a, epsilon);
  b->cpqrDecomposition(sigma_b, tau_b, &rank_b, epsilon);
  IndexSet row_(0,rank_a);
  IndexSet col_(0,rank_b);
  ScalarArray<T> coef(rank_a,rank_b,true);
  ScalarArray<T> R_a(rank_a, rank(), true);
  ScalarArray<T> R_b(rank_b, rank(), true);

  //construction of R_a and R_b according to sigma_a and sigma_b

  for (int j = 0 ; j < rank() ; j++)
  {
    memcpy(&R_a.get(0,sigma_a[j]), &a->get(0,j), sizeof(T)*std::min(j+1,rank_a));
    memcpy(&R_b.get(0,sigma_b[j]), &b->get(0,j), sizeof(T)*std::min(j+1,rank_b));
  }
  delete sigma_a;
  delete sigma_b;
  coef.gemm('N', 'T', 1 , &R_a , &R_b , 0);
  FullMatrix<T> midMat(&coef ,&row_ ,&col_);
  RkMatrix<T> *RkMid=rankRevealingQR(&midMat , epsilon);
  ScalarArray<T> *newA=new ScalarArray<T>(a->rows, RkMid->rank(), true);
  ScalarArray<T> *newB=new ScalarArray<T>(b->rows, RkMid->rank(), true);
  newA->copyMatrixAtOffset(RkMid->a , 0 , 0);
  newB->copyMatrixAtOffset(RkMid->b , 0 , 0);

  //product by Q_A and Q_B (can't we use LAPACK calls ?)

  for (int k = rank_a-1 ; k>=0 ; k--)
  {
    Vector<T> v_k(a->rows , true);
    v_k[k]=1;
    memcpy(&(v_k[k+1]), &(a->get(k+1,k)), (a->rows-k-1)*sizeof(T));
    newA->reflect(v_k, tau_a[k], transA);
  }
  for (int k = rank_b-1 ; k>=0 ; k--)
  {
    Vector<T> v_k(b->rows , true);
    v_k[k]=1;
    memcpy(&(v_k[k+1]), &(b->get(k+1,k)), (b->rows-k-1)*sizeof(T));
    newB->reflect(v_k, tau_b[k], transA);
  }
  delete tau_a;
  delete tau_b;
  delete a;
  a=newA;
  delete b;
  b=newB;
}

template <typename T>
void RkMatrix<T>::validateRecompression(double epsilon , int initialPivotA , int initialPivotB)
{
        RkMatrix<T> *copy=RkMatrix<T>::copy();
        auto start1 = std::chrono::high_resolution_clock::now();
        truncate(epsilon , initialPivotA , initialPivotB);
        auto end1 = std::chrono::high_resolution_clock::now();
        auto start2 = std::chrono::high_resolution_clock::now();
        copy->truncateAlter(epsilon);
        auto end2 = std::chrono::high_resolution_clock::now();
        double exec_time_truncate=std::chrono::duration_cast<std::chrono::nanoseconds>(end1-start1).count();
        double exec_time_truncateAlter=std::chrono::duration_cast<std::chrono::nanoseconds>(end2-start2).count();
        exec_time_truncate*=1e-9;
        exec_time_truncateAlter*=1e-9;
        ScalarArray<T> mat1(rows->size(), cols->size());
        ScalarArray<T> mat2(rows->size(), cols->size());
        mat1.gemm('N', 'T', 1, copy->a , copy->b , 0);
        mat2.gemm('N', 'T', 1, a ,b , 0);
        double norm_classic=mat2.norm();
        for (int i = 0 ; i < rows->size() ; i++)
        {
          for (int j =0 ; j<cols->size() ; j++)
          {
            mat1.get(i,j)-=mat2.get(i,j);
          }
        }
        std::cout<<std::scientific<<"recompression test :\n"
                  <<"||addClassic(R1,R2)-addToTest(R1,R2)||/||addClassic(R1,R2)|| ="<<mat1.norm()/norm_classic<<std::endl
                  <<" rank with classical method = "<<rank()<<std::endl
                  <<" rank with tested method = "<<copy->rank()<<std::endl
                  <<" recompression time with classical method = "<<exec_time_truncate<<std::setprecision(9)<<" s"<<std::endl
                  <<" recompression time with tested method = "<<exec_time_truncateAlter<<" s"<<std::endl;
        delete copy;
}
template<typename T> void RkMatrix<T>::mGSTruncate(double epsilon, int initialPivotA, int initialPivotB) {
  DECLARE_CONTEXT;
  if (rank() == 0) {
    assert(!(a || b));
    return;
  }

  int kA, kB, newK;

  int krank = rank();

  // Gram-Schmidt on a
  // On input, a0(m,k)
  // On output, a(m,kA), ra(kA,k) such that a0 = a * ra
  ScalarArray<T> ra(krank, krank);
  kA = a->modifiedGramSchmidt( &ra, epsilon, initialPivotA);
  if (kA==0) {
    clear();
    return;
  }

  // Gram-Schmidt on b
  // On input, b0(p,k)
  // On output, b(p,kB), rb(kB,k) such that b0 = b * rb
  ScalarArray<T> rb(krank, krank);
  kB = b->modifiedGramSchmidt( &rb, epsilon, initialPivotB);
  if (kB==0) {
    clear();
    return;
  }

  // M = a0*b0^T = a*(ra*rb^T)*b^T
  // We perform an SVD on ra*rb^T:
  //  (ra*rb^T) = U*S*S*Vt
  // and M = (a*U*S)*(S*Vt*b^T) = (a*U*S)*(b*(S*Vt)^T)^T
  ScalarArray<T> matR(kA, kB);
  matR.gemm('N','T', 1, &ra, &rb , 0);

  // truncatedSVD (allows failure)
  ScalarArray<T>* ur = nullptr;
  ScalarArray<T>* vr = nullptr;
  newK = matR.truncatedSvdDecomposition(&ur, &vr, epsilon, true);
  // On output, ur->rows = kA, vr->rows = kB

  if (newK == 0) {
    clear();
    return;
  }

  /* Multiplication by orthogonal matrix Q: no or/un-mqr as
    this comes from Gram-Schmidt procedure not Householder
  */
  ScalarArray<T> *newA = new ScalarArray<T>(a->rows, newK);
  newA->gemm('N', 'N', 1, a, ur, 0);

  ScalarArray<T> *newB = new ScalarArray<T>(b->rows, newK);
  newB->gemm('N', 'N', 1, b, vr, 0);

  newA->setOrtho(ur->getOrtho());
  newB->setOrtho(vr->getOrtho());
  delete ur;
  delete vr;

  delete a;
  a = newA;
  delete b;
  b = newB;
}

// Swap members with members from another instance.
template<typename T> void RkMatrix<T>::swap(RkMatrix<T>& other)
{
  assert(*rows == *other.rows);
  assert(*cols == *other.cols);
  std::swap(a, other.a);
  std::swap(b, other.b);
}

template<typename T> void RkMatrix<T>::axpy(double epsilon, T alpha, const FullMatrix<T>* mat) {
  formattedAddParts(epsilon, &alpha, &mat, 1);
}

template<typename T> void RkMatrix<T>::axpy(double epsilon, T alpha, const RkMatrix<T>* mat) {
  formattedAddParts(epsilon, &alpha, &mat, 1);
}

/*! \brief Try to optimize the order of the Rk matrix to maximize initialPivot

  We re-order the Rk matrices in usedParts[] and the associated constants in usedAlpha[]
  in order to maximize the number of orthogonal columns starting at column 0 in the A and B
  panels.
  \param[in] notNullParts number of elements in usedParts[] and usedAlpha[]
  \param[in,out] usedParts Array of Rk matrices
  \param[in,out] usedAlpha Array of constants
  \param[out] initialPivotA Number of orthogonal columns starting at column 0 in panel A
  \param[out] initialPivotB Number of orthogonal columns starting at column 0 in panel A
*/
template<typename T>
static void optimizeRkArray(int notNullParts, const RkMatrix<T>** usedParts, T *usedAlpha, int &initialPivotA, int &initialPivotB){
  // 1st optim: Put in first position the Rk matrix with orthogonal panels AND maximum rank
  int bestRk=-1, bestGain=-1;
  for (int i=0 ; i<notNullParts ; i++) {
    // Roughly, the gain from an initial pivot 'p' in a QR factorisation 'm x n' is to reduce the flops
    // from 2mn^2 to 2m(n^2-p^2), so the gain grows like p^2 for each panel
    // hence the gain formula : number of orthogonal panels x rank^2
    int gain = (usedParts[i]->a->getOrtho() + usedParts[i]->b->getOrtho())*usedParts[i]->rank()*usedParts[i]->rank();
    if (gain > bestGain) {
      bestGain = gain;
      bestRk = i;
    }
  }
  if (bestRk > 0) {
    std::swap(usedParts[0], usedParts[bestRk]) ;
    std::swap(usedAlpha[0], usedAlpha[bestRk]) ;
  }
  initialPivotA = usedParts[0]->a->getOrtho() ? usedParts[0]->rank() : 0;
  initialPivotB = usedParts[0]->b->getOrtho() ? usedParts[0]->rank() : 0;

  // 2nd optim:
  // When coallescing Rk from childs toward parent, it is possible to "merge" Rk from (extra-)diagonal
  // childs because with non-intersecting rows and cols we will extend orthogonality between separate Rk.
  int best_i1=-1, best_i2=-1, best_rkA=-1, best_rkB=-1;
  for (int i1=0 ; i1<notNullParts ; i1++)
    for (int i2=0 ; i2<notNullParts ; i2++)
      if (i1 != i2) {
        const RkMatrix<T>* Rk1 = usedParts[i1];
        const RkMatrix<T>* Rk2 = usedParts[i2];
        // compute the gain expected from puting Rk1-Rk2 in first position
        // Orthogonality of Rk2->a is useful only if Rk1->a is ortho AND rows dont intersect (cols for panel b)
        int rkA = Rk1->a->getOrtho() ? Rk1->rank() + (Rk2->a->getOrtho() && !Rk1->rows->intersects(*Rk2->rows) ? Rk2->rank() : 0) : 0;
        int rkB = Rk1->b->getOrtho() ? Rk1->rank() + (Rk2->b->getOrtho() && !Rk1->cols->intersects(*Rk2->cols) ? Rk2->rank() : 0) : 0;
        int gain = rkA*rkA + rkB*rkB ;
        if (gain > bestGain) {
          bestGain = gain;
          best_i1 = i1;
          best_i2 = i2;
          best_rkA = rkA;
          best_rkB = rkB;
        }
      }

  if (best_i1 >= 0) {
    // put i1 in first position, i2 in second
    std::swap(usedParts[0], usedParts[best_i1]) ;
    std::swap(usedAlpha[0], usedAlpha[best_i1]) ;
    if (best_i2==0) best_i2 = best_i1; // handles the case where best_i2 was usedParts[0] which has just been moved
    std::swap(usedParts[1], usedParts[best_i2]) ;
    std::swap(usedAlpha[1], usedAlpha[best_i2]) ;
    initialPivotA = best_rkA;
    initialPivotB = best_rkB;
  }

}

template<typename T> bool allSame(const RkMatrix<T>** rks, int n) {
  for(int i = 1; i < n; i++) {
    if(!(*rks[0]->rows == *rks[i]->rows) || !(*rks[0]->cols == *rks[i]->cols))
      return false;
  }
  return true;
}

template<typename T>
void RkMatrix<T>::formattedAddParts(double epsilon, const T* alpha, const RkMatrix<T>* const * parts,
                                    const int n, bool hook) {
  if(hook && formatedAddPartsHook && formatedAddPartsHook(this, epsilon, alpha, parts, n))
    return;
  // TODO check if formattedAddParts() actually uses sometimes this 'alpha' parameter (or is it always 1 ?)
  DECLARE_CONTEXT;

  /* List of non-null and non-empty Rk matrices to coalesce, and the corresponding scaling coefficients */
  std::vector< RkMatrix<T> const * > usedParts(n + 1);
  std::vector<T> usedAlpha(n+1);
  /* Number of elements in usedParts[] */
  int notNullParts = 0;
  /* Sum of the ranks */
  int rankTotal = 0;

  // If needed, put 'this' in first position in usedParts[]
  if (rank()) {
    usedAlpha[0] = 1 ;
    usedParts[notNullParts++] = this ;
    rankTotal += rank();
  }

  for (int i = 0; i < n; i++) {
    // exclude the nullptr and 0-rank matrices
    if (!parts[i] || parts[i]->rank() == 0 || parts[i]->rows->size() == 0 || parts[i]->cols->size() == 0 || alpha[i] == T(0))
      continue;
    // Check that partial RkMatrix indices are subsets of their global indices set.
    assert(parts[i]->rows->isSubset(*rows));
    assert(parts[i]->cols->isSubset(*cols));
    // Add this Rk to the list
    rankTotal += parts[i]->rank();
    usedAlpha[notNullParts] = alpha[i] ;
    usedParts[notNullParts] = parts[i] ;
    notNullParts++;
  }

  if(notNullParts == 0)
    return;

  // In case the sum of the ranks of the sub-matrices is greater than
  // the matrix size, it is more efficient to put everything in a
  // full matrix.
  if (rankTotal >= std::min(rows->size(), cols->size())) {
    const FullMatrix<T>** fullParts = new const FullMatrix<T>*[notNullParts];
    fullParts[0] = nullptr ;
    for (int i = rank() ? 1 : 0 ; i < notNullParts; i++) // exclude usedParts[0] if it is 'this'
      fullParts[i] = usedParts[i]->eval();
    formattedAddParts(std::abs(epsilon), usedAlpha.data(), fullParts, notNullParts);
    for (int i = 0; i < notNullParts; i++)
      delete fullParts[i];
    delete[] fullParts;
    return;
  }

  // Find if the QR factorisation can be accelerated using orthogonality information
  int initialPivotA = usedParts[0]->a->getOrtho() ? usedParts[0]->rank() : 0;
  int initialPivotB = usedParts[0]->b->getOrtho() ? usedParts[0]->rank() : 0;

  // Try to optimize the order of the Rk matrix to maximize initialPivot
  static char *useBestRk = getenv("HMAT_MGS_BESTRK");
  if (useBestRk)
    optimizeRkArray(notNullParts, usedParts.data(), usedAlpha.data(), initialPivotA, initialPivotB);

  // According to the indices organization, the sub-matrices are
  // contiguous blocks in the "big" matrix whose columns offset is
  //      kOffset = usedParts[0]->k + ... + usedParts[i-1]->k
  // rows offset is
  //      usedParts[i]->rows->offset - rows->offset
  // rows size
  //      usedParts[i]->rows->size x usedParts[i]->k (rows x columns)
  // Same for columns.

  // when possible realloc this a & b arrays to limit memory usage and avoid a copy
  bool useRealloc = usedParts[0] == this && allSame(usedParts.data(), notNullParts);
  // concatenate a(i) then b(i) to limite memory usage
  ScalarArray<T>* resultA, *resultB;
  int rankOffset;
  if(useRealloc) {
    resultA = a;
    rankOffset = a->cols;
    a->resize(rankTotal);
  }
  else {
    rankOffset = 0;
    resultA = new ScalarArray<T>(rows->size(), rankTotal);
  }

  for (int i = useRealloc ? 1 : 0; i < notNullParts; i++) {
    // Copy 'a' at position rowOffset, kOffset
    int rowOffset = usedParts[i]->rows->offset() - rows->offset();
    resultA->copyMatrixAtOffset(usedParts[i]->a, rowOffset, rankOffset);
    // Scaling the matrix already in place inside resultA
    if (usedAlpha[i] != T(1)) {
      ScalarArray<T> tmp(*resultA, rowOffset, usedParts[i]->a->rows, rankOffset, usedParts[i]->a->cols);
      tmp.scale(usedAlpha[i]);
    }
    // Update the rank offset
    rankOffset += usedParts[i]->rank();
  }
  assert(rankOffset==rankTotal);

  if(!useRealloc && a != nullptr)
    delete a;
  a = resultA;

  if(useRealloc) {
    resultB = b;
    rankOffset = b->cols;
    b->resize(rankTotal);
  }
  else {
    rankOffset = 0;
    resultB = new ScalarArray<T>(cols->size(), rankTotal);
  }

  for (int i = useRealloc ? 1 : 0; i < notNullParts; i++) {
    // Copy 'b' at position colOffset, kOffset
    int colOffset = usedParts[i]->cols->offset() - cols->offset();
    resultB->copyMatrixAtOffset(usedParts[i]->b, colOffset, rankOffset);
    // Update the rank offset
    rankOffset += usedParts[i]->b->cols;
  }

  if(!useRealloc && b != nullptr)
    delete b;
  b = resultB;

  assert(rankOffset==rankTotal);
  // If only one of the parts is non-zero, then the recompression is not necessary
  if (notNullParts > 1 && epsilon >= 0){
    if(HMatrix<T>::validateRecompression)
      validateRecompression(epsilon , initialPivotA , initialPivotB);
    else
      truncate(epsilon , initialPivotA , initialPivotB);    
  }
}

template<typename T>
void RkMatrix<T>::formattedAddParts(double epsilon, const T* alpha, const FullMatrix<T>* const * parts, int n) {
  DECLARE_CONTEXT;
  FullMatrix<T>* me = eval();
  HMAT_ASSERT(me);

  // TODO: here, we convert Rk->Full, Update the Full with parts[], and Full->Rk. We could also
  // create a new empty Full, update, convert to Rk and add it to 'this'.
  // If the parts[] are smaller than 'this', convert them to Rk and add them could be less expensive
  for (int i = 0; i < n; i++) {
    if (!parts[i])
      continue;
    const IndexSet *rows_full = parts[i]->rows_;
    const IndexSet *cols_full = parts[i]->cols_;
    assert(rows_full->isSubset(*rows));
    assert(cols_full->isSubset(*cols));
    int rowOffset = rows_full->offset() - rows->offset();
    int colOffset = cols_full->offset() - cols->offset();
    int maxCol = cols_full->size();
    int maxRow = rows_full->size();
    ScalarArray<T> sub(me->data, rowOffset, maxRow, colOffset, maxCol);
    sub.axpy(alpha[i], &parts[i]->data);
  }
  RkMatrix<T>* result = truncatedSvd(me, epsilon); // TODO compress with something else than SVD
  delete me;
  swap(*result);
  delete result;
}


template<typename T> RkMatrix<T>* RkMatrix<T>::multiplyRkFull(char transR, char transM,
                                                              const RkMatrix<T>* rk,
                                                              const FullMatrix<T>* m) {
  DECLARE_CONTEXT;

  assert(((transR == 'N') ? rk->cols->size() : rk->rows->size()) == ((transM == 'N') ? m->rows() : m->cols()));
  const IndexSet *rkRows = ((transR == 'N')? rk->rows : rk->cols);
  const IndexSet *mCols = ((transM == 'N')? m->cols_ : m->rows_);

  if(rk->rank() == 0) {
      return new RkMatrix<T>(nullptr, rkRows, nullptr, mCols);
  }
  // If transM is 'N' and transR is 'N', we compute
  //  A * B^T * M ==> newA = A, newB = M^T * B
  // We can deduce all other cases from this one:
  //   * if transR is 'T', all we have to do is to swap A and B
  //   * if transR is 'C', we swap A and B, but they must also
  //     be conjugate; let us look at the different cases:
  //     + if transM is 'N', newB = M^T * conj(B) = conj(M^H * B)
  //     + if transM is 'T', newB = M * conj(B)
  //     + if transM is 'C', newB = conj(M) * conj(B) = conj(M * B)

  ScalarArray<T> *newA, *newB;
  ScalarArray<T>* ra = rk->a;
  ScalarArray<T>* rb = rk->b;
  if (transR != 'N') {
    // if transR == 'T', we permute ra and rb; if transR == 'C', they will
    // also have to be conjugated, but this cannot be done here because rk
    // is const, this will be performed below.
    std::swap(ra, rb);
  }
  newA = ra->copy();
  newB = new ScalarArray<T>(transM == 'N' ? m->cols() : m->rows(), rb->cols);
  if (transR == 'C') {
    newA->conjugate();
    if (transM == 'N') {
      newB->gemm('C', 'N', 1, &m->data, rb, 0);
      newB->conjugate();
    } else if (transM == 'T') {
      ScalarArray<T> *conjB = rb->copy();
      conjB->conjugate();
      newB->gemm('N', 'N', 1, &m->data, conjB, 0);
      delete conjB;
    } else {
      assert(transM == 'C');
      newB->gemm('N', 'N', 1, &m->data, rb, 0);
      newB->conjugate();
    }
  } else {
    if (transM == 'N') {
      newB->gemm('T', 'N', 1, &m->data, rb, 0);
    } else if (transM == 'T') {
      newB->gemm('N', 'N', 1, &m->data, rb, 0);
    } else {
      assert(transM == 'C');
      ScalarArray<T> *conjB = rb->copy();
      conjB->conjugate();
      newB->gemm('N', 'N', 1, &m->data, conjB, 0);
      newB->conjugate();
      delete conjB;
    }
  }
  RkMatrix<T>* result = new RkMatrix<T>(newA, rkRows, newB, mCols);
  return result;
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::multiplyFullRk(char transM, char transR,
                                         const FullMatrix<T>* m,
                                         const RkMatrix<T>* rk) {
  DECLARE_CONTEXT;
  // If transM is 'N' and transR is 'N', we compute
  //  M * A * B^T  ==> newA = M * A, newB = B
  // We can deduce all other cases from this one:
  //   * if transR is 'T', all we have to do is to swap A and B
  //   * if transR is 'C', we swap A and B, but they must also
  //     be conjugate; let us look at the different cases:
  //     + if transM is 'N', newA = M * conj(A)
  //     + if transM is 'T', newA = M^T * conj(A) = conj(M^H * A)
  //     + if transM is 'C', newA = M^H * conj(A) = conj(M^T * A)
  ScalarArray<T> *newA, *newB;
  ScalarArray<T>* ra = rk->a;
  ScalarArray<T>* rb = rk->b;
  if (transR != 'N') { // permutation to transpose the matrix Rk
    std::swap(ra, rb);
  }
  const IndexSet *rkCols = ((transR == 'N')? rk->cols : rk->rows);
  const IndexSet *mRows = ((transM == 'N')? m->rows_ : m->cols_);

  newA = new ScalarArray<T>(mRows->size(), rb->cols);
  newB = rb->copy();
  if (transR == 'C') {
    newB->conjugate();
    if (transM == 'N') {
      ScalarArray<T> *conjA = ra->copy();
      conjA->conjugate();
      newA->gemm('N', 'N', 1, &m->data, conjA, 0);
      delete conjA;
    } else if (transM == 'T') {
      newA->gemm('C', 'N', 1, &m->data, ra, 0);
      newA->conjugate();
    } else {
      assert(transM == 'C');
      newA->gemm('T', 'N', 1, &m->data, ra, 0);
      newA->conjugate();
    }
  } else {
    newA->gemm(transM, 'N', 1, &m->data, ra, 0);
  }
  RkMatrix<T>* result = new RkMatrix<T>(newA, mRows, newB, rkCols);
  return result;
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::multiplyRkH(char transR, char transH,
                                      const RkMatrix<T>* rk, const HMatrix<T>* h) {
  DECLARE_CONTEXT;
  assert(((transR == 'N') ? *rk->cols : *rk->rows) == ((transH == 'N')? *h->rows() : *h->cols()));

  const IndexSet* rkRows = ((transR == 'N')? rk->rows : rk->cols);

  // If transR == 'N'
  //    transM == 'N': (A*B^T)*M = A*(M^T*B)^T
  //    transM == 'T': (A*B^T)*M^T = A*(M*B)^T
  //    transM == 'C': (A*B^T)*M^H = A*(conj(M)*B)^T = A*conj(M*conj(B))^T
  // If transR == 'T', we only have to swap A and B
  // If transR == 'C', we swap A and B, then
  //    transM == 'N': R^H*M = conj(A)*(M^T*conj(B))^T = conj(A)*conj(M^H*B)^T
  //    transM == 'T': R^H*M^T = conj(A)*(M*conj(B))^T
  //    transM == 'C': R^H*M^H = conj(A)*conj(M*B)^T
  //
  // Size of the HMatrix is n x m,
  // So H^t size is m x n and the product is m x cols(B)
  // and the number of columns of B is k.
  ScalarArray<T> *newA, *newB;
  ScalarArray<T>* ra = rk->a;
  ScalarArray<T>* rb = rk->b;
  if (transR != 'N') { // permutation to transpose the matrix Rk
    std::swap(ra, rb);
  }

  const IndexSet *newCols = ((transH == 'N' )? h->cols() : h->rows());

  newA = ra->copy();
  newB = new ScalarArray<T>(transH == 'N' ? h->cols()->size() : h->rows()->size(), rb->cols);
  if (transR == 'C') {
    newA->conjugate();
    if (transH == 'N') {
      h->gemv('C', 1, rb, 0, newB);
      newB->conjugate();
    } else if (transH == 'T') {
      ScalarArray<T> *conjB = rb->copy();
      conjB->conjugate();
      h->gemv('N', 1, conjB, 0, newB);
      delete conjB;
    } else {
      assert(transH == 'C');
      h->gemv('N', 1, rb, 0, newB);
      newB->conjugate();
    }
  } else {
    if (transH == 'N') {
      h->gemv('T', 1, rb, 0, newB);
    } else if (transH == 'T') {
      h->gemv('N', 1, rb, 0, newB);
    } else {
      assert(transH == 'C');
      ScalarArray<T> *conjB = rb->copy();
      conjB->conjugate();
      h->gemv('N', 1, conjB, 0, newB);
      delete conjB;
      newB->conjugate();
    }
  }
  RkMatrix<T>* result = new RkMatrix<T>(newA, rkRows, newB, newCols);
  return result;
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::multiplyHRk(char transH, char transR,
                                      const HMatrix<T>* h, const RkMatrix<T>* rk) {

  DECLARE_CONTEXT;
  if (rk->rank() == 0) {
    const IndexSet* newRows = ((transH == 'N') ? h-> rows() : h->cols());
    const IndexSet* newCols = ((transR == 'N') ? rk->cols : rk->rows);
    return new RkMatrix<T>(nullptr, newRows, nullptr, newCols);
  }

  // If transH is 'N' and transR is 'N', we compute
  //  M * A * B^T  ==> newA = M * A, newB = B
  // We can deduce all other cases from this one:
  //   * if transR is 'T', all we have to do is to swap A and B
  //   * if transR is 'C', we swap A and B, but they must also
  //     be conjugate; let us look at the different cases:
  //     + if transH is 'N', newA = M * conj(A)
  //     + if transH is 'T', newA = M^T * conj(A) = conj(M^H * A)
  //     + if transH is 'C', newA = M^H * conj(A) = conj(M^T * A)
  ScalarArray<T> *newA, *newB;
  ScalarArray<T>* ra = rk->a;
  ScalarArray<T>* rb = rk->b;
  if (transR != 'N') { // permutation to transpose the matrix Rk
    std::swap(ra, rb);
  }
  const IndexSet *rkCols = ((transR == 'N')? rk->cols : rk->rows);
  const IndexSet* newRows = ((transH == 'N')? h-> rows() : h->cols());

  newA = new ScalarArray<T>(transH == 'N' ? h->rows()->size() : h->cols()->size(), rb->cols);
  newB = rb->copy();
  if (transR == 'C') {
    newB->conjugate();
    if (transH == 'N') {
      ScalarArray<T> *conjA = ra->copy();
      conjA->conjugate();
      h->gemv('N', 1, conjA, 0, newA);
      delete conjA;
    } else if (transH == 'T') {
      h->gemv('C', 1, ra, 0, newA);
      newA->conjugate();
    } else {
      assert(transH == 'C');
      h->gemv('T', 1, ra, 0, newA);
      newA->conjugate();
    }
  } else {
    h->gemv(transH, 1, ra, 0, newA);
  }
  RkMatrix<T>* result = new RkMatrix<T>(newA, newRows, newB, rkCols);
  return result;
}

template<typename T>
RkMatrix<T>* RkMatrix<T>::multiplyRkRk(char trans1, char trans2,
                                       const RkMatrix<T>* r1, const RkMatrix<T>* r2, double epsilon) {
  DECLARE_CONTEXT;
  assert(((trans1 == 'N') ? *r1->cols : *r1->rows) == ((trans2 == 'N') ? *r2->rows : *r2->cols));
  // It is possible to do the computation differently, yielding a
  // different rank and a different amount of computation.
  // TODO: choose the best order.
  ScalarArray<T>* a1 = (trans1 == 'N' ? r1->a : r1->b);
  ScalarArray<T>* b1 = (trans1 == 'N' ? r1->b : r1->a);
  ScalarArray<T>* a2 = (trans2 == 'N' ? r2->a : r2->b);
  ScalarArray<T>* b2 = (trans2 == 'N' ? r2->b : r2->a);

  assert(b1->rows == a2->rows); // compatibility of the multiplication

  // We want to compute the matrix a1.t^b1.a2.t^b2 and return an Rk matrix
  // Usually, the best way is to start with tmp=t^b1.a2 which produces a 'small' matrix rank1 x rank2
  //
  // OLD version (default):
  // Then we can either :
  // - compute a1.tmp : the cost is rank1.rank2.row_a, the resulting Rk has rank rank2
  // - compute tmp.t^b2 : the cost is rank1.rank2.col_b, the resulting Rk has rank rank1
  // We use the solution which gives the lowest resulting rank.
  // With this version, orthogonality is lost on one panel, it is preserved on the other.
  //
  // NEW version :
  // Other solution: once we have the small matrix tmp=t^b1.a2, we can do a recompression on it for low cost
  // using SVD + truncation. This also removes the choice above, since tmp=U.S.V is then applied on both sides
  // This version is default, it can be deactivated by setting env. var. HMAT_OLD_RKRK
  // With this version, orthogonality is lost on both panel.

  ScalarArray<T> tmp(r1->rank(), r2->rank(), false);
  if (trans1 == 'C' && trans2 == 'C') {
    tmp.gemm('T', 'N', 1, b1, a2, 0);
    tmp.conjugate();
  } else if (trans1 == 'C') {
    tmp.gemm('C', 'N', 1, b1, a2, 0);
  } else if (trans2 == 'C') {
    tmp.gemm('C', 'N', 1, b1, a2, 0);
    tmp.conjugate();
  } else {
    tmp.gemm('T', 'N', 1, b1, a2, 0);
  }

  ScalarArray<T> *newA=nullptr, *newB=nullptr;
  static char *oldRKRK = getenv("HMAT_OLD_RKRK"); // Option to use the OLD version, without SVD-in-the-middle
  if (!oldRKRK) {
    // NEW version
    ScalarArray<T>* ur = nullptr;
    ScalarArray<T>* vr = nullptr;
    // truncated SVD tmp = ur.t^vr
    int newK = tmp.truncatedSvdDecomposition(&ur, &vr, epsilon, true);
    //printf("%d %d\n", newK, std::min(tmp.rows, tmp.cols));
    if (newK > 0) {
      /* Now compute newA = a1.ur and newB = b2.vr */
      newA = new ScalarArray<T>(a1->rows, newK, false);
      if (trans1 == 'C') ur->conjugate();
      newA->gemm('N', 'N', 1, a1, ur, 0);
      if (trans1 == 'C') newA->conjugate();
      newB = new ScalarArray<T>(b2->rows, newK, false);
      if (trans2 == 'C') vr->conjugate();
      newB->gemm('N', 'N', 1, b2, vr, 0);
      if (trans2 == 'C') newB->conjugate();
      delete ur;
      delete vr;
    }
  } else {
    // OLD version
    if (r1->rank() < r2->rank()) {
      // newA = a1, newB = b2.t^tmp
      newA = a1->copy();
      if (trans1 == 'C') newA->conjugate();
      newB = new ScalarArray<T>(b2->rows, r1->rank());
      if (trans2 == 'C') {
        newB->gemm('N', 'C', 1, b2, &tmp, 0);
        newB->conjugate();
      } else {
        newB->gemm('N', 'T', 1, b2, &tmp, 0);
      }
    } else { // newA = a1.tmp, newB = b2
      newA = new ScalarArray<T>(a1->rows, r2->rank());
      if (trans1 == 'C') tmp.conjugate(); // be careful if you re-use tmp after this...
      newA->gemm('N', 'N', 1, a1, &tmp, 0);
      if (trans1 == 'C') newA->conjugate();
      newB = b2->copy();
      if (trans2 == 'C') newB->conjugate();
    }
  }
  return new RkMatrix<T>(newA, ((trans1 == 'N') ? r1->rows : r1->cols), newB, ((trans2 == 'N') ? r2->cols : r2->rows));
}

template<typename T>
void RkMatrix<T>::multiplyWithDiagOrDiagInv(const HMatrix<T> * d, bool inverse, Side side) {
  assert(*d->rows() == *d->cols());
  assert(side == Side::RIGHT || (*rows == *d->cols()));
  assert(side == Side::LEFT  || (*cols == *d->rows()));

  // extracting the diagonal
  Vector<T>* diag = new Vector<T>(d->cols()->size());
  d->extractDiagonal(diag->ptr());

  // left multiplication by d of b (if M<-M*D : side = RIGHT) or a (if M<-D*M : side = LEFT)
  ScalarArray<T>* aOrB = (side == Side::LEFT ? a : b);
  aOrB->multiplyWithDiagOrDiagInv(diag, inverse, Side::LEFT);

  delete diag;
}

template<typename T> void RkMatrix<T>::gemmRk(double epsilon, char transHA, char transHB,
                                              T alpha, const HMatrix<T>* ha, const HMatrix<T>* hb) {
  DECLARE_CONTEXT;
  if (!ha->isLeaf() && !hb->isLeaf()) {
    // Recursion case
    int nbRows = transHA == 'N' ? ha->nrChildRow() : ha->nrChildCol() ; /* Row blocks of the product */
    int nbCols = transHB == 'N' ? hb->nrChildCol() : hb->nrChildRow() ; /* Col blocks of the product */
    int nbCom  = transHA == 'N' ? ha->nrChildCol() : ha->nrChildRow() ; /* Common dimension between A and B */
    int nSubRks = nbRows * nbCols;
    std::vector<RkMatrix<T>*> subRks(nSubRks, nullptr);
    for (int i = 0; i < nbRows; i++) {
      for (int j = 0; j < nbCols; j++) {
        int p = i + j * nbRows;
        for (int k = 0; k < nbCom; k++) {
          // C_ij = A_ik * B_kj
          HMatrix<T>* a_ik = transHA == 'N' ? ha->get(i, k) : ha->get(k, i);
          HMatrix<T>* b_kj = transHB == 'N' ? hb->get(k, j) : hb->get(j, k);
          if (a_ik && b_kj) {
            if (subRks[p] == nullptr) {
              const IndexSet* subRows = transHA == 'N' ? a_ik->rows() : a_ik->cols();
              const IndexSet* subCols = transHB == 'N' ? b_kj->cols() : b_kj->rows();
              subRks[p] = new RkMatrix<T>(nullptr, subRows, nullptr, subCols);
            }
            subRks[p]->gemmRk(epsilon, transHA, transHB, alpha, a_ik, b_kj);
          }
        } // k loop
      } // j loop
    } // i loop
    // Reconstruction of C by adding the parts
    // This test is not needed, it is there only to workaround bogus warning from GCC 12:
    //   error: ‘<unknown>’ may be used uninitialized [-Werror=maybe-uninitialized]
    if (nSubRks > 0) {
      std::vector<T> alphaV(nSubRks, 1);
      formattedAddParts(epsilon, alphaV.data(), subRks.data(), nSubRks);
    }
    for (int i = 0; i < nSubRks; i++) {
      delete subRks[i];
    }
  } else {
    RkMatrix<T>* rk = nullptr;
    // One of the product matrix is a leaf
    if (ha->isRecursivelyNull() || hb->isRecursivelyNull()) {
      // Nothing to do
    } else if (ha->isRkMatrix() || hb->isRkMatrix()) {
      rk = HMatrix<T>::multiplyRkMatrix(epsilon, transHA, transHB, ha, hb);
    } else {
      assert(ha->isFullMatrix() || hb->isFullMatrix());
      FullMatrix<T>* fullMat = HMatrix<T>::multiplyFullMatrix(transHA, transHB, ha, hb);
      if(fullMat) {
        rk = acaFull(fullMat, epsilon);
        delete fullMat;
      }
    }
    if(rk) {
      if(rank() == 0) {
        // save memory by not allocating a temporary Rk
        rk->scale(alpha);
        swap(*rk);
      } else
        axpy(epsilon, alpha, rk);
      delete rk;
    }
  }
}

template<typename T> void RkMatrix<T>::copy(const RkMatrix<T>* o) {
  delete a;
  delete b;
  rows = o->rows;
  cols = o->cols;
  a = (o->a ? o->a->copy() : nullptr);
  b = (o->b ? o->b->copy() : nullptr);
}

template<typename T> RkMatrix<T>* RkMatrix<T>::copy() const {
  RkMatrix<T> *result = new RkMatrix<T>(nullptr, rows, nullptr, cols);
  result->copy(this);
  return result;
}


template<typename T> void RkMatrix<T>::checkNan() const {
  if (rank() == 0) {
    return;
  }
  a->checkNan();
  b->checkNan();
}

template<typename T> void RkMatrix<T>::conjugate() {
  if (a) a->conjugate();
  if (b) b->conjugate();
}

template<typename T> T RkMatrix<T>::get(int i, int j) const {
  return a->dot_aibj(i, *b, j);
}

template<typename T> void RkMatrix<T>::writeArray(hmat_iostream writeFunc, void * userData) const{
  a->writeArray(writeFunc, userData);
  b->writeArray(writeFunc, userData);
}

template <typename T>
bool (*RkMatrix<T>::formatedAddPartsHook)(RkMatrix<T> *me, double epsilon, const T *alpha,
                                               const RkMatrix<T> *const *parts,
                                               const int n) = nullptr;

// Templates declaration
template class RkMatrix<S_t>;
template class RkMatrix<D_t>;
template class RkMatrix<C_t>;
template class RkMatrix<Z_t>;

}  // end namespace hmat
