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

#ifndef _RK_MATRIX_HPP
#define _RK_MATRIX_HPP
/* Implementation of Rk-matrices */

#include "full_matrix.hpp"
#include "compression.hpp"
#include "common/my_assert.h"


#include <stdbool.h>



namespace hmat {



  
template<typename T>
struct FPAdaptiveCompressor{
    int nb_blocs;
    int n_rows_A;
    int n_rows_B;
    int n_cols;
    std::vector<FPCompressorInterface<T>*> compressors_A;
    std::vector<FPCompressorInterface<T>*> compressors_B;

    std::vector<int> cols;
    
    double compressionRatio;
    double compressionTime;
    double decompressionTime;

    FPAdaptiveCompressor(hmat_FPcompress_t method = hmat_FPcompress_t::DEFAULT_COMPRESSOR, int n = 1);

    ~FPAdaptiveCompressor();

    FPAdaptiveCompressor* copy() {
      FPAdaptiveCompressor* newComp = new FPAdaptiveCompressor();
      newComp->nb_blocs = nb_blocs;
      newComp->n_rows_A = n_rows_A;
      newComp->n_rows_B = n_rows_B;
      newComp->n_cols = n_cols;
      newComp->compressionRatio = compressionRatio;
      newComp->compressionTime = compressionTime;
      newComp->decompressionTime = decompressionTime;
      
      newComp->cols = cols;

      newComp->compressors_A.resize(nb_blocs);
      newComp->compressors_B.resize(nb_blocs);

      for(int i =0; i < nb_blocs; i++)
      {
        newComp->compressors_A[i] = compressors_A[i]->copy();
        newComp->compressors_B[i] = compressors_B[i]->copy();
      }

      return newComp;
    }
};



template<typename T> class HMatrix;
template<typename T, template <typename> class F> class AssemblyFunction;
class ClusterData;
class IndexSet;

/** Control the approximation of Rk-matrices.

     In the case where k != 0, we do an approximation with a fixed rank k,
     otherwise the approximation is adaptive, it stops when the
     singular value is less than an error relative to the
     sum of the singular values in the SVD.
 */
class RkApproximationControl {
public:
  double coarseningEpsilon; /// Tolerance for the coarsening
  int compressionMinLeafSize;

  /** Initialization with impossible values by default
   */
  RkApproximationControl() : coarseningEpsilon(-1.),
                             compressionMinLeafSize(100) {}
};


template<typename T> class RkMatrix {
  // Type double precision associated to T
  typedef typename Types<T>::dp dp_t;
  /**  Swaps members of two RkMatrix instances.
       Since rows and cols are constant, they cannot be swaped and
       the other instance must have the same members.

       \param other  other RkMatrix instance.
  */
  void swap(RkMatrix<T>& other);
  /** Recompress an RkMatrix in place with a modified Gram-Schmidt algorithm.

      @warning The previous rk->a and rk->b are no longer valid after this function.
      \param epsilon is the accuracy of the recompression
      \param initialPivotA/B is the number of orthogonal columns in panels a and b
   */
  void mGSTruncate(double epsilon, int initialPivotA=0, int initialPivotB=0);
public:
  /** @brief A hook which can be called at the begining of formatedAddParts */
  static bool (*formatedAddPartsHook)(RkMatrix<T> * me, double epsilon, const T* alpha, const RkMatrix<T>* const * parts, const int n);
  const IndexSet *rows;
  const IndexSet *cols;

  // A B^t
  ScalarArray<T>* a;
  ScalarArray<T>* b;
  FPAdaptiveCompressor<T>* _compressors;
  /// Control of the approximation. See \a RkApproximationControl for more
  /// details.
  static RkApproximationControl approx;

  /** Construction of a RkMatrix .

       A Rk-matrix is a compressed representation of a matrix of size
       n*m by a matrix R = AB^t with A a n*k matrix and B is a m*k matrix.
       The represented matrix R corresponds to a set of indices _rows and _cols.

       \param _a matrix A
       \param _rows indices of the rows (of size n)
       \param _b matrix B (not B^t)
       \param _cols indices of the columns (of size k)
   */
  RkMatrix(ScalarArray<T>* _a, const IndexSet* _rows,
           ScalarArray<T>* _b, const IndexSet* _cols);
  ~RkMatrix();

  RkMatrix(const RkMatrix& other) = delete;
  
  RkMatrix& operator=(const RkMatrix& other) = delete;

  int rank() const {
      return a ? a->cols : 0;
  }

  /**  Returns a pointer to a new RkMatrix representing a subset of indices.
       The pointer is supposed to be read-only (for efficiency reasons).

       \param subRows subset of rows
       \param subCols subset of columns
       \return pointer to a new matrix with subRows and subCols.
   */
  const RkMatrix* subset(const IndexSet* subRows, const IndexSet* subCols) const;
  RkMatrix* truncatedSubset(const IndexSet* subRows, const IndexSet* subCols, double epsilon) const;
  /** Returns the compression ratio (stored_elements, total_elements).
   */
  size_t compressedSize();

  size_t uncompressedSize();

  /** Returns a pointer to a new FullMatrix M = AB^t (uncompressed)
   */
  FullMatrix<T>* eval() const;
  /** Returns a pointer to a new ScalarArray M = AB^t (uncompressed) or fill an existing one
   */
  ScalarArray<T>* evalArray(ScalarArray<T> *result=NULL) const ;
  /** Recompress an RkMatrix in place.

      @warning The previous rk->a and rk->b are no longer valid after this function.
      \param initialPivotA/B is the number of orthogonal columns in panels a and b
   */
  
  /** Compress the panels of a RkMatrix using FP Compression.*/
  void FPcompress(double epsilon, int nb_blocs, hmat_FPcompress_t method = hmat_FPcompress_t::DEFAULT_COMPRESSOR, Vector<typename Types<T>::real> *sigma=NULL);

/** Decompress the panels of a RkMatrix after FP Compression.*/
  void FPdecompress();


  /** Decompress the panels of a RkMatrix after FP compression into a copy of the original RkMatrix */
  RkMatrix<T>* FPdecompressCopy(RkMatrix<T>* result = NULL) const;

  /** Return True iff the rk matrix is FP compressed */
  bool isFPcompressed() const;


  void truncate(double epsilon, int initialPivotA=0, int initialPivotB=0);
  /** Recompress an RkMatrix in place using RRQR */
  void truncateAlter(double epsilon);
  /** Recompress an RKMatrix in place and validate using an alternative RRQR-based method */
  void validateRecompression(double epsilon , int initialPivotA , int initialPivotB);
  /** Add randomness to the RkMatrix */
  void addRand(double epsilon);
  /*! \brief Return square of the Frobenius norm of the matrix.

    \return the matrix norm.
   */
  double normSqr() const;
  /** this <- this + alpha * mat

      \param alpha
      \param mat
   */
  void axpy(double epsilon, T alpha, const FullMatrix<T>* mat);
  /** this <- this + alpha * mat

      \param alpha
      \param mat
   */
  void axpy(double epsilon, T alpha, const RkMatrix<T>* mat);
  /** Adds a list of RkMatrix to a RkMatrix.

      In this function, RkMatrix may include some
      only indices, that is to say be subsets of
      evidence of this.

      \param units The list RkMatrix adding.
      \param n Number of matrices to add
      \param epsilon truncate epsilon (negative value disable truncate)
      \param hook true to call formattedAddPartsHook, false to do not call
      \return truncate(*this + parts[0] + parts[1] + ... + parts[n-1])
   */
  void formattedAddParts(double epsilon, const T* alpha, const RkMatrix<T>* const * parts, const int n,
                                 bool hook = true);
  /** Adds a list of MatrixXd (solid matrices) to RkMatrix.

      In this function, MatrixXd may cover a portion of
      index only, that is to say be subsets of indices
      this.
      The MatrixXd are converted RkMatrix before the addition, which is
      with RkMatrix :: formattedAddParts.

      \param units The list MatrixXd adding.
      \param n Number of matrices to add
      \return truncate (*this + parts[0] + parts[1] + ... + parts[n-1])
   */
  void formattedAddParts(double epsilon, const T* alpha, const FullMatrix<T>* const * parts, int n);

  /*! \brief Add a product of HMatrix to an RkMatrix
     */
  void gemmRk(double epsilon, char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>* b);

  /** Multiplication by a scalar.

       \param alpha the scalar
   */
  void scale(T alpha);
  /** \brief Transpose in place.
   */
  void transpose();
  void clear();

  /** Copy  RkMatrix into this.
   */
  void copy(const RkMatrix<T>* o);

  /** Return a copy of this.
   */
  RkMatrix<T>* copy() const;

  /** Compute y <- alpha * op(A) * y + beta * y if side is Side::LEFT
           or y <- alpha * y * op(A) + beta * y if side is Side::RIGHT
      with x and y ScalarArray<T>*.

      The arguments are similar to BLAS GEMV.
   */
  void gemv(char trans, T alpha, const ScalarArray<T>* x, T beta, ScalarArray<T>* y, Side side = Side::LEFT) const;

  /**  Right multiplication of RkMatrix by a matrix.

       \param transR 'N' or 'T' depending on whether R is transposed or not
       \param Beam 'N' or 'T' depending on whether M is transposed or not
       \return R * M
  */
  static RkMatrix<T>* multiplyRkFull(char transR, char transM,
                                     const RkMatrix<T>* rk, const FullMatrix<T>* m);

  /** Left multiplication of RkMatrix by a matrix.

       \param transR 'N' or 'T' depending on whether R is transposed or not
       \param Beam 'N' or 'T' depending on whether M is transposed or not
       \return R * M

   */
  static RkMatrix<T>* multiplyFullRk(char transM, char transR,
                                     const FullMatrix<T>* m,
                                     const RkMatrix<T>* rk);
  /* These functions are added to manage the particular case of the product by
      H-matrix, which is treated by decomposing the product into the succession of
      products by a vector, the result being a RkMatrix.
   */
  /** Right multiplying of RkMatrix by HMatrix.

       The product is made by multiplying the vector by vector.

       \param R rk
       \param h H
       \param transRk 'N' or 'T' depending on whether or not R is transposed
       \param transh 'N' or 'T' depending on whether or not H is transposed
       \return R * H
   */
  static RkMatrix<T>* multiplyRkH(char transRk, char transH, const RkMatrix<T>* rk, const HMatrix<T>* h);
  /**  Left multiplying a RkMatrix by HMatrix.

       The product is made by multiplying vector by vector.

       \param h H
       \param R rk
       \param transR 'N' or 'T' depending on whether or not R is transposed
       \param transh 'N' or 'T' depending on whether or not H is transposed
       \return H * R
  */
  static RkMatrix<T>* multiplyHRk(char transH, char transR, const HMatrix<T>* h, const RkMatrix* rk);
  /** Multiplying a RkMatrix by a RkMatrix

       The product is made by multiplying vector by vector.

       \param a
       \param b
       \return A * B
  */
  static RkMatrix<T>* multiplyRkRk(char transA, char transB, const RkMatrix<T>* a, const RkMatrix<T>* b, double epsilon);
  /*! \brief in situ multiplication of the matrix by the diagonal of the matrix given as argument

     \param d D matrix which we just considered the diagonal
     \param inverse D or D^{-1}
     \param left multiplication (side = Side::LEFT) or right (side = Side::RIGHT) of this by D
  */
  void multiplyWithDiagOrDiagInv(const HMatrix<T> * d, bool inverse, Side side = Side::RIGHT);
  /*! \brief Triggers an assertion if there are NaNs in the RkMatrix.
   */
  void checkNan() const;

  /*! Conjugate the content of the complex matrix */
  void conjugate();

  /** Simpler accessor for the data.

      There is only 1 type to because we cant modify the data.
   */
  T get(int i, int j) const ;

  /*! \brief Write the RkMatrix data 'a' and 'b' in a stream (FILE*, unix fd, ...)
    */
  void writeArray(hmat_iostream writeFunc, void * userData) const;




};

}  // end namespace hmat

#endif
