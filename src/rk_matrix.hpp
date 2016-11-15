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
#include <vector>
#include <algorithm>
#include <utility>

#include "full_matrix.hpp"
#include "compression.hpp"

namespace hmat {

template<typename T> class HMatrix;
template<typename T> class AssemblyFunction;
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
  int k; /// If != 0, fixed-rank approximation
  double assemblyEpsilon; /// Tolerance for the assembly
  double recompressionEpsilon; /// Tolerance for the recompressions
  CompressionMethod method;
  int compressionMinLeafSize;

  /** Initialization with impossible values by default
   */
  RkApproximationControl() : k(0), assemblyEpsilon(-1.),
                             recompressionEpsilon(-1.), method(Svd), compressionMinLeafSize(100) {}
  /** Returns the number of singular values to keep.

       The stop criterion is (assuming that the singular value
       are in descending order):
           sigma [k] / SUM (sigma) <epsilon

       \param sigma table of singular values at least maxK elements.
       \param maxK maximum number of singular values to keep.
       \param epsilon tolerance.
       \return int the number of singular values to keep.

       note : the parameters maxK and sigma seem have contradictory explanation
   */
  int findK(double *sigma, int maxK, double epsilon);
};


template<typename T> class RkMatrix {

  /**  Swaps members of two RkMatrix instances.
       Since rows and cols are constant, they cannot be swaped and
       the other instance must have the same members.

       \param other  other RkMatrix instance.
  */
  void swap(RkMatrix<T>& other);

public:
  const IndexSet *rows;
  const IndexSet *cols;
  // A B^t
  ScalarArray<T>* a;
  ScalarArray<T>* b;
  CompressionMethod method; /// Method used to compress this RkMatrix

public:
  /// Control of the approximation. See \a RkApproximationControl for more
  /// details.
  static RkApproximationControl approx;

  // Type double precision associated to T
  typedef typename Types<T>::dp dp_t;

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
           ScalarArray<T>* _b, const IndexSet* _cols,
           CompressionMethod _method);
  ~RkMatrix();

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
  /** \brief Returns number of allocated zeros
   */
  size_t storedZeros();
  /** Returns the compression ratio (stored_elements, total_elements).
   */
  size_t compressedSize();

  size_t uncompressedSize();

  /** Returns a pointer to a new matrix M = AB^t (uncompressed)
   */
  FullMatrix<T>* eval() const;
  /** Recompress an RkMatrix in place.

      @warning The previous rk->a and rk->b are no longer valid after this function.
   */
  void truncate(double epsilon);
  /*! \brief Return square of the Frobenius norm of the matrix.

    \return the matrix norm.
   */
  double normSqr() const;
  /** this <- this + alpha * mat

      \param alpha
      \param mat
   */
  void axpy(T alpha, const FullMatrix<T>* mat);
  /** this <- this + alpha * mat

      \param alpha
      \param mat
   */
  void axpy(T alpha, const RkMatrix<T>* mat);
  /** Formatted addition of two Rk-matrices.

      The two matrices must be on the same sets of indices in the case
      otherwise use RkMatrix::formattedAddParts. The formatted addition of R
      and S is defined by:
       \code
       truncate (R + S)
       \endcode
      with the addition defined by the juxtaposition of the matrices A and B of
      each RkMatrix component of the product.

      \param o The matrix sum
      \return truncate(*this + m) A new matrix.
   */
  RkMatrix<T> *formattedAdd(const FullMatrix<T>* o, T alpha = Constants<T>::pone) const;
  /** Addition of a matrix and formatted a RkMatrix.

      The two matrices must be on the same sets of indices in the case
      otherwise use RkMatrix::formattedAddParts. The formatted addition of R
      and S is defined by:
       \code
       truncate(R + S)
       \endcode
      with the addition defined by the juxtaposition of the matrices A and B of
      RkMatrix each component of the product.

      \param o The matrix sum
      \return truncate(*this + m) A new matrix.
   */
  RkMatrix<T> *formattedAdd(const RkMatrix<T>* o) const;
  /** Adds a list of RkMatrix to a RkMatrix.

      In this function, RkMatrix may include some
      only indices, that is to say be subsets of
      evidence of this.

      \param units The list RkMatrix adding.
      \param n Number of matrices to add
      \return truncate(*this + parts[0] + parts[1] + ... + parts[n-1])
   */
  RkMatrix<T>* formattedAddParts(T* alpha, const RkMatrix<T>** parts, int n) const;
  /** Adds a list of MatrixXd (solid matrices) to RkMatrix.

      In this function, MatrixXd may cover a portion of
      index only, that is to say be subsets of indices
      this.
      The MatrixXd are converted RkMatrix before the addition, which is
      with RkMatrix :: formattedAddParts.

      \param units The list MatrixXd adding.
      \param rowsList The list of indices rows
      \param colsList The list of column indices
      \param n Number of matrices to add
      \return truncate (*this + parts[0] + parts[1] + ... + parts[n-1])
   */
  RkMatrix<T>* formattedAddParts(T* alpha, const FullMatrix<T>** parts,
                                 const IndexSet** rowsList,
                                 const IndexSet** colsList, int n) const;
  void gemmRk(char transA, char transB, T alpha, const HMatrix<T>* a, const HMatrix<T>* b, T beta);

  /** Multiplication by a scalar.

       \param alpha the scalar
   */
  void scale(T alpha);
  void clear();
  /** Copy  RkMatrix into this.
   */
  void copy(RkMatrix<T>* o);

  /** Compute y <- alpha * op(A) * y + beta * y with x and y FullMatrix<T>*

      The arguments are similar to BLAS GEMV.
   */
  void gemv(char trans, T alpha, const FullMatrix<T>* x, T beta, FullMatrix<T>* y) const;

  /** Compute y <- alpha * op(A) * y + beta * y with x and y ScalarArray<T>*

      The arguments are similar to BLAS GEMV.
   */
  void gemv(char trans, T alpha, const ScalarArray<T>* x, T beta, ScalarArray<T>* y) const;

  /**  Right multiplication of RkMatrix by a matrix.

       \param transR 'N' or 'T' depending on whether R is transposed or not
       \param Beam 'N' or 'T' depending on whether M is transposed or not
       \return R * M
  */
  static RkMatrix<T>* multiplyRkFull(char transR, char transM,
                                     const RkMatrix<T>* rk, const FullMatrix<T>* m,
                                     const IndexSet* mCols);

  /** Left multiplication of RkMatrix by a matrix.

       \param transR 'N' or 'T' depending on whether R is transposed or not
       \param Beam 'N' or 'T' depending on whether M is transposed or not
       \return R * M

   */
  static RkMatrix<T>* multiplyFullRk(char transM, char transR,
                                     const FullMatrix<T>* m,
                                     const RkMatrix<T>* rk,
                                     const IndexSet* mRows);
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
  static RkMatrix<T>* multiplyRkRk(char transA, char transB, const RkMatrix<T>* a, const RkMatrix<T>* b);
  /*! \brief in situ multiplication of the matrix by the diagonal of the matrix given as argument

     \param d D matrix which we just considered the diagonal
     \param inverse D or D^{-1}
     \param left multiplication (left = true) or right (left = false) of this by D
  */
  void multiplyWithDiagOrDiagInv(const HMatrix<T> * d, bool inverse, bool left = false);
  /*! \brief Triggers an assertion if there are NaNs in the RkMatrix.
   */
  void checkNan() const;

  /** Return the memory size of the a*b product */
  static size_t computeRkRkMemorySize(char transA, char transB, const RkMatrix<T>* a, const RkMatrix<T>* b);
};

}  // end namespace hmat

#endif
