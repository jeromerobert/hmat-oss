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

/*! \file
  \ingroup HMatrix
  \brief Dense Matrix type used by the HMatrix library.
*/
#ifndef _FULL_MATRIX_HPP
#define _FULL_MATRIX_HPP
#include <cstddef>

#include "data_types.hpp"
#include "scalar_array.hpp"
#include "h_matrix.hpp"

namespace hmat {

class IndexSet;

  /*! \brief Templated dense Matrix type.

  The template parameter represents the scalar type of the matrix elements.  The
  supported types are \a S_t, \a D_t, \a C_t and \a Z_t, as defined in
  @data_types.hpp.
 */
template<typename T> class FullMatrix {
public:
  ScalarArray<T> data;
private:
  /*! Is this matrix upper triangular? */
  char triUpper_:1;
  /*! Is this matrix lower triangular? */
  char triLower_:1;
  /// Disallow the copy
  FullMatrix(const FullMatrix<T>& o);

public:
  const IndexSet *rows_;
  const IndexSet *cols_;

public:
  /*! Holds the pivots for the LU decomposition. */
  int* pivots;
  /*! Diagonal in an LDL^t factored matrix */
  Vector<T>* diagonal;

  /** \brief Initialize the matrix with existing data.

      In this case the matrix doesn't own the data (the memory is not
      freed at the object destruction).
       The represented matrix R corresponds to a set of indices _rows and _cols.

      \param _m Pointer to the data
      \param _rows indices of the rows
      \param _cols indices of the columns
      \param lda Leading dimension, as in BLAS
   */
  FullMatrix(T* _m, const IndexSet*  _rows, const IndexSet*  _cols, int _lda=-1);
  /** \brief Initialize the matrix with an existing ScalarArray and 2 IndexSets.

      \param _s a ScalarArray
      \param _rows indices of the rows
      \param _cols indices of the columns
   */
  FullMatrix(ScalarArray<T> *s, const IndexSet*  _rows, const IndexSet*  _cols);
  /** \brief Create an empty matrix, filled with 0s.

     In this case, the memory is freed when the object is destroyed.

      \param _rows indices of the rows
      \param _cols indices of the columns
   */
  FullMatrix(const IndexSet*  _rows, const IndexSet*  _cols);
  ~FullMatrix();

  bool isTriUpper() {
      return triUpper_;
  }

  bool isTriLower() {
      return triLower_;
  }

  int rows() const {
    assert(data.rows==this->rows_->size());
    return data.rows;
  }
  int cols() const {
    assert(data.cols==this->cols_->size());
    return data.cols;
  }

  /** This <- 0.
   */
  void clear();
  /** \brief Returns number of allocated zeros
   */
  size_t storedZeros();
  /** \brief Returns some info about block, like size of square that fit nnz locality (i.e all null rows/cols ignored)
   */
  size_t info(hmat_info_t & result, size_t& rowsMin, size_t& colsMin, size_t& rowsMax, size_t& colsMax);
  /** \brief this *= alpha.

      \param alpha The scaling factor.
   */
  void scale(T alpha);
  /** \brief Transpose in place.
   */
  void transpose();
  /** Return a copy of this.
   */
  FullMatrix<T>* copy(FullMatrix<T>* result = NULL) const;
  /** \brief Return a new matrix that is a transposed version of this.
   */
  FullMatrix<T>* copyAndTranspose() const;

  /**  Returns a pointer to a new FullMatrix representing a subset of indices.
       The pointer is supposed to be read-only (for efficiency reasons).

       \param subRows subset of rows
       \param subCols subset of columns
       \return pointer to a new matrix with subRows and subCols.
   */
  const FullMatrix<T>* subset(const IndexSet* subRows, const IndexSet* subCols) const ;

  /** this = alpha * op(A) * op(B) + beta * this

      Standard "GEMM" call, as in BLAS.

      \param transA 'N' or 'T', as in BLAS
      \param transB 'N' or 'T', as in BLAS
      \param alpha alpha
      \param a the matrix A
      \param b the matrix B
      \param beta beta
   */
  void gemm(char transA, char transB, T alpha, const FullMatrix<T>* a,
            const FullMatrix<T>* b, T beta);
  /*! \brief B <- B*D or B <- B*D^-1  (or with D on the left).

    B = this, and D a diagonal matrix.

     \param d  D
     \param inverse true : B<-B*D^-1, false B<-B*D
     \param left true : B<-D*B, false B<-B*D
  */
  void multiplyWithDiagOrDiagInv(const Vector<T>* d, bool inverse, bool left = false);
  /*! \brief Compute a LU factorization in place.
   */
  void luDecomposition();
  /*! \brief Compute a LDLt factorization in place.
   */
  void ldltDecomposition();
  /*! \brief Compute a LLt factorization in place (aka Cholesky).
   */
  void lltDecomposition();
  /*! \brief Solve the system L X = B, with B = X on entry, and L = this.

    This function requires the matrix to be factored by
    HMatrix::luDecomposition() beforehand.

    \param x B on entry, the solution on exit.
   */
  void solveLowerTriangularLeft(ScalarArray<T>* x, bool unitriangular) const;
  /*! \brief Solve the system X U = B, with B = X on entry, and U = this.

    This function requires the matrix to be factored by
    HMatrix::luDecomposition() beforehand.

    \param x B on entry, the solution on exit.
   */
  void solveUpperTriangularRight(ScalarArray<T>* x, bool unitriangular, bool lowerStored) const;
  /*! \brief Solve the system U X = B, with B = X on entry, and U = this.

    This function requires the matrix to be factored by
    HMatrix::luDecomposition() beforehand.

    \param x B on entry, the solution on exit.
   */
  void solveUpperTriangularLeft(ScalarArray<T>* x, bool unitriangular, bool lowerStored) const;
  /*! \brief Solve the system U X = B, with B = X on entry, and U = this.

    This function requires the matrix to be factored by
    HMatrix::luDecomposition() beforehand.

    \param x B on entry, the solution on exit.
   */
  void solve(ScalarArray<T>* x) const;
  /*! \brief Compute the inverse of this in place.
   */
  void inverse();
  /*! Copy a matrix A into 'this' at offset (rowOffset, colOffset) (indices start at 0).

    \param a the matrix A
    \param rowOffset the row offset
    \param colOffset the column offset
   */
  void copyMatrixAtOffset(const FullMatrix<T>* a, int rowOffset, int colOffset);
  /*! Copy a matrix A into 'this' at offset (rowOffset, colOffset) (indices start at 0).

    In this function, only copy a sub-matrix of size (rowsToCopy, colsToCopy).

    \param a the matrix A
    \param rowOffset the row offset
    \param colOffset the column offset
    \param rowsToCopy number of rows to copy
    \param colsToCopy number of columns to copy
   */
  void copyMatrixAtOffset(const FullMatrix<T>* a, int rowOffset, int colOffset,
                          int rowsToCopy, int colsToCopy);
  /*! \brief this += alpha * A

    \param a the Matrix A
   */
  void axpy(T alpha, const FullMatrix<T>* a);
  /*! \brief Return square of the Frobenius norm of the matrix.

    \return the matrix norm.
   */
  double normSqr() const;
  /*! \brief Return the Frobenius norm of the matrix.

    \return the matrix norm.
   */
  double norm() const;
  /*! \brief Write the matrix to a binary file.

    \param filename output filename
   */
  void toFile(const char *filename) const;

  void fromFile(const char * filename);
  /** Simpler accessors for the data.

      There are 2 types to allow matrix modification or not.
   */
  T& get(int i, int j) {
    return data.m[i + ((size_t) data.lda) * j];
  }
  T get(int i, int j) const {
    return data.m[i + ((size_t) data.lda) * j];
  }
  /** Simpler accessors for the diagonal.

      \warning This will only work if the matrix has been factored
      with \a FullMatrix::ldltDecomposition() beforehand.
   */
  T getD(int i) const {
    return diagonal->m[i];
  }
  T& getD(int i) {
    return diagonal->m[i];
  }
  /*! Check the matrix for the presence of NaN numbers.

    If a NaN is found, an assertion is triggered.
   */
  void checkNan() const;
  size_t memorySize() const;

  /*! \brief Return a short string describing the content of this FullMatrix for debug (like: "FullMatrix [320, 452]x[760, 890] norm=22.34758")
    */
  std::string description() const {
    std::ostringstream convert;
    convert << "FullMatrix " << this->rows_->description() << "x" << this->cols_->description() ;
    convert << "norm=" << norm();
    return convert.str();
  }
};

}  // end namespace hmat

#endif
