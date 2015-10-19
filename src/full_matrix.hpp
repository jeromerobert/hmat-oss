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
#include "h_matrix.hpp"

namespace hmat {

/*! \brief Templated dense Matrix type.

  The template parameter represents the scalar type of the matrix elements.  The
  supported types are \a S_t, \a D_t, \a C_t and \a Z_t, as defined in
  @data_types.hpp.
 */
template<typename T> class FullMatrix {
  /*! True if the matrix owns its memory, ie has to free it upon destruction */
  char ownsMemory:1;
  /*! Is this matrix upper triangular? */
  char triUpper_:1;
  /*! Is this matrix lower triangular? */
  char triLower_:1;
  /// Disallow the copy
  FullMatrix(const FullMatrix<T>& o);

public:
  /// Fortran style pointer (columnwise)
  T* m;
  /// Number of rows
  int rows;
  /// Number of columns
  int cols;
  /*! Leading dimension, as in BLAS */
  int lda;
  /*! Holds the pivots for the LU decomposition. */
  int* pivots;
  /*! Diagonal in an LDL^t factored matrix */
  Vector<T>* diagonal;

  /** \brief Initialize the matrix with existing data.

      In this case the matrix doesn't own the data (the memory is not
      freed at the object destruction).

      \param _m Pointer to the data
      \param _rows Number of rows
      \param _cols Number of cols
      \param lda Leading dimension, as in BLAS
   */
  FullMatrix(T* _m, int _rows, int _cols, int _lda=-1);
  /* \brief Create an empty matrix, filled with 0s.

     In this case, the memory is freed when the object is destroyed.

     \param _rows Number of rows
     \param _cols Number of columns
   */
  FullMatrix(int _rows, int _cols);
  /** \brief Create a matrix filled with 0s.

     In this case, the memory is freed when the object is destroyed.

     \param _rows Number of rows
     \param _cols Number of columns
   */
  static FullMatrix* Zero(int rows, int cols);
  ~FullMatrix();

  bool isTriUpper() {
      return triUpper_;
  }

  bool isTriLower() {
      return triLower_;
  }

  /** This <- 0.
   */
  void clear();
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
  void solveLowerTriangularLeft(FullMatrix<T>* x, bool unitriangular) const;
  /*! \brief Solve the system X U = B, with B = X on entry, and U = this.

    This function requires the matrix to be factored by
    HMatrix::luDecomposition() beforehand.

    \param x B on entry, the solution on exit.
   */
  void solveUpperTriangularRight(FullMatrix<T>* x, bool unitriangular, bool lowerStored) const;
  /*! \brief Solve the system U X = B, with B = X on entry, and U = this.

    This function requires the matrix to be factored by
    HMatrix::luDecomposition() beforehand.

    \param x B on entry, the solution on exit.
   */
  void solveUpperTriangularLeft(FullMatrix<T>* x, bool unitriangular, bool lowerStored) const;
  /*! \brief Solve the system U X = B, with B = X on entry, and U = this.

    This function requires the matrix to be factored by
    HMatrix::luDecomposition() beforehand.

    \param x B on entry, the solution on exit.
   */
  void solve(FullMatrix<T>* x) const;
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
  /** Simpler accessors for the data.

      There are 2 types to allow matrix modification or not.
   */
  T& get(int i, int j) {
    return m[i + ((size_t) lda) * j];
  }
  T get(int i, int j) const {
    return m[i + ((size_t) lda) * j];
  }
  /** Simpler accessors for the diagonal.

      \warning This will only work if the matrix has been factored
      with \a FullMatrix::ldltDecomposition() beforehand.
   */
  T getD(int i) const {
    return diagonal->v[i];
  }
  T& getD(int i) {
    return diagonal->v[i];
  }
  /*! Check the matrix for the presence of NaN numbers.

    If a NaN is found, an assertion is triggered.
   */
  void checkNan() const;
  size_t memorySize() const;
};


/** \brief Wraps a FullMatrix backed by a file (using mmap()).
 */
template<typename T> class MmapedFullMatrix {
public:
  FullMatrix<T> m;
private:
  void* mmapedFile;
  int fd;
  size_t size;

public:
  /** \brief Maps a .res file into memory.

      The mapping is read-only.

      \param filename The filename
      \return an instance of MMapedFullMatrix<T> mapping the file.
   */
  static MmapedFullMatrix<T>* fromFile(const char* filename);
  /** \brief Creates a FullMatrix backed by a file, wrapped into a MMapedFullMatrix<T>.

      \param rows number of rows of the matrix.
      \param cols number of columns of the matrix.
      \param filename Filename. The file is not destroyed with the object.
   */
  MmapedFullMatrix(int rows, int cols, const char* filename);
  ~MmapedFullMatrix();

private:
  MmapedFullMatrix() : m(NULL, 0, 0), mmapedFile(NULL), fd(-1), size(0) {};
};


/*! \brief Templated Vector class.

  As for \a FullMatrix, the template parameter is the scalar type.
 */
template<typename T> class Vector {
private:
  /// True if the vector owns its memory
  bool ownsMemory;

public:
  /// Pointer to the data
  T* v;
  /// Rows count
  int rows;

public:
  Vector(T* _v, int _rows);
  Vector(int _rows);
  ~Vector();
  /** Create a vector filled with 0s.
   */
  static Vector<T>* Zero(int rows);
  /**  this = alpha * A * x + beta * this

       Wrapper around the 'GEMV' BLAS call.

       \param trans 'N' or 'T', as in BLAS
       \param alpha alpha
       \param a A
       \param x X
       \param beta beta
   */
  void gemv(char trans, T alpha, const FullMatrix<T>* a, const Vector<T>* x,
            T beta);
  /** \brief this += v
   */
  void addToMe(const Vector<T>* x);
  /** \brief this -= v
   */
  void subToMe(const Vector<T>* x);
  /** L2 norm of the vector.
   */
  double norm() const;
  /** Set the vector to 0.
   */
  void clear();
  /** This *= alpha.
   */
  void scale(T alpha);
  /** Compute This += alpha X, X a vector.
   */
  void axpy(T alpha, const Vector<T>* x);
  /** \brief Return the index of the maximum absolute value element.

      This function is similar to iXamax() in BLAS.

      \return the index of the maximum absolute value element in the vector.
   */
  int absoluteMaxIndex() const;
  /** Compute the dot product of two \Vector.

      For real-valued vectors, this is the usual dot product. For
      complex-valued ones, this is defined as:
         <x, y> = \bar{x}^t \times y
      as in BLAS

      \warning DOES NOT work with vectors with >INT_MAX elements

      \param x
      \param y
      \return <x, y>
   */
  static T dot(const Vector<T>* x, const Vector<T>* y);

private:
  /// Disallow the copy
  Vector<T>(const Vector<T>& o);
};

/*! \brief Wrapper around BLAS copy function.

  \param n Number of elements to copy
  \param from Source
  \param incFrom increment between elements in from
  \param to Destination
  \paarm incTo increment between elenents in to
 */
template<typename T> void blasCopy(int n, T* from, int incFrom, T* to, int incTo);

}  // end namespace hmat

#endif
