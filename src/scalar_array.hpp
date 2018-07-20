#pragma once
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
  \brief Scalar Array type used by the HMatrix library.
*/
#pragma once

#include <cstddef>

#include "data_types.hpp"

namespace hmat {

/*! \brief Templated dense Matrix type.

  The template parameter represents the scalar type of the matrix elements.  The
  supported types are \a S_t, \a D_t, \a C_t and \a Z_t, as defined in
  @data_types.hpp.
 */
template<typename T> class ScalarArray {
  /*! True if the matrix owns its memory, ie has to free it upon destruction */
  char ownsMemory:1;

public:
  /// Fortran style pointer (columnwise)
  T* m;
  /// Number of rows
  int rows;
  /// Number of columns
  int cols;
  /*! Leading dimension, as in BLAS */
  int lda;

  /** \brief Initialize the ScalarArray with existing ScalarArray.

      In this case the matrix doesn't own the data (the memory is not
      freed at the object destruction).

      \param d a ScalarArray
   */
  ScalarArray(const ScalarArray& d) : ownsMemory(false), m(d.m), rows(d.rows), cols(d.cols), lda(d.lda) {}
  /** \brief Initialize the matrix with existing data.

      In this case the matrix doesn't own the data (the memory is not
      freed at the object destruction).

      \param _m Pointer to the data
      \param _rows Number of rows
      \param _cols Number of cols
      \param lda Leading dimension, as in BLAS
   */
  ScalarArray(T* _m, int _rows, int _cols, int _lda=-1);
  /** \brief Create an empty matrix, filled with 0s.

     In this case, the memory is freed when the object is destroyed.

     \param _rows Number of rows
     \param _cols Number of columns
   */
  ScalarArray(int _rows, int _cols);
  ~ScalarArray();

  /** This <- 0.
   */
  void clear();
  /** \brief Returns number of allocated zeros
   */
  size_t storedZeros() const;
  /** \brief this *= alpha.

      \param alpha The scaling factor.
   */
  void scale(T alpha);
  /** \brief Transpose in place.
   */
  void transpose();
  /** \brief Conjugate in place.
   */
  void conjugate();
  /** Return a copy of this.
   */
  ScalarArray<T>* copy(ScalarArray<T>* result = NULL) const;
  /** \brief Return a new matrix that is a transposed version of this.

    This new matrix is created in \a result (if provided)
   */
  ScalarArray<T>* copyAndTranspose(ScalarArray<T>* result = NULL) const;
  /**  Returns a pointer to a new ScalarArray representing a subset of row indices.
       Columns and leading dimension are unchanged.
       \param rowOffset offset to apply on the rows
       \param rowSize new number of rows
       \return pointer to a new ScalarArray.
   */
  ScalarArray<T> rowsSubset(const int rowOffset, const int rowSize) const ;

  /** this = alpha * op(A) * op(B) + beta * this

      Standard "GEMM" call, as in BLAS.

      \param transA 'N' or 'T', as in BLAS
      \param transB 'N' or 'T', as in BLAS
      \param alpha alpha
      \param a the matrix A
      \param b the matrix B
      \param beta beta
   */
  void gemm(char transA, char transB, T alpha, const ScalarArray<T>* a,
            const ScalarArray<T>* b, T beta);
  /*! Copy a matrix A into 'this' at offset (rowOffset, colOffset) (indices start at 0).

    \param a the matrix A
    \param rowOffset the row offset
    \param colOffset the column offset
   */
  void copyMatrixAtOffset(const ScalarArray<T>* a, int rowOffset, int colOffset);
  /*! Copy a matrix A into 'this' at offset (rowOffset, colOffset) (indices start at 0).

    In this function, only copy a sub-matrix of size (rowsToCopy, colsToCopy).

    \param a the matrix A
    \param rowOffset the row offset
    \param colOffset the column offset
    \param rowsToCopy number of rows to copy
    \param colsToCopy number of columns to copy
   */
  void copyMatrixAtOffset(const ScalarArray<T>* a, int rowOffset, int colOffset,
                          int rowsToCopy, int colsToCopy);
  /*! \brief add term by term a random value

    \param epsilon  x *= (1 + a),  a = epsilon*(1.0-2.0*rand()/(double)RAND_MAX)
   */
  void addRand(double epsilon);
  /*! \brief this += alpha * A

    \param a the Matrix A
   */
  void axpy(T alpha, const ScalarArray<T>* a);
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
  inline T& get(int i, int j) {
    return m[i + ((size_t) lda) * j];
  }
  inline T get(int i, int j) const {
    return m[i + ((size_t) lda) * j];
  }
  /*! Check the matrix for the presence of NaN numbers.

    If a NaN is found, an assertion is triggered.
   */
  void checkNan() const;
  size_t memorySize() const;

  /*! \brief Return a short string describing the content of this ScalarArray for debug (like: "ScalarArray [320 x 100] norm=22.34758")
    */
  std::string description() const {
    std::ostringstream convert;   // stream used for the conversion
    convert << "ScalarArray [" << rows << " x " << cols << "] norm=" << norm() ;
    return convert.str();
  }

};

  /*! \brief Templated Vector class = a ScalarArray with 1 column

    As for \a ScalarArray, the template parameter is the scalar type.
   */
  template<typename T> class Vector : public ScalarArray<T> {

  public:
    Vector(T* _m, int _rows):ScalarArray<T>(_m, _rows, 1){}
    Vector(int _rows):ScalarArray<T>(_rows, 1){}
    //~Vector(){}
    /** \brief this = alpha.a.x + beta.this
     */
    void gemv(char trans, T alpha, const ScalarArray<T>* a, const Vector<T>* x,
              T beta);
    /** \brief this += x
     */
    void addToMe(const Vector<T>* x);
    /** \brief this -= x
     */
    void subToMe(const Vector<T>* x);
    /** L2 norm of the vector.
     */
    int absoluteMaxIndex(int startIndex = 0) const;
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

    /** Simpler accessors for the vector data.
     */
    inline T& operator[](std::size_t i){
      return this->m[i];
    }
    inline const T& operator[] (std::size_t i) const {
      return this->m[i];
    }
  private:
    /// Disallow the copy
    Vector<T>(const Vector<T>& o);
  };

  /** \brief Wraps a ScalarArray backed by a file (using mmap()).
   */
  template<typename T> class MmapedScalarArray {
  public:
    ScalarArray<T> m;
  private:
    void* mmapedFile;
    int fd;
    size_t size;

  public:
    /** \brief Maps a .res file into memory.

        The mapping is read-only.

        \param filename The filename
        \return an instance of MMapedScalarArray<T> mapping the file.
     */
    static MmapedScalarArray<T>* fromFile(const char* filename);
    /** \brief Creates a ScalarArray backed by a file, wrapped into a MMapedScalarArray<T>.

        \param rows number of rows of the matrix.
        \param cols number of columns of the matrix.
        \param filename Filename. The file is not destroyed with the object.
     */
    MmapedScalarArray(int rows, int cols, const char* filename);
    ~MmapedScalarArray();

  private:
    MmapedScalarArray() : m(NULL, 0, 0), mmapedFile(NULL), fd(-1), size(0) {}
  };

}  // end namespace hmat

