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

#ifndef _INTERACTION_HPP
#define _INTERACTION_HPP
#include <vector>
#include "data_types.hpp"
#include "hmat/hmat.h"

class ClusterData;
class ClusterTree;
template<typename T> class FullMatrix;
template<typename T> class Vector;

/** Abstract base class representing an assembly function.
 */
template<typename T> class AssemblyFunction {
public:
  /** Return the element (i, j) of the matrix.

      This function has to ignore any mapping.
   */
  virtual typename Types<T>::dp interaction(int i, int j) const = 0;
  virtual ~AssemblyFunction() {};
  virtual FullMatrix<typename Types<T>::dp>* assemble(const ClusterData* rows,
                                                      const ClusterData* cols,
                                                      void *handle=NULL,
                                                      hmat_block_info_t * block_info=NULL) const = 0;
  /*! \brief Prepare the Assembly function to optimize getRow() and getCol().

    In some cases, it is more efficient to tell the client code that a
    given block is going to be assembled. The data needed to track
    this are stored in \a handle.

    \param rows the rows of the block
    \param cols the columns of the block
    \param handle The handle that is storing the associated data.
  */
  virtual void prepareBlock(const ClusterData* rows, const ClusterData* cols,
                            void** handle, hmat_block_info_t * block_info) const {}
  /*! \brief Release a block prepared with \a AssemblyFunction::releaseBlock().

    \param handle the handle passed to \a AssemblyFunction::releaseBlock().
  */
  virtual void releaseBlock(void* handle, hmat_block_info_t *) const {}
  /*! \brief Return a row of a matrix block.

    This functions returns a \a Vector representing the row of index
    \a rowIndex in the subblock defined by its \a rows and \a cols.

    \param rows the rows of the subblock
    \param cols the columns of the subblock
    \param rowIndex the row index in the subblock
    \param handle the optional handle created by \a AssemblyFunction::prepareBlock()
    \return the row as a \a Vector
  */
  virtual Vector<typename Types<T>::dp>* getRow(const ClusterData* rows,
                                                const ClusterData* cols,
                                                int rowIndex, void* handle=NULL) const = 0;
  /*! \brief Return a row of a matrix block.

    This functions returns a \a Vector representing the row of index
    \a rowIndex in the subblock defined by its \a rows and \a cols.

    \param rows the rows of the subblock
    \param cols the columns of the subblock
    \param rowIndex the row index in the subblock
    \param handle the optional handle created by \a AssemblyFunction::prepareBlock()
    \param result the computed row
    \return the row as a \a Vector
  */
  virtual void getRow(const ClusterData* rows, const ClusterData* cols,
                      int rowIndex, void* handle,
                      Vector<typename Types<T>::dp>* result) const = 0;
  /*! \brief Return a column of a matrix block.

    This functions returns a \a Vector representing the column of
    index \a rowIndex in the subblock defined by its \a rows and \a
    cols.

    \param rows the rows of the subblock
    \param cols the columns of the subblock
    \param colIndex the row index in the subblock
    \param handle the optional handle created by \a AssemblyFunction::prepareBlock()
    \return the column as a \a Vector
  */
  virtual Vector<typename Types<T>::dp>* getCol(const ClusterData* rows,
                                                const ClusterData* cols,
                                                int colIndex, void* handle=NULL) const = 0;
  /*! \brief Return a column of a matrix block.

    This functions returns a \a Vector representing the column of
    index \a rowIndex in the subblock defined by its \a rows and \a
    cols.

    \param rows the rows of the subblock
    \param cols the columns of the subblock
    \param colIndex the row index in the subblock
    \param handle the optional handle created by \a AssemblyFunction::prepareBlock()
    \param result the computed column
    \return the column as a \a Vector
  */
  virtual void getCol(const ClusterData* rows, const ClusterData* cols,
                      int colIndex, void* handle,
                      Vector<typename Types<T>::dp>* result) const = 0;
};

/** Simple \a AssemblyFunction that allows to only redefine \a AssemblyFunction::interaction().

    The rest of the function work by executing a trivial loop on
    \a SimpleAssemblyFunction<T>::interaction().
 */
template<typename T> class SimpleAssemblyFunction : public AssemblyFunction<T> {
public:
  virtual typename Types<T>::dp interaction(int i, int j) const = 0;
  virtual ~SimpleAssemblyFunction() {};
  virtual FullMatrix<typename Types<T>::dp>* assemble(const ClusterData* rows,
                                                      const ClusterData* cols,
                                                      void *handle=NULL,
                                                      hmat_block_info_t * block_info=NULL) const;
  virtual Vector<typename Types<T>::dp>* getRow(const ClusterData* rows,
                                                const ClusterData* cols,
                                                int rowIndex, void* handle=NULL) const;
  virtual void getRow(const ClusterData* rows, const ClusterData* cols,
                      int rowIndex, void* handle,
                      Vector<typename Types<T>::dp>* result) const;
  virtual Vector<typename Types<T>::dp>* getCol(const ClusterData* rows,
                                                const ClusterData* cols,
                                                int colIndex, void* handle=NULL) const;
  virtual void getCol(const ClusterData* rows, const ClusterData* cols,
                      int colIndex, void* handle,
                      Vector<typename Types<T>::dp>* result) const;
};


template<typename T> class BlockAssemblyFunction : public AssemblyFunction<T> {
private:
  prepare_func prepare;
  compute_func compute;
  release_func free_data;
  void* user_context;
  int* rowMapping;
  int* rowReverseMapping;
  int* colMapping;
  int* colReverseMapping;

public:
  BlockAssemblyFunction(const ClusterData* _rowData, const ClusterData* _colData,
                         void* _user_context,
                         prepare_func _prepare, compute_func _compute,
                         release_func _free_data);
  ~BlockAssemblyFunction();
  typename Types<T>::dp interaction(int i, int j) const;
  FullMatrix<typename Types<T>::dp>* assemble(const ClusterData* rows,
                                              const ClusterData* cols,
                                              void *handle=NULL,
                                              hmat_block_info_t * block_info=NULL) const;
  virtual void prepareBlock(const ClusterData* rows, const ClusterData* cols,
                            void** handle, hmat_block_info_t * block_info) const;
  virtual void releaseBlock(void* handle, hmat_block_info_t * block_info) const;
  virtual Vector<typename Types<T>::dp>* getRow(const ClusterData* rows,
                                                const ClusterData* cols,
                                                int rowIndex, void* handle=NULL) const;
  virtual void getRow(const ClusterData* rows, const ClusterData* cols,
                      int rowIndex, void* handle, Vector<typename Types<T>::dp>* result) const;
  virtual Vector<typename Types<T>::dp>* getCol(const ClusterData* rows,
                                                const ClusterData* cols,
                                                int colIndex, void* handle=NULL) const;
  virtual void getCol(const ClusterData* rows, const ClusterData* cols,
                      int colIndex, void* handle,
                      Vector<typename Types<T>::dp>* result) const;
};
#endif
