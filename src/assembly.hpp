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

namespace hmat {

class ClusterData;
class ClusterTree;
struct LocalSettings;
template<typename T> class FullMatrix;
template<typename T> class Vector;
template<typename T> class RkMatrix;
template<typename T> class Function;
template<typename T> class BlockFunction;
template<typename T> class SimpleFunction;


/** Allow to be notified when the prepareBlock method need to allocate memory */
class AllocationObserver {
public:
    virtual void allocate(size_t) const {}
    virtual void free(size_t) const {}
};

/**
 * Abstract class, describing the creation of the H-matrix blocks
 */
template<typename T> class Assembly {
public:
    /**
     * @brief assemble Assemble a block of the matrix
     * This function must set fullMatrix or rkMatrix following this rules:
     * - Setting both fullMatrix and rkMatrix is an error
     * - Setting none mean the block is empty and not compressed
     */
    virtual void assemble(const LocalSettings & settings,
                          const ClusterTree & rows, const ClusterTree & cols,
                          bool admissible,
                          FullMatrix<T> * & fullMatrix, RkMatrix<T> * & rkMatrix,
                          double epsilon, const AllocationObserver & = AllocationObserver()) = 0;
    virtual ~Assembly(){};
};

/**
 * An assembling which use a Function object to compute blocks
 */
template<typename T, template <typename> class F> class AssemblyFunction: public Assembly<T> {
public:
    AssemblyFunction(const F<T> function): function_(function) {}
    virtual void assemble(const LocalSettings & settings,
                          const ClusterTree & rows, const ClusterTree & cols,
                          bool admissible,
                          FullMatrix<T> * & fullMatrix, RkMatrix<T> * & rkMatrix,
                          double epsilon, const AllocationObserver & = AllocationObserver());
protected:
    const F<T> function_;
};

/** Abstract base class representing an assembly function.
 */
template<typename T> class Function {
public:
  virtual ~Function() {}
  virtual FullMatrix<typename Types<T>::dp>* assemble(const ClusterData* rows,
                                                      const ClusterData* cols,
                                                      const hmat_block_info_t * block_info = NULL,
                                                      const AllocationObserver & = AllocationObserver()) const = 0;
  /*! \brief Prepare the Assembly function to optimize getRow() and getCol().

    In some cases, it is more efficient to tell the client code that a
    given block is going to be assembled. The data needed to track
    this are stored in \a handle.

    \param rows the rows of the block
    \param cols the columns of the block
    \param handle The handle that is storing the associated data.
  */
  virtual void prepareBlock(const ClusterData* rows, const ClusterData* cols,
                            hmat_block_info_t * block_info, const AllocationObserver &) const;
  /*! \brief Release a block prepared with \a AssemblyFunction::prepareBlock().

    \param handle the handle passed to \a AssemblyFunction::prepareBlock().
  */
  virtual void releaseBlock(hmat_block_info_t *, const AllocationObserver &) const {}

  /*! \brief Return a row of a matrix block.

    This functions returns a \a Vector representing the row of index
    \a rowIndex in the subblock defined by its \a rows and \a cols.

    \param rows the rows of the subblock
    \param cols the columns of the subblock
    \param rowIndex the row index in the subblock
    \param handle the optional handle created by \a AssemblyFunction::prepareBlock()
    \param result the computed row
    \param stratum the stratum id or -1 for all strata
    \return the row as a \a Vector
  */
  virtual void getRow(const ClusterData* rows, const ClusterData* cols,
                      int rowIndex, void* handle,
                      Vector<typename Types<T>::dp>* result, int stratum) const = 0;

  /*! \brief Return a column of a matrix block.

    This functions returns a \a Vector representing the column of
    index \a colIndex in the subblock defined by its \a rows and \a
    cols.

    \param rows the rows of the subblock
    \param cols the columns of the subblock
    \param colIndex the row index in the subblock
    \param handle the optional handle created by \a AssemblyFunction::prepareBlock()
    \param result the computed column
    \param stratum the stratum id or -1 for all strata
    \return the column as a \a Vector
  */
  virtual void getCol(const ClusterData* rows, const ClusterData* cols,
                      int colIndex, void* handle,
                      Vector<typename Types<T>::dp>* result, int stratum) const = 0;

    /*! \brief Return an element of a matrix block.

      This functions returns the value element representing the element in the
      column of index \a colIndex and in the row of index \a rowIndex,
      in the subblock defined by its \a rows and \a cols.

      \param rows the rows of the subblock
      \param cols the columns of the subblock
      \param rowIndex the row index in the subblock
      \param colIndex the col index in the subblock
      \param handle the optional handle created by \a AssemblyFunction::prepareBlock()
      \param stratum the stratum id or -1 for all strata
      \return the value of the element as a \a T
    */
    virtual T getElement(const ClusterData* rows, const ClusterData* cols,
                        int rowIndex, int colIndex, void* handle, int stratum) const = 0;
};

/**
 * \brief Initialiaze a hmat_bloc_info_t structure.
 * Should be used in any prepareBlock implementations
 */
void initBlockInfo(hmat_block_info_t * info);

/** C++ wrapper for hmat_interaction_func_t */
template<typename T> class SimpleFunction : public Function<T> {
  hmat_interaction_func_t compute_;
  void * userContext_;
public:
  SimpleFunction(hmat_interaction_func_t compute, void * userContext):
      compute_(compute), userContext_(userContext) {}
  virtual ~SimpleFunction() {}
  virtual FullMatrix<typename Types<T>::dp>* assemble(const ClusterData* rows,
                                                      const ClusterData* cols,
                                                      const hmat_block_info_t * block_info = NULL,
                                                      const AllocationObserver & = AllocationObserver()) const;
  virtual void getRow(const ClusterData* rows, const ClusterData* cols,
                      int rowIndex, void* handle,
                      Vector<typename Types<T>::dp>* result, int stratum) const;
  virtual void getCol(const ClusterData* rows, const ClusterData* cols,
                      int colIndex, void* handle,
                      Vector<typename Types<T>::dp>* result, int stratum) const;

  virtual T getElement(const ClusterData* rows, const ClusterData* cols,
                      int rowIndex, int colIndex, void* handle, int stratum) const;

};

template<typename T> class BlockFunction : public Function<T> {
  hmat_prepare_func_t prepare;
  void (*compute_)(struct hmat_block_compute_context_t*);
  hmat_compute_func_t legacyCompute_;
  void* matrixUserData_;
  int* rowMapping;
  int* rowReverseMapping;
  int* colMapping;
  int* colReverseMapping;
  void prepareImpl(const ClusterData* rows, const ClusterData* cols,
                   hmat_block_info_t * block_info) const;
public:
  BlockFunction(const ClusterData* _rowData, const ClusterData* _colData,
                void* matrixUserData_, hmat_prepare_func_t _prepare,
                hmat_compute_func_t legacyCompute,
                void (*compute)(struct hmat_block_compute_context_t*));
  ~BlockFunction();
  FullMatrix<typename Types<T>::dp>* assemble(const ClusterData* rows,
                                              const ClusterData* cols,
                                              const hmat_block_info_t * block_info,
                                              const AllocationObserver & = AllocationObserver()) const;
  virtual void prepareBlock(const ClusterData* rows, const ClusterData* cols,
                            hmat_block_info_t * block_info, const AllocationObserver &) const;
  virtual void releaseBlock(hmat_block_info_t * block_info, const AllocationObserver &) const;

  virtual void getRow(const ClusterData* rows, const ClusterData* cols, int rowIndex,
                      void* handle, Vector<typename Types<T>::dp>* result, int stratum) const;

  virtual void getCol(const ClusterData* rows, const ClusterData* cols, int colIndex,
                      void* handle, Vector<typename Types<T>::dp>* result, int stratum) const;

  virtual T getElement(const ClusterData* rows, const ClusterData* cols,
                       int rowIndex, int colIndex, void* handle, int stratum) const;
};

}  // end namespace hmat

#endif
