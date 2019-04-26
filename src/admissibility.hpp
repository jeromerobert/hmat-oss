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
  \brief Spatial cluster tree for the Dofs.
*/
#ifndef _ADMISSIBLITY_HPP
#define _ADMISSIBLITY_HPP

#include <cstddef>
#include <string>

namespace hmat {

// Forward declarations
class ClusterTree;

class AdmissibilityCondition
{
public:
  virtual ~AdmissibilityCondition() {}
  /*! \brief Returns true if the block of interaction between 2 nodes has a
      low-rank representation.

    \return true  if the block should be Rk.
   */
  virtual bool isLowRank(const ClusterTree& rows, const ClusterTree& cols) const = 0;
  /*! \brief Returns a boolean telling if the block of interaction between 2 nodes
      is too small to recurse.
      Note: stopRecursion and forceRecursion must not both return true.

    \return true  if the block is too small to recurse
   */
  virtual bool stopRecursion(const ClusterTree& rows, const ClusterTree& cols) const {
      (void)rows, (void)cols; // unused
      return false;
  }
  /*! \brief Returns a boolean telling if the block of interaction between 2 nodes
      is too large to perform compression, if it is low rank; in this case, recursion
      is performed.
      Note: stopRecursion and forceRecursion must not both return true.
      \param elemSize the size of the element of the matrix (i.e. sizeof(T))
    \return true  if the block is too large to perform compression
   */
  virtual bool forceRecursion(const ClusterTree& rows, const ClusterTree& cols, size_t elemSize) const {
      (void)rows, (void)cols, (void)elemSize; // unused
      return false;
  }
  /*! \brief Returns a boolean telling if the block of interaction between 2 nodes
      must not be compressed, even if admissible.
      Note: forceFull and forceRk must not both return true.

    \return true  if the admissible block must not be compressed
   */
  virtual bool forceFull(const ClusterTree& rows, const ClusterTree& cols) const {
      (void)rows, (void)cols; // unused
      return false;
  }
  /*! \brief Returns a boolean telling if the block of interaction between 2 nodes
      must be compressed, even if not admissible.
      Note: forceFull and forceRk must not both return true.

    \return true  if the block must be compressed
   */
  virtual bool forceRk(const ClusterTree& rows, const ClusterTree& cols) const {
      (void)rows, (void)cols; // unused
      return false;
  }
  /*! \brief Tell whether a block must be splitted along rows, cols or both.
      Note: This method must not return {false, false}
    \return a pair of boolean.

   */
  virtual std::pair<bool, bool> splitRowsCols(const ClusterTree& rows, const ClusterTree& cols) const;

  /**
   * Return true if the block is always null,
   * (i.e. we know that is will not even be filled during the factorization).
   * @param rows the rows cluster tree
   * @param cols the cols cluster tree
   */
  virtual bool isInert(const ClusterTree& rows, const ClusterTree& cols) {
      (void)rows, (void)cols; // unused
      return false;
  }

  /*! \brief Get approximate rank of a block cluster */
  virtual int getApproximateRank(const ClusterTree& rows, const ClusterTree& cols) const {
      (void)rows, (void)cols; // unused
      return 25;
  }

  /*! \brief Clean up data which may be allocated by isLowRank  */
  virtual void clean(const ClusterTree&) const {}

  virtual std::string str() const = 0;
};

/**
 * @brief Combine Hackbusch admissibility with block size
 * @param eta    a parameter used in the evaluation of the admissibility.
 * @param ratio  allows to cut tall and skinny matrices along only one direction:
      if size(rows) < ratio*size(cols), rows is not subdivided.
      if size(cols) < ratio*size(rows), cols is not subdivided.
 */
class StandardAdmissibilityCondition : public AdmissibilityCondition
{
public:
  StandardAdmissibilityCondition(double eta, double ratio = 0);
  // Returns true if block is admissible (Hackbusch condition)
  bool isLowRank(const ClusterTree& rows, const ClusterTree& cols) const;
  // Returns true when there is less than 2 rows or cols
  bool stopRecursion(const ClusterTree& rows, const ClusterTree& cols) const;
  // Returns true when there is less than 2 rows or cols
  bool forceFull(const ClusterTree& rows, const ClusterTree& cols) const;
  std::pair<bool, bool> splitRowsCols(const ClusterTree& rows, const ClusterTree& cols) const;
  void clean(const ClusterTree& current) const;
  std::string str() const;
  void setEta(double eta);
  double getEta() const;
  void setRatio(double ratio);
  static StandardAdmissibilityCondition DEFAULT_ADMISSIBLITY;
protected:
  double eta_;
  double ratio_;
};

class AlwaysAdmissibilityCondition : public AdmissibilityCondition {
public:
    struct BlockSizeDetector {
      virtual void compute(size_t & max_block_size, unsigned int & min_nr_block,
                           bool never)=0;
      virtual ~BlockSizeDetector(){};
    };
private:
    size_t max_block_size_;
    unsigned int min_nr_block_;
    std::pair<bool, bool> split_rows_cols_;
    mutable size_t max_block_size_impl_;
    bool never_;
    static BlockSizeDetector * blockSizeDetector_;
public:
    /**
     * @brief AlwaysAdmissibilityCondition
     * Create an admissibility condiction which set all blocks as admissible
     * @param max_block_size The maximum acceptable block size in number of values (rows * cols)
     * @param min_nr_block The minimum acceptable number of blocks created with this condition
     * @param split_rows Tel whether or not to split rows (see splitRowsCols)
     * @param split_cols Tel whether or not to split cols (see splitRowsCols)
     */
    AlwaysAdmissibilityCondition(size_t max_block_size, unsigned int min_nr_block,
                                 bool split_rows = true, bool split_cols = false);
    std::string str() const;
    bool isLowRank(const ClusterTree&, const ClusterTree&) const;
    std::pair<bool, bool> splitRowsCols(const ClusterTree& rows, const ClusterTree&) const;
    bool forceRecursion(const ClusterTree& rows, const ClusterTree& cols, size_t elemSize) const;
    bool stopRecursion(const ClusterTree& rows, const ClusterTree& cols) const;
    bool forceFull(const ClusterTree& rows, const ClusterTree& cols) const;
    /** @Brief Let this admissibility condition always create full blocks */
    void never(bool n);
    static void setBlockSizeDetector(BlockSizeDetector * b) { blockSizeDetector_ = b; }
};
} //  end namespace hmat
#endif
