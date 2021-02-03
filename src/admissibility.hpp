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
class AxisAlignedBoundingBox;

class AdmissibilityCondition
{
public:
  AdmissibilityCondition() : maxWidth_((size_t)-1L) {}
  /*! \brief Virtual copy constructor */
  virtual AdmissibilityCondition * clone() const = 0;

  virtual ~AdmissibilityCondition() {}

  /*! \brief Precompute ClusterTree::cache_ */
  virtual void prepare(const ClusterTree& rows, const ClusterTree& cols) const {}
  /*! \brief Clean up data which may be allocated by prepare  */
  virtual void clean(const ClusterTree& rows, const ClusterTree& cols) const {}

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
  virtual bool forceRecursion(const ClusterTree& rows, const ClusterTree& cols, size_t elemSize) const;
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
  virtual bool isInert(const ClusterTree& rows, const ClusterTree& cols) const {
      (void)rows, (void)cols; // unused
      return false;
  }

  /*! \brief Get approximate rank of a block cluster */
  virtual int getApproximateRank(const ClusterTree& rows, const ClusterTree& cols) const {
      (void)rows, (void)cols; // unused
      return 25;
  }

  /**
   * @brief Get axis aligned bounding box of a cluster tree
   * @param current cluster tree
   * @param is_rows current is a rows (resp. cols) cluster when is_rows
        is true (resp. false)
   */
  virtual const AxisAlignedBoundingBox* getAxisAlignedBoundingBox(const ClusterTree& current, bool is_rows) const;

  virtual std::string str() const = 0;

  /**
   * @brief Set ratio to cut tall and skinny matrices
   * @param ratio  allows to cut tall and skinny matrices along only one direction:
      if size(rows) < ratio*size(cols), rows is not subdivided.
      if size(cols) < ratio*size(rows), cols is not subdivided.
 */
  void setRatio(double ratio) { ratio_ = ratio; }

  /**
   * @brief Set maximum width (default is unlimited)
   * @param maxWidth  force recursion if size(rows) or size(cols) is larger than this threshold.
 */
  void setMaxWidth(size_t maxWidth) { maxWidth_ = maxWidth; }

protected:
  double ratio_;
  size_t maxWidth_;
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
  StandardAdmissibilityCondition * clone() const { return new StandardAdmissibilityCondition(*this); }
  // Precompute axis aligned bounding blocks
  void prepare(const ClusterTree& rows, const ClusterTree& cols) const;
  void clean(const ClusterTree& rows, const ClusterTree& cols) const;
  // Returns true if block is admissible (Hackbusch condition)
  bool isLowRank(const ClusterTree& rows, const ClusterTree& cols) const;
  // Returns true when there is less than 2 rows or cols
  bool stopRecursion(const ClusterTree& rows, const ClusterTree& cols) const;
  // Returns true when there is less than 2 rows or cols
  bool forceFull(const ClusterTree& rows, const ClusterTree& cols) const;
  std::string str() const;
  void setEta(double eta);
  double getEta() const;
protected:
  double eta_;
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
    AlwaysAdmissibilityCondition * clone() const { return new AlwaysAdmissibilityCondition(*this); }
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

/**
 * @brief Class which can be used as a base class to override only some methods.
 * @param admissibility All methods which are not redefined are delegated to this instance.
 */
class ProxyAdmissibilityCondition : public AdmissibilityCondition
{
public:
  explicit ProxyAdmissibilityCondition(AdmissibilityCondition * admissibility) : proxy_(admissibility ? admissibility->clone() : NULL) {}
  ProxyAdmissibilityCondition * clone() const { return new ProxyAdmissibilityCondition(*this); }
  ~ProxyAdmissibilityCondition() { delete proxy_; }
  AdmissibilityCondition * getProxy() const { return proxy_; }
  void setProxy(AdmissibilityCondition * admissibility) { proxy_ = admissibility; }

  void prepare(const ClusterTree& rows, const ClusterTree& cols) const {
    proxy_->prepare(rows, cols);
  }
  void clean(const ClusterTree& rows, const ClusterTree& cols) const {
    return proxy_->clean(rows, cols);
  }
  bool isLowRank(const ClusterTree& rows, const ClusterTree& cols) const {
    return proxy_->isLowRank(rows, cols);
  }
  bool stopRecursion(const ClusterTree& rows, const ClusterTree& cols) const {
    return proxy_->stopRecursion(rows, cols);
  }
  bool forceRecursion(const ClusterTree& rows, const ClusterTree& cols, size_t elemSize) const {
    return proxy_->forceRecursion(rows, cols, elemSize);
  }
  bool forceFull(const ClusterTree& rows, const ClusterTree& cols) const {
    return proxy_->forceFull(rows, cols);
  }
  bool forceRk(const ClusterTree& rows, const ClusterTree& cols) const {
    return proxy_->forceRk(rows, cols);
  }
  std::pair<bool, bool> splitRowsCols(const ClusterTree& rows, const ClusterTree& cols) const {
    return proxy_->splitRowsCols(rows, cols);
  }
  bool isInert(const ClusterTree& rows, const ClusterTree& cols) const {
    return proxy_->isInert(rows, cols);
  }
  int getApproximateRank(const ClusterTree& rows, const ClusterTree& cols) const {
    return proxy_->getApproximateRank(rows, cols);
  }
  const AxisAlignedBoundingBox* getAxisAlignedBoundingBox(const ClusterTree& current, bool is_rows) const {
    return proxy_->getAxisAlignedBoundingBox(current, is_rows);
  }
  std::string str() const {
    return proxy_->str();
  }

private:
  AdmissibilityCondition * proxy_;
};

} //  end namespace hmat
#endif
