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

    \return true  if the block is too large to perform compression
   */
  virtual bool forceRecursion(const ClusterTree& rows, const ClusterTree& cols) const {
      (void)rows, (void)cols; // unused
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
  /*! \brief Returns a pair of boolean telling if the block of interaction between 2 nodes
   is computed on this block (both values are false), or if block can be split along row- or
   col-axis.

    \return a pair of boolean.

   */
  virtual std::pair<bool, bool> splitRowsCols(const ClusterTree& rows, const ClusterTree& cols) const;

  /**
   * Return true if the block is always null,
   * (i.e. we know that is will not even be filled during the factorization).
   * @param rows the rows cluster tree
   * @param cols the cols cluster tree
   */
  bool isInert(const ClusterTree& rows, const ClusterTree& cols) {
      (void)rows, (void)cols; // unused
      return false;
  }
  /*! \brief Clean up data which may be allocated by isCompressible  */
  virtual void clean(const ClusterTree&) const {}

  virtual std::string str() const = 0;
};

/**
 * @brief Combine Hackbusch admissibility with block size
 * @param eta    a parameter used in the evaluation of the admissibility.
 * @param ratio  llows to cut tall and skinny matrices along only one direction:
      if size(rows) < ratio*size(cols), rows is not subdivided.
      if size(cols) < ratio*size(rows), cols is not subdivided.
 * @param maxElementsPerBlock limit memory size of a bloc with AcaFull and Svd compression
 * @param maxElementsPerAca limit memory size of a bloc with AcaPlus and AcaPartial compression
 */
class StandardAdmissibilityCondition : public AdmissibilityCondition
{
public:
  StandardAdmissibilityCondition(double eta, double ratio = 0, size_t maxElementsPerBlock = 20000000,
                                 size_t maxElementsPerBlockAca = 0);
  bool isLowRank(const ClusterTree& rows, const ClusterTree& cols) const;
  bool stopRecursion(const ClusterTree& rows, const ClusterTree& cols) const;
  bool forceRecursion(const ClusterTree& rows, const ClusterTree& cols) const;
  bool forceFull(const ClusterTree& rows, const ClusterTree& cols) const;
  std::pair<bool, bool> splitRowsCols(const ClusterTree& rows, const ClusterTree& cols) const;
  void clean(const ClusterTree& current) const;
  std::string str() const;
  void setEta(double eta);
  void setRatio(double ratio);
  static StandardAdmissibilityCondition DEFAULT_ADMISSIBLITY;
private:
  double eta_;
  double ratio_;
  size_t maxElementsPerBlock;
  size_t maxElementsPerBlockAca_;
};

} //  end namespace hmat
#endif
