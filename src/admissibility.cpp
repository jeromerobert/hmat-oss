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
#include "config.h"
#include "admissibility.hpp"
#include "cluster_tree.hpp"

#include "common/my_assert.h"
#include "hmat_cpp_interface.hpp"

#include <cmath>
#include <sstream>
#include <algorithm>

namespace hmat {

std::pair<bool, bool>
AdmissibilityCondition::splitRowsCols(const ClusterTree& rows, const ClusterTree& cols) const
{
  return std::pair<bool, bool>(!rows.isLeaf(), !cols.isLeaf());
}

StandardAdmissibilityCondition::StandardAdmissibilityCondition(
    double eta, double ratio, size_t maxElementsPerBlock, size_t maxElementsPerBlockRows):
    eta_(eta), ratio_(ratio), maxElementsPerBlock_(maxElementsPerBlock),
    maxElementsPerBlockAca_(maxElementsPerBlockRows)
{
    if(maxElementsPerBlockAca_ == 0) {
#ifdef HMAT_32BITS
        maxElementsPerBlockAca_ = std::numeric_limits<int>::max();
#else
      // 2^34 = 16 G elements = 256 Gbytes in Z_t = a square block of 131k x 131k
      // But this is the size of the *full* block. If the square block has rank 'r', it will store
      // two arrays of 2^17.r elements
      maxElementsPerBlockAca_ = 17179869184L;
#endif
    }
}

std::pair<bool, bool>
StandardAdmissibilityCondition::splitRowsCols(const ClusterTree& rows, const ClusterTree& cols) const
{
  if (cols.data.size() <= ratio_ * rows.data.size()  && cols.isLeaf() ) {
    // rows are two times larger than cols so we won't subdivide cols
    return std::pair<bool, bool>(!rows.isLeaf(), false);
  } else if (rows.data.size() <= ratio_ * cols.data.size()  && rows.isLeaf()) {
    // cols are two times larger than rows so we won't subdivide rows
    return std::pair<bool, bool>(false, !cols.isLeaf());
  } else if (rows.isLeaf() || cols.isLeaf()) {
    // approximately the same size, but one is a leaf so we forbid subdivision for both
    return std::pair<bool, bool>(false, false);
  } else // approximately the same size, we can subdivide both
    return std::pair<bool, bool>(true, true);
}

bool
StandardAdmissibilityCondition::stopRecursion(const ClusterTree& rows, const ClusterTree& cols) const
{
    // If we do not want to split rows nor cols, we must stop recursion (to allow Rk)
    std::pair<bool, bool> split = splitRowsCols(rows, cols);
    // If there is less than 2 rows or cols, recursion is useless
    return (rows.data.size() < 2 || cols.data.size() < 2 || (!split.first && !split.second));
}

bool
StandardAdmissibilityCondition::forceFull(const ClusterTree& rows, const ClusterTree& cols) const
{
    // If there is less than 2 rows or cols, compression is useless
    return (rows.data.size() < 2 || cols.data.size() < 2);
}

bool
StandardAdmissibilityCondition::forceRecursion(const ClusterTree& rows, const ClusterTree& cols) const
{
    if (stopRecursion(rows, cols))
        return false;
    // If the block is too large for current algorithm compression, split it
    CompressionMethod m = HMatSettings::getInstance().compressionMethod;
    bool isFullAlgo = !(m == AcaPartial || m == AcaPlus);
    size_t elements = ((size_t) rows.data.size()) * cols.data.size();
    if(isFullAlgo && elements > maxElementsPerBlock_)
        return true;

    // TODO: we may not be low rank for example if rows or cols is a large span cluster
    if(!isFullAlgo && elements > maxElementsPerBlockAca_)
        return true;

    // Otherwise, do not force split
    return false;
}

bool
StandardAdmissibilityCondition::isLowRank(const ClusterTree& rows, const ClusterTree& cols) const
{
  AxisAlignedBoundingBox* rows_bbox = static_cast<AxisAlignedBoundingBox*>(rows.admissibilityAlgoData_);
  if (rows_bbox == NULL)
  {
    rows_bbox = new AxisAlignedBoundingBox(rows.data);
    rows.admissibilityAlgoData_ = rows_bbox;
  }
  AxisAlignedBoundingBox* cols_bbox = static_cast<AxisAlignedBoundingBox*>(cols.admissibilityAlgoData_);
  if (cols_bbox == NULL)
  {
    cols_bbox = new AxisAlignedBoundingBox(cols.data);
    cols.admissibilityAlgoData_ = cols_bbox;
  }

  const double min_diameter = std::min(rows_bbox->diameter(), cols_bbox->diameter());
  return min_diameter > 0.0 && min_diameter <= eta_ * rows_bbox->distanceTo(*cols_bbox);
}

void
StandardAdmissibilityCondition::clean(const ClusterTree& current) const
{
    delete static_cast<AxisAlignedBoundingBox*>(current.admissibilityAlgoData_);
    current.admissibilityAlgoData_ = NULL;
}

std::string
StandardAdmissibilityCondition::str() const
{
  std::ostringstream oss;
  oss << "Hackbusch formula, with eta = " << eta_;
  return oss.str();
}

void StandardAdmissibilityCondition::setEta(double eta) {
    eta_ = eta;
}

double StandardAdmissibilityCondition::getEta() const {
    return eta_;
}

void StandardAdmissibilityCondition::setRatio(double ratio) {
    ratio_ = ratio;
}

StandardAdmissibilityCondition StandardAdmissibilityCondition::DEFAULT_ADMISSIBLITY = StandardAdmissibilityCondition(2.0);

AlwaysAdmissibilityCondition::AlwaysAdmissibilityCondition(size_t max_block_size, unsigned int min_block,
                                                           bool row_split, bool col_split):
    max_block_size_(max_block_size), min_nr_block_(min_block), split_rows_cols_(row_split, col_split), never_(false) {
    HMAT_ASSERT(row_split || col_split);
}

std::string AlwaysAdmissibilityCondition::str() const {
    std::ostringstream oss;
    oss << "Always admissible with max_block_size=" << max_block_size_
        << " min_nr_block=" << min_nr_block_
        << " split(rows,cols)="  << split_rows_cols_.first << "," << split_rows_cols_.second;
    return oss.str();
}

bool AlwaysAdmissibilityCondition::isLowRank(const ClusterTree&, const ClusterTree&) const {
    return !never_;
}

std::pair<bool, bool> AlwaysAdmissibilityCondition::splitRowsCols(const ClusterTree&, const ClusterTree&) const {
    return split_rows_cols_;
}

bool AlwaysAdmissibilityCondition::forceRecursion(const ClusterTree& rows, const ClusterTree& cols) const {
    size_t block_size = ((size_t)rows.data.size()) * cols.data.size();
    if(rows.father == NULL) {
        assert(cols.father == NULL);
        max_block_size_impl_ = std::min(block_size / min_nr_block_, max_block_size_);
    }
    return block_size > max_block_size_impl_;
}

bool AlwaysAdmissibilityCondition::forceFull(const ClusterTree& rows, const ClusterTree& cols) const {
    return never_ || rows.data.size() <= 2 || cols.data.size() <= 2;
}
}  // end namespace hmat
