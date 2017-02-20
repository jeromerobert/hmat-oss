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
    eta_(eta), ratio_(ratio), maxElementsPerBlock(maxElementsPerBlock),
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
  if (cols.data.size() < ratio_ * rows.data.size() ) {
    // rows are two times larger than cols so we won't subdivide cols
    return std::pair<bool, bool>(!rows.isLeaf(), false);
  } else if (rows.data.size() < ratio_ * cols.data.size() ) {
    // cols are two times larger than rows so we won't subdivide rows
    return std::pair<bool, bool>(false, !cols.isLeaf());
  } else // approximately the same size, we can subdivide both
  return std::pair<bool, bool>(!rows.isLeaf(), !cols.isLeaf());
}

bool
StandardAdmissibilityCondition::isTooSmall(const ClusterTree& rows, const ClusterTree& cols) const
{
    // If there is less than 2 rows or cols, compression is useless
    return (rows.data.size() < 2 || cols.data.size() < 2);
}

bool
StandardAdmissibilityCondition::isTooLarge(const ClusterTree& rows, const ClusterTree& cols) const
{
    // If the block is too large for current algorithm compression, split it
    CompressionMethod m = HMatSettings::getInstance().compressionMethod;
    bool isFullAlgo = !(m == AcaPartial || m == AcaPlus);
    size_t elements = ((size_t) rows.data.size()) * cols.data.size();
    if(isFullAlgo && elements > maxElementsPerBlock)
        return true;
    if(!isFullAlgo && elements > maxElementsPerBlockAca_)
        return true;

    // Otherwise, compression is performed
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

void StandardAdmissibilityCondition::setRatio(double ratio) {
    ratio_ = ratio;
}

StandardAdmissibilityCondition StandardAdmissibilityCondition::DEFAULT_ADMISSIBLITY = StandardAdmissibilityCondition(2.0);

}  // end namespace hmat
