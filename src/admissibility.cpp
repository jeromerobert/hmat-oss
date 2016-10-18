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
AdmissibilityCondition::isRowsColsAdmissible(const ClusterTree& rows, const ClusterTree& cols)
{
  bool admissible = (isAdmissible(rows, cols) );
  return std::pair<bool, bool>(admissible, admissible);
}

bool
AdmissibilityCondition::isCompressible(const ClusterTree& rows, const ClusterTree& cols)
{
  return isAdmissible(rows, cols);
}

std::pair<bool, bool>
TallSkinnyAdmissibilityCondition::isRowsColsAdmissible(const ClusterTree& rows, const ClusterTree& cols)
{
  if (rows.data.size() >= ratio * cols.data.size() ) {
    // rows are two times larger than cols so we won't subdivide rows
    return std::pair<bool, bool>(false, true);
  } else if (cols.data.size() >= ratio * rows.data.size() ) {
    // cols are two times larger than rows so we won't subdivide cols
    return std::pair<bool, bool>(true, false);
  } else // approximately the same size and non leaf
  return std::pair<bool, bool>(false, false);
}

StandardAdmissibilityCondition::StandardAdmissibilityCondition(
    double eta, size_t maxElementsPerBlock, size_t maxElementsPerBlockRows):
    eta_(eta), maxElementsPerBlock(maxElementsPerBlock),
    maxElementsPerBlockAca_(maxElementsPerBlockRows), always_(false)
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

bool
StandardAdmissibilityCondition::isAdmissible(const ClusterTree& rows, const ClusterTree& cols)
{
    CompressionMethod m = HMatSettings::getInstance().compressionMethod;
    bool isFullAlgo = !(m == AcaPartial || m == AcaPlus);
    size_t elements = ((size_t) rows.data.size()) * cols.data.size();

    if(always_ && (rows.isLeaf() || cols.isLeaf()))
        return true;
    if(isFullAlgo && elements > maxElementsPerBlock)
        return false;
    if(!isFullAlgo && elements > maxElementsPerBlockAca_)
        return false;

    // a one element cluster would have a 0 diameter so we stop
    // before it happen.
    if(rows.data.size() < 2 || cols.data.size() < 2)
        return false;

    if(always_)
        return true;

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

    return std::min(rows_bbox->diameter(), cols_bbox->diameter()) <=
        eta_ * rows_bbox->distanceTo(*cols_bbox);
}

std::pair<bool, bool>
StandardAdmissibilityCondition::isRowsColsAdmissible(const ClusterTree& rows, const ClusterTree& cols)
{
  // We mix the 2 admissibility conditions, one on the ratio number of rows / number of cols coming from
  // TallSkinnyAdmissibilityCondition and one on the ratio distance/diameter adapted to BEM matrices.
  // The new criteria is that to be row- or col-admissible, you need to be admissible for 1 of the 2 conditions.
  // Therefore, you subdivide only if both criteria for subdivision are satisfied (i.e. the matrix block is not too large
  // in the considered dimension AND the 2 cluster trees are too close w.r.t. their sizes).
  std::pair<bool, bool> tall_skinny_admissible = TallSkinnyAdmissibilityCondition::isRowsColsAdmissible(rows, cols);
  bool standard_admissible = isAdmissible(rows, cols);
  tall_skinny_admissible.first |= standard_admissible;
  tall_skinny_admissible.second |= standard_admissible;
  // If I want to subdivide in both direction but one is a leaf, then I subdivide in neither direction
  //  if ( !tall_skinny_admissible.first && !tall_skinny_admissible.second && (rows.isLeaf() || cols.isLeaf()))
  //    tall_skinny_admissible.first = tall_skinny_admissible.second = true;
  return tall_skinny_admissible;
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

void StandardAdmissibilityCondition::setAlways(bool b) {
    always_ = b;
}

StandardAdmissibilityCondition StandardAdmissibilityCondition::DEFAULT_ADMISSIBLITY = StandardAdmissibilityCondition(2.0);

}  // end namespace hmat
