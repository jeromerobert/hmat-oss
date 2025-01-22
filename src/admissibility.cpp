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
#include <sstream>
#include <algorithm>

namespace hmat {

bool
AdmissibilityCondition::forceRecursion(const ClusterTree& rows, const ClusterTree& cols, size_t) const
{
    return rows.data.size() > maxWidth_ || cols.data.size() > maxWidth_;
}

std::pair<bool, bool>
AdmissibilityCondition::splitRowsCols(const ClusterTree& rows, const ClusterTree& cols) const
{
  if (cols.data.size() < ratio_ * rows.data.size() ) {
    // rows are much larger than cols so we won't subdivide cols
    return std::pair<bool, bool>(!rows.isLeaf(), false);
  } else if (rows.data.size() < ratio_ * cols.data.size() ) {
    // cols are much larger than rows so we won't subdivide rows
    return std::pair<bool, bool>(false, !cols.isLeaf());
  } else // approximately the same size, we can subdivide both
    return std::pair<bool, bool>(!rows.isLeaf(), !cols.isLeaf());
}

const AxisAlignedBoundingBox*
AdmissibilityCondition::getAxisAlignedBoundingBox(const ClusterTree& current, bool) const
{
  return static_cast<const AxisAlignedBoundingBox*>(current.cache_);
}

StandardAdmissibilityCondition::StandardAdmissibilityCondition(double eta, double ratio):
    eta_(eta) { ratio_ = ratio; }

namespace {
void
recursive_compute_bounding_box(const ClusterTree& current)
{
  if (current.cache_)
    return;
  current.cache_ = new AxisAlignedBoundingBox(current.data);
  for (int i = 0; i < current.nrChild(); ++i)
  {
    if (current.getChild(i))
      recursive_compute_bounding_box(*((ClusterTree*)current.getChild(i)));
  }
}

void
recursive_delete_bounding_box(const ClusterTree& current)
{
  delete static_cast<AxisAlignedBoundingBox*>(current.cache_);
  current.cache_ = NULL;
  for (int i = 0; i < current.nrChild(); ++i)
  {
    if (current.getChild(i))
      recursive_delete_bounding_box(*((ClusterTree*)current.getChild(i)));
  }
}
}

void
StandardAdmissibilityCondition::prepare(const ClusterTree& rows, const ClusterTree& cols) const
{
  recursive_compute_bounding_box(rows);
  if (&rows != &cols)
    recursive_compute_bounding_box(cols);
}

bool
StandardAdmissibilityCondition::stopRecursion(const ClusterTree& rows, const ClusterTree& cols) const
{
    // If there is less than 2 rows or cols, recursion is useless
    return (rows.data.size() < 2 || cols.data.size() < 2);
}

bool
StandardAdmissibilityCondition::forceFull(const ClusterTree& rows, const ClusterTree& cols) const
{
    // If there is less than 2 rows or cols, compression is useless
    return (rows.data.size() < 2 || cols.data.size() < 2);
}

bool
StandardAdmissibilityCondition::isLowRank(const ClusterTree& rows, const ClusterTree& cols) const
{
  const AxisAlignedBoundingBox* rows_bbox = getAxisAlignedBoundingBox(rows, true);
  const AxisAlignedBoundingBox* cols_bbox = getAxisAlignedBoundingBox(cols, false);

  const double min_diameter = std::min(rows_bbox->diameter(), cols_bbox->diameter());
  return min_diameter > 0.0 && min_diameter <= eta_ * rows_bbox->distanceTo(*cols_bbox);
}

void
StandardAdmissibilityCondition::clean(const ClusterTree& rows, const ClusterTree& cols) const
{
    recursive_delete_bounding_box(rows);
    if (&rows != &cols)
        recursive_delete_bounding_box(cols);
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

struct DefaultBlockSizeDetector: public AlwaysAdmissibilityCondition::BlockSizeDetector {
  static DefaultBlockSizeDetector& instance()
  {
     static DefaultBlockSizeDetector INSTANCE;
     return INSTANCE;
  }
  void compute(size_t & max_block_size, unsigned int & min_nr_block, bool never) {
    if(max_block_size == 0)
      max_block_size = 1 << 20;
    if(min_nr_block == 0)
      min_nr_block = 1;
  }
};

AlwaysAdmissibilityCondition::BlockSizeDetector * AlwaysAdmissibilityCondition::blockSizeDetector_=
    &DefaultBlockSizeDetector::instance();

AlwaysAdmissibilityCondition::AlwaysAdmissibilityCondition(size_t max_block_size, unsigned int min_block,
                                                           bool row_split, bool col_split):
    max_block_size_(max_block_size), min_nr_block_(min_block),
    split_rows_cols_(row_split, col_split), never_(false) {
    HMAT_ASSERT(row_split || col_split);
    ratio_ = sqrt(0.5);
    blockSizeDetector_->compute(max_block_size_, min_nr_block_, never_);
}

std::string AlwaysAdmissibilityCondition::str() const {
    std::ostringstream oss;
    oss << "Always admissible with max_block_size=" << max_block_size_
        << " min_nr_block=" << min_nr_block_
        << " split(rows,cols)="  << split_rows_cols_.first << "," << split_rows_cols_.second;
    return oss.str();
}

void AlwaysAdmissibilityCondition::never(bool n) {
  never_ = n;
  blockSizeDetector_->compute(max_block_size_, min_nr_block_, never_);
}

bool AlwaysAdmissibilityCondition::isLowRank(const ClusterTree&, const ClusterTree&) const {
    return !never_;
}

std::pair<bool, bool> AlwaysAdmissibilityCondition::splitRowsCols(const ClusterTree& r, const ClusterTree& c) const {
    std::pair<bool, bool> s = split_rows_cols_;
    s.first = s.first && !r.isLeaf();
    s.second = s.second && !c.isLeaf();
    if(s.first && s.second) {
        // Try to do square blocks
        s = AdmissibilityCondition::splitRowsCols(r, c);
    }
    if(!s.first && !s.second) {
        // The split_rows_cols_ parameter cannot be honored because we reach a leaf.
        // So we split the only cluster which can be splitted and ignore split_rows_cols_
        if(r.isLeaf()) {
            assert(!c.isLeaf());
            s.second = true;
        } else {
            s.first = true;
        }
    }
    return s;
}

bool AlwaysAdmissibilityCondition::stopRecursion(const ClusterTree& rows, const ClusterTree& cols) const {
    size_t block_size = ((size_t)rows.data.size()) * cols.data.size();
    if(rows.father == NULL && cols.father == NULL) {
        max_block_size_impl_ = std::min(block_size / min_nr_block_, max_block_size_);
    }
    if (never_ && block_size <= max_block_size_impl_)
        return true;
    return AdmissibilityCondition::stopRecursion(rows, cols);
}

bool AlwaysAdmissibilityCondition::forceRecursion(const ClusterTree& rows, const ClusterTree& cols, size_t elem_size) const {
    size_t block_size = ((size_t)rows.data.size()) * cols.data.size();
    if(rows.father == NULL && cols.father == NULL) {
        max_block_size_impl_ = std::min(block_size / min_nr_block_, max_block_size_);
    }
    if (block_size > max_block_size_impl_)
        return true;
    return AdmissibilityCondition::forceRecursion(rows, cols, elem_size);
}

bool AlwaysAdmissibilityCondition::forceFull(const ClusterTree& rows, const ClusterTree& cols) const {
    return never_ || rows.data.size() <= 2 || cols.data.size() <= 2;
}

std::string HODLRAdmissibilityCondition::str() const {
  return "HODLRAdmissibilityCondition";
}

bool HODLRAdmissibilityCondition::isLowRank(const ClusterTree& row, const ClusterTree& col) const {
  return !(row.data == col.data);
}

HODLRAdmissibilityCondition* HODLRAdmissibilityCondition::clone() const {
  return new HODLRAdmissibilityCondition();
}
}  // end namespace hmat
