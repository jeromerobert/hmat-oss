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

/*! \file
  \ingroup HMatrix
  \brief Spatial cluster tree implementation.
*/
#include "cluster_tree.hpp"
#include "coordinates.hpp"
#include "common/my_assert.h"
#include "common/context.hpp"

#include <algorithm>
#include <cstring>

namespace hmat {

bool IndexSet::operator==(const IndexSet& o) const {
  // Attention ! On ne fait pas de verification sur les indices, pour
  // pouvoir parler d'egalite entre des noeuds ayant ete generes
  // differemment (differentes matrices).
  return (offset_ == o.offset_) && (size_ == o.size_);
}

bool IndexSet::isSubset(const IndexSet& o) const {
  return (offset_ >= o.offset_) && (offset_ + size_ <= o.offset_ + o.size_);
}

bool IndexSet::isSuperSet(const IndexSet& o) const {
  return o.isSubset(*this);
}

bool IndexSet::intersects(const IndexSet& o) const {
  int start = std::max(offset_, o.offset_);
  int end = std::min(offset_ + size_, o.offset_ + o.size_);
  return (end > start);
}

void IndexSet::intersection(const IndexSet& s1, const IndexSet& s2) {
  int start = std::max(s1.offset(), s2.offset());
  int end = std::min(s1.offset() + s1.size(), s2.offset() + s2.size());
  int interN = end - start;
  if(interN < 0) {
      interN = 0;
      start = -1;
  }
  this->offset_ = start;
  this->size_ = interN;
}

DofData::DofData(const DofCoordinates& coordinates, int* group_index)
{
  int size = coordinates.numberOfDof();
  perm_i2e_ = new int[size];
  perm_e2i_ = new int[size];
  for (int i = 0; i < size; ++i)
  {
    perm_i2e_[i] = i;
    perm_e2i_[i] = i;
  }
  coordinates_ = new DofCoordinates(coordinates);
  if (group_index)
  {
    group_index_ = new int[size];
    memcpy(group_index_, group_index, sizeof(int) * size);
  }
  else
  {
    group_index_ = NULL;
  }
}

DofData::~DofData()
{
  delete[] perm_i2e_;
  delete[] perm_e2i_;
  delete[] group_index_;
  delete coordinates_;
}


DofData*
DofData::copy() const
{
  DofData* result = new DofData(*coordinates_, group_index_);
  size_t s = sizeof(int) * coordinates_->numberOfDof();
  memcpy(result->perm_i2e_, perm_i2e_, s);
  memcpy(result->perm_e2i_, perm_e2i_, s);
  return result;
}

void ClusterData::moveDoF(int index, ClusterData* right)
{
  HMAT_ASSERT(offset_ + size_ == right->offset_ );
  HMAT_ASSERT(index >= offset_);
  HMAT_ASSERT(index < offset_ + size_);
  // Swap degree of freedom and the last element of this bucket
  std::swap(indices()[index], indices()[offset_ + size_ - 1]);
  // Make current index set smaller ...
  size_ -= 1;
  // ... and expand right sibling
  right->offset_ -=  1;
  right->size_ += 1;
}

ClusterTree::ClusterTree(const DofData* dofData)
  : Tree<ClusterTree>(NULL)
  , data(dofData)
  , clusteringAlgoData_(NULL)
  , admissibilityAlgoData_(NULL)
{
}

ClusterTree::ClusterTree(const ClusterTree& other)
  : Tree<ClusterTree>(NULL)
  , data(other.data)
  , clusteringAlgoData_(NULL)
  , admissibilityAlgoData_(NULL)
{
}

ClusterTree::~ClusterTree() {
  if(father == NULL)
  {
    delete data.dofData_;
  }
}

ClusterTree*
ClusterTree::slice(int offset, int size) const
{
  assert(offset >= 0);
  assert(size >= 0);
  ClusterTree* result = new ClusterTree(*this);
  result->data.offset_ = offset;
  result->data.size_ = size;
  return result;
}

ClusterTree* ClusterTree::copy(const ClusterTree* copyFather) const {
  ClusterTree* result = NULL;
  if (!copyFather) {
    // La racine doit s'occuper le tableau des points et le mapping.
    result = new ClusterTree(data.dofData_->copy());
    copyFather = result;
  } else {
    result = copyFather->slice(data.offset(), data.size());
  }
  if (!isLeaf()) {
    for (int i=0 ; i<nrChild(); i++)
      result->insertChild(i, ((ClusterTree*) getChild(i))->copy(copyFather));
  }
  return result;
}

AxisAlignedBoundingBox::AxisAlignedBoundingBox(const ClusterData& data)
    : dimension_(data.coordinates()->dimension())
    , bb_(new double[2 * dimension_])
{
    if (data.size() == 0)
        return;
    int* myIndices = data.indices() + data.offset();
    const DofCoordinates& coords = *data.coordinates();
    for (unsigned i = 0; i < dimension_; i++) {
        bb_[i] = coords.spanPoint(myIndices[0], 0, i);
        bb_[i + dimension_] = bb_[i];
    }

    for (unsigned i = 1; i < data.size(); ++i) {
        int dof = myIndices[i];
        for (unsigned pointId = 0; pointId < coords.spanSize(dof); pointId++)
            coords.spanAABB(dof, bb_);
    }
}

AxisAlignedBoundingBox::~AxisAlignedBoundingBox() {
    delete[] bb_;
}

double
AxisAlignedBoundingBox::diameter() const
{
  double result = 0.0;
  for(int i = 0; i < dimension_; ++i)
  {
    double delta = bbMin()[i] - bbMax()[i];
    result += delta * delta;
  }

  return sqrt(result);
}

double AxisAlignedBoundingBox::extends(int dim) const {
    return bbMax()[dim] - bbMin()[dim];
}

int AxisAlignedBoundingBox::greatestDim() const {
    double dmax = 0;
    int imax = 0;
    for(int i = 0; i < dimension_; ++i) {
        double delta = bbMax()[i] - bbMin()[i];
        if(delta > dmax) {
            dmax = delta;
            imax = i;
        }
    }
    return imax;
}

double
AxisAlignedBoundingBox::distanceTo(const AxisAlignedBoundingBox& other) const
{
  double result = 0.;
  double difference = 0.;

  for(int i = 0; i < dimension_; ++i)
  {
    difference = std::max(0., bbMin()[i] - other.bbMax()[i]);
    result += difference * difference;
    difference = std::max(0., other.bbMin()[i] - bbMax()[i]);
    result += difference * difference;
  }

  return sqrt(result);
}

}  // end namespace hmat
