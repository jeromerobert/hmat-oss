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
  const int size(coordinates.size());
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
  memcpy(result->perm_i2e_, perm_i2e_, sizeof(int) * coordinates_->size());
  memcpy(result->perm_e2i_, perm_e2i_, sizeof(int) * coordinates_->size());
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

void ClusterTree::sameDepth(const ClusterTree * other) {
    if(other->isLeaf() || data.size() == 0)
        return;

    if(isLeaf()) {
        insertChild(0, new ClusterTree(*this));
        insertChild(1, slice(data.offset() + data.size(), 0));
    }

    for(int i = 0; i < other->nrChild(); i++) {
        for(int j = 0; j < nrChild(); j++) {
            getChild(j)->sameDepth(other->getChild(i));
        }
    }
}

AxisAlignedBoundingBox::AxisAlignedBoundingBox(const ClusterData& data)
  : dimension_(data.coordinates()->dimension())
  , bbMin(new double[dimension_])
  , bbMax(new double[dimension_])
{
  if (data.size() == 0) return;
  int* myIndices = data.indices() + data.offset();
  const double* coord = &data.coordinates()->get(0, 0);
  memcpy(&bbMin[0], &coord[dimension_*myIndices[0]], sizeof(double) * dimension_);
  memcpy(&bbMax[0], &coord[dimension_*myIndices[0]], sizeof(double) * dimension_);

  for (int i = 0; i < data.size(); ++i) {
    int index = myIndices[i];
    const double* p = &coord[dimension_*index];
    for (int dim = 0; dim < dimension_; ++dim) {
      bbMin[dim] = std::min(bbMin[dim], p[dim]);
      bbMax[dim] = std::max(bbMax[dim], p[dim]);
    }
  }
}

AxisAlignedBoundingBox::AxisAlignedBoundingBox(int dim, const double *bboxMin, const double *bboxMax)
  : dimension_(dim)
  , bbMin(new double[dimension_])
  , bbMax(new double[dimension_])
{
  memcpy(&bbMin[0], bboxMin, sizeof(double) * dimension_);
  memcpy(&bbMax[0], bboxMax, sizeof(double) * dimension_);
}

AxisAlignedBoundingBox::~AxisAlignedBoundingBox()
{
  delete [] bbMin;
  bbMin = NULL;
  delete [] bbMax;
  bbMax = NULL;
}

double
AxisAlignedBoundingBox::diameter() const
{
  double result = 0.0;
  for(int i = 0; i < dimension_; ++i)
  {
    double delta = bbMin[i] - bbMax[i];
    result += delta * delta;
  }

  return sqrt(result);
}

double
AxisAlignedBoundingBox::distanceTo(const AxisAlignedBoundingBox& other) const
{
  double result = 0.;
  double difference = 0.;

  for(int i = 0; i < dimension_; ++i)
  {
    difference = std::max(0., bbMin[i] - other.bbMax[i]);
    result += difference * difference;
    difference = std::max(0., other.bbMin[i] - bbMax[i]);
    result += difference * difference;
  }

  return sqrt(result);
}

}  // end namespace hmat
