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

#include "clustering.hpp"
#include "cluster_tree.hpp"
#include "common/my_assert.h"

#include <algorithm>
#include <cstring>

namespace {

/*! \brief Comparateur pour deux indices selon une de leur coordonnee.

  On utilise les templates pour eviter un trop grand cout a
  l'execution ou le copier-coller, avec pour le parametre du template :
    - 0 pour une comparaison selon x
    - 1 pour une comparaison selon y
    - 2 pour une comparaison selon z
 */
template<int N>
class IndicesComparator
{
private:
  const double* coordinates_;
  const int* group_index_;
  const int dimension_;

public:
  IndicesComparator(const hmat::ClusterData& data)
    : coordinates_(&data.coordinates()->get(0,0))
    , group_index_(data.group_index())
    , dimension_(data.coordinates()->dimension())
  {}
  bool operator() (int i, int j) {
    if (group_index_ == NULL || group_index_[i] == group_index_[j])
      return coordinates_[i * dimension_ + N] < coordinates_[j * dimension_ + N];
    return group_index_[i] < group_index_[j];
  }
};

void sortByDimension(hmat::ClusterTree& node, int dim)
{
  int* myIndices = node.data_.indices() + node.data_.offset();
  switch (dim) {
  case 0:
    std::stable_sort(myIndices, myIndices + node.data_.size(), IndicesComparator<0>(node.data_));
    break;
  case 1:
    std::stable_sort(myIndices, myIndices + node.data_.size(), IndicesComparator<1>(node.data_));
    break;
  case 2:
    std::stable_sort(myIndices, myIndices + node.data_.size(), IndicesComparator<2>(node.data_));
    break;
  default:
    strongAssert(false);
  }
}

hmat::AxisAlignedBoundingBox*
getAxisAlignedBoundingbox(const hmat::ClusterTree& node)
{
  hmat::AxisAlignedBoundingBox* bbox = static_cast<hmat::AxisAlignedBoundingBox*>(node.clusteringAlgoData_);
  if (bbox == NULL)
  {
    bbox = new hmat::AxisAlignedBoundingBox(node);
    node.clusteringAlgoData_ = bbox;
  }
  return bbox;
}

int
largestDimension(const hmat::ClusterTree& node)
{
  int maxDim = -1;
  double maxSize = -1.0;
  hmat::AxisAlignedBoundingBox* bbox = getAxisAlignedBoundingbox(node);
  for (int i = 0; i < 3; i++) {
    double size = (bbox->bbMax.xyz[i] - bbox->bbMin.xyz[i]);
    if (size > maxSize) {
      maxSize = size;
      maxDim = i;
    }
  }
  return maxDim;
}

double
volume(const hmat::ClusterTree& node)
{
  hmat::AxisAlignedBoundingBox* bbox = getAxisAlignedBoundingbox(node);
  double result = 1.;
  for (int dim = 0; dim < 3; dim++) {
    result *= (bbox->bbMax.xyz[dim] - bbox->bbMin.xyz[dim]);
  }
  return result;
}

}  // End of anonymous namespace


namespace hmat {

void
GeometricBisectionAlgorithm::partition(ClusterTree& current, std::vector<ClusterTree*>& children) const
{
  int dim;
  if (axisIndex_ < 0) {
    dim = largestDimension(current);
  } else {
    if (spatialDimension_ < 0)
      spatialDimension_ = current.data_.coordinates()->dimension();
    dim = ((axisIndex_ + current.depth) % spatialDimension_);
  }
  sortByDimension(current, dim);
  AxisAlignedBoundingBox* bbox = new AxisAlignedBoundingBox(current);
  current.clusteringAlgoData_ = bbox;

  double middle = .5 * (bbox->bbMin.xyz[dim] + bbox->bbMax.xyz[dim]);
  int middleIndex = 0;
  int* myIndices = current.data_.indices() + current.data_.offset();
  const double* coord = &current.data_.coordinates()->get(0,0);
  while (coord[myIndices[middleIndex]*3+dim] < middle) {
    middleIndex++;
  }
  if (NULL != current.data_.group_index())
  {
    // Ensure that we do not split inside a group
    const int* group_index = current.data_.group_index() + current.data_.offset();
    const int group(group_index[middleIndex]);
    if (group_index[middleIndex-1] == group)
    {
      int upper = middleIndex;
      int lower = middleIndex-1;
      while (upper < current.data_.size() && group_index[upper] == group)
        ++upper;
      while (lower > 0 && group_index[lower] == group)
        --lower;
      if (lower == 0 || (coord[myIndices[upper]*3+dim] + coord[myIndices[lower]*3+dim] < 2.0 * middle))
        middleIndex = upper;
      else
        middleIndex = lower;
    }
  }
  children.push_back(current.slice(current.data_.offset(), middleIndex));
  children.push_back(current.slice(current.data_.offset()+ middleIndex, current.data_.size() - middleIndex));
}

void
GeometricBisectionAlgorithm::clean(ClusterTree& current) const
{
  AxisAlignedBoundingBox* bbox = static_cast<AxisAlignedBoundingBox*>(current.clusteringAlgoData_);
  delete bbox;
  current.clusteringAlgoData_ = NULL;
}

void
MedianBisectionAlgorithm::partition(ClusterTree& current, std::vector<ClusterTree*>& children) const
{
  int dim;
  if (axisIndex_ < 0) {
    dim = largestDimension(current);
  } else {
    if (spatialDimension_ < 0)
      spatialDimension_ = current.data_.coordinates()->dimension();
    dim = ((axisIndex_ + current.depth) % spatialDimension_);
  }
  sortByDimension(current, dim);
  int middleIndex = current.data_.size() / 2;
  if (NULL != current.data_.group_index())
  {
    // Ensure that we do not split inside a group
    const int* group_index = current.data_.group_index() + current.data_.offset();
    const int group(group_index[middleIndex]);
    if (group_index[middleIndex-1] == group)
    {
      int upper = middleIndex;
      int lower = middleIndex-1;
      while (upper < current.data_.size() && group_index[upper] == group)
        ++upper;
      while (lower > 0 && group_index[lower] == group)
        --lower;
      if (lower == 0 || (upper + lower < 2 * middleIndex))
        middleIndex = upper;
      else
        middleIndex = lower;
    }
  }
  children.push_back(current.slice(current.data_.offset(), middleIndex));
  children.push_back(current.slice(current.data_.offset()+ middleIndex, current.data_.size() - middleIndex));
}

void
MedianBisectionAlgorithm::clean(ClusterTree& current) const
{
  AxisAlignedBoundingBox* bbox = static_cast<AxisAlignedBoundingBox*>(current.clusteringAlgoData_);
  delete bbox;
  current.clusteringAlgoData_ = NULL;
}

void
HybridBisectionAlgorithm::partition(ClusterTree& current, std::vector<ClusterTree*>& children) const
{
  // We first split tree node with an MedianBisectionAlgorithm instance, and compute
  // ratios of volume of children node divided by volume of current node.  If any ratio
  // is larger than a given threshold, this splitting is discarded and replaced by
  // a GeometricBisectionAlgorithm instead.
  medianAlgorithm_.partition(current, children);
  if (children.size() != 2)
    return;
  double currentVolume = volume(current);
  double leftVolume = volume(*children[0]);
  double rightVolume = volume(*children[1]);
  double maxRatio = std::max(rightVolume / currentVolume, leftVolume / currentVolume);
  if (maxRatio > thresholdRatio_)
  {
    children.clear();
    geometricAlgorithm_.partition(current, children);
  }
}

void
HybridBisectionAlgorithm::clean(ClusterTree& current) const
{
  medianAlgorithm_.clean(current);
  geometricAlgorithm_.clean(current);
}

ClusterTreeBuilder::ClusterTreeBuilder(const ClusteringAlgorithm& algo, int maxLeafSize)
  : maxLeafSize_(maxLeafSize)
{
  algo_.push_front(std::pair<int, ClusteringAlgorithm*>(0, algo.clone()));
}

ClusterTree*
ClusterTreeBuilder::build(const DofCoordinates& coordinates, int* group_index)
{
  DofData* dofData = new DofData(coordinates, group_index);
  ClusterTree* rootNode = new ClusterTree(dofData);

  divide_recursive(*rootNode);
  clean_recursive(*rootNode);
  return rootNode;
}

void
ClusterTreeBuilder::clean_recursive(ClusterTree& current) const
{
  ClusteringAlgorithm* algo = getAlgorithm(current.depth);
  algo->clean(current);
  if (!current.isLeaf())
  {
    for (size_t i = 0; i < 2; ++i)
    {
      if (current.getChild(i))
        clean_recursive(*((ClusterTree*)current.getChild(i)));
    }
  }
}

ClusteringAlgorithm*
ClusterTreeBuilder::getAlgorithm(int depth) const
{
  ClusteringAlgorithm* last = NULL;
  for (std::list<std::pair<int, ClusteringAlgorithm*> >::const_iterator it = algo_.begin(); it != algo_.end(); ++it)
  {
    if (it->first <= depth)
      last = it->second;
    else
      break;
  }
  return last;
}

ClusterTreeBuilder&
ClusterTreeBuilder::addAlgorithm(int depth, const ClusteringAlgorithm& algo)
{
  for (std::list<std::pair<int, ClusteringAlgorithm*> >::iterator it = algo_.begin(); it != algo_.end(); ++it)
  {
    if (it->first > depth)
    {
      algo_.insert(it, std::pair<int, ClusteringAlgorithm*>(depth, algo.clone()));
      return * this;
    }
  }
  algo_.insert(algo_.end(), std::pair<int, ClusteringAlgorithm*>(depth, algo.clone()));
  return *this;
}

void
ClusterTreeBuilder::divide_recursive(ClusterTree& current) const
{
  if (current.data_.size() <= maxLeafSize_)
    return;
  ClusteringAlgorithm* algo = getAlgorithm(current.depth);

  // Sort degrees of freedom and partition current node
  std::vector<ClusterTree*> children;
  algo->partition(current, children);
  for (size_t i = 0; i < children.size(); ++i)
  {
    current.insertChild(i, children[i]);
    divide_recursive(*children[i]);
  }
}

}  // end namespace hmat

