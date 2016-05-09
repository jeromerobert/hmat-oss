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
#include "hmat_cpp_interface.hpp"

#include <algorithm>
#include <cstring>

namespace {

/*! \brief Comparateur pour deux indices selon une de leur coordonnee.
 */
class IndicesComparator
{
private:
  const double* coordinates_;
  const int* group_index_;
  const int dimension_;
  const int axis_;

public:
  IndicesComparator(int axis, const hmat::ClusterData& data)
    : coordinates_(&data.coordinates()->get(0,0))
    , group_index_(data.group_index())
    , dimension_(data.coordinates()->dimension())
    , axis_(axis)
  {}
  bool operator() (int i, int j) {
    if (group_index_ == NULL || group_index_[i] == group_index_[j])
      return coordinates_[i * dimension_ + axis_] < coordinates_[j * dimension_ + axis_];
    return group_index_[i] < group_index_[j];
  }
};

void sortByDimension(hmat::ClusterTree& node, int dim)
{
  int* myIndices = node.data.indices() + node.data.offset();
  std::stable_sort(myIndices, myIndices + node.data.size(), IndicesComparator(dim, node.data));
}

hmat::AxisAlignedBoundingBox*
getAxisAlignedBoundingbox(const hmat::ClusterTree& node)
{
  hmat::AxisAlignedBoundingBox* bbox = static_cast<hmat::AxisAlignedBoundingBox*>(node.clusteringAlgoData_);
  if (bbox == NULL)
  {
    bbox = new hmat::AxisAlignedBoundingBox(node.data);
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
  const int dimension = node.data.coordinates()->dimension();
  for (int i = 0; i < dimension; i++) {
    double size = (bbox->bbMax[i] - bbox->bbMin[i]);
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
  const int dimension = node.data.coordinates()->dimension();
  for (int dim = 0; dim < dimension; dim++) {
    result *= (bbox->bbMax[dim] - bbox->bbMin[dim]);
  }
  return result;
}

}  // End of anonymous namespace


namespace hmat {

void
ClusteringAlgorithm::setMaxLeafSize(int maxLeafSize)
{
  maxLeafSize_ = maxLeafSize;
}

int
ClusteringAlgorithm::getMaxLeafSize() const
{
  if (maxLeafSize_ >= 0)
    return maxLeafSize_;
  const HMatSettings& settings = HMatSettings::getInstance();
  return settings.maxLeafSize;
}

void
GeometricBisectionAlgorithm::partition(ClusterTree& current, std::vector<ClusterTree*>& children) const
{
  int dim;
  if (spatialDimension_ < 0) {
    spatialDimension_ = current.data.coordinates()->dimension();
  }
  if (axisIndex_ < 0) {
    dim = largestDimension(current);
  } else {
    dim = ((axisIndex_ + current.depth) % spatialDimension_);
  }
  sortByDimension(current, dim);
  AxisAlignedBoundingBox* bbox = new AxisAlignedBoundingBox(current.data);
  current.clusteringAlgoData_ = bbox;

  double middle = .5 * (bbox->bbMin[dim] + bbox->bbMax[dim]);
  int middleIndex = 0;
  int* myIndices = current.data.indices() + current.data.offset();
  const double* coord = &current.data.coordinates()->get(0,0);
  while (coord[myIndices[middleIndex]*spatialDimension_+dim] < middle) {
    middleIndex++;
  }
  if (NULL != current.data.group_index())
  {
    // Ensure that we do not split inside a group
    const int* group_index = current.data.group_index() + current.data.offset();
    const int group(group_index[middleIndex]);
    if (group_index[middleIndex-1] == group)
    {
      int upper = middleIndex;
      int lower = middleIndex-1;
      while (upper < current.data.size() && group_index[upper] == group)
        ++upper;
      while (lower >= 0 && group_index[lower] == group)
        --lower;
      if (lower < 0 && upper == current.data.size())
      {
        // All degrees of freedom belong to the same group, this is fine
      }
      else if (lower < 0)
        middleIndex = upper;
      else if (upper == current.data.size())
        middleIndex = lower + 1;
      else if (coord[myIndices[upper]*spatialDimension_+dim] + coord[myIndices[lower]*spatialDimension_+dim] < 2.0 * middle)
        middleIndex = upper;
      else
        middleIndex = lower + 1;
    }
  }
  children.push_back(current.slice(current.data.offset(), middleIndex));
  children.push_back(current.slice(current.data.offset()+ middleIndex, current.data.size() - middleIndex));
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
      spatialDimension_ = current.data.coordinates()->dimension();
    dim = ((axisIndex_ + current.depth) % spatialDimension_);
  }
  sortByDimension(current, dim);
  int previousIndex = 0;
  // Loop on 'divider_' = the number of children created
  for (int i=1 ; i<divider_ ; i++) {
    int middleIndex = current.data.size() * i / divider_;
  if (NULL != current.data.group_index())
  {
    // Ensure that we do not split inside a group
    const int* group_index = current.data.group_index() + current.data.offset();
    const int group(group_index[middleIndex]);
    if (group_index[middleIndex-1] == group)
    {
      int upper = middleIndex;
      int lower = middleIndex-1;
      while (upper < current.data.size() && group_index[upper] == group)
        ++upper;
      while (lower >= 0 && group_index[lower] == group)
        --lower;
      if (lower < 0 && upper == current.data.size())
      {
        // All degrees of freedom belong to the same group, this is fine
      }
      else if (lower < 0)
        middleIndex = upper;
      else if (upper == current.data.size())
        middleIndex = lower + 1;
      else if (upper + lower < 2 * middleIndex)
        middleIndex = upper;
      else
        middleIndex = lower + 1;
    }
  }
    children.push_back(current.slice(current.data.offset()+previousIndex, middleIndex-previousIndex));
    previousIndex = middleIndex;
  }
  // Add the last child :
  children.push_back(current.slice(current.data.offset()+ previousIndex, current.data.size() - previousIndex));
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
  double leftVolume    = volume(*children[0]);
  double rightVolume   = volume(*children[1]);
  double maxVolume     = std::max(rightVolume, leftVolume);
  if (maxVolume > thresholdRatio_*currentVolume)
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

void
VoidClusteringAlgorithm::partition(ClusterTree& current, std::vector<ClusterTree*>& children) const
{
  if (current.depth % 2 == 0)
  {
    algo_->partition(current, children);
  } else {
    children.push_back(current.slice(current.data.offset(), current.data.size()));
    children.push_back(current.slice(current.data.offset() + current.data.size(), 0));
  }
}

void
VoidClusteringAlgorithm::clean(ClusterTree& current) const
{
  algo_->clean(current);
}

ClusterTreeBuilder::ClusterTreeBuilder(const ClusteringAlgorithm& algo)
{
  algo_.push_front(std::pair<int, ClusteringAlgorithm*>(0, algo.clone()));
}

ClusterTreeBuilder::~ClusterTreeBuilder()
{
  for (std::list<std::pair<int, ClusteringAlgorithm*> >::iterator it = algo_.begin(); it != algo_.end(); ++it)
  {
    delete it->second;
    it->second = NULL;
  }
}

ClusterTree*
ClusterTreeBuilder::build(const DofCoordinates& coordinates, int* group_index) const
{
  DofData* dofData = new DofData(coordinates, group_index);
  ClusterTree* rootNode = new ClusterTree(dofData);

  divide_recursive(*rootNode);
  clean_recursive(*rootNode);
  // Update reverse mapping
  int* indices_i2e = rootNode->data.indices();
  int* indices_e2i = rootNode->data.indices_rev();

  for (int i = 0; i < rootNode->data.size(); ++i) {
    indices_e2i[indices_i2e[i]] = i;
  }
  return rootNode;
}

void
ClusterTreeBuilder::clean_recursive(ClusterTree& current) const
{
  ClusteringAlgorithm* algo = getAlgorithm(current.depth);
  algo->clean(current);
  if (!current.isLeaf())
  {
    for (int i = 0; i < current.nrChild(); ++i)
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
  ClusteringAlgorithm* algo = getAlgorithm(current.depth);
  if (current.data.size() <= algo->getMaxLeafSize())
    return;

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

