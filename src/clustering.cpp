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
#include <unordered_map>

namespace {

/*! \brief Compare two DOF indices based on their coordinates
 */
class IndicesComparator
{
private:
  const hmat::DofCoordinates * coordinates_;
  const int* group_index_;
  const int dimension_;
  const int axis_;
  std::unordered_map<int, double> group_axis_coordinate_;

public:
  IndicesComparator(int axis, const hmat::ClusterData& data)
    : coordinates_(data.coordinates())
    , group_index_(data.group_index())
    , dimension_(data.coordinates()->dimension())
    , axis_(axis)
  {
     if (group_index_) {
       std::unordered_map<int, int> group_counter;
       for (int i = 0; i < coordinates_->size(); ++i) {
         group_counter[group_index_[i]]++;
         group_axis_coordinate_[group_index_[i]] += coordinates_->spanCenter(i, axis_);
       }
       for (int i = 0; i < group_axis_coordinate_.size(); ++i) {
         group_axis_coordinate_[i] /= group_counter[i];
       }
     }
  }
  bool operator() (int i, int j) {
    if (group_index_ == NULL || group_index_[i] == group_index_[j])
      return coordinates_->spanCenter(i, axis_) < coordinates_->spanCenter(j, axis_);
    return group_index_[i] < group_index_[j];
  }
};

/** @brief Compare two DOF based on their "large span" status */
class LargeSpanComparator {
    const hmat::DofCoordinates& coordinates_;
    double threshold_;
    int dimension_;
public:
	LargeSpanComparator(const hmat::DofCoordinates& coordinates,
        double threshold, int dimension)
        : coordinates_(coordinates), threshold_(threshold), dimension_(dimension){}
    bool operator()(int i, int j) {
        bool vi = coordinates_.spanDiameter(i, dimension_) > threshold_;
        bool vj = coordinates_.spanDiameter(j, dimension_) > threshold_;
        return vi < vj;
    }
};

}

namespace hmat {

void
AxisAlignClusteringAlgorithm::sortByDimension(ClusterTree& node, int dim)
const
{
  IndicesComparator comparator(dim, node.data);
  int* myIndices = node.data.indices() + node.data.offset();
  std::stable_sort(myIndices, myIndices + node.data.size(), comparator);
}

AxisAlignedBoundingBox*
AxisAlignClusteringAlgorithm::getAxisAlignedBoundingbox(const ClusterTree& node)
const
{
    hmat::AxisAlignedBoundingBox* bbox = static_cast<hmat::AxisAlignedBoundingBox*>(node.cache_);
    if (bbox == NULL) {
        bbox = new hmat::AxisAlignedBoundingBox(node.data);
        node.cache_ = bbox;
    }
    return bbox;
}

void
AxisAlignClusteringAlgorithm::clean(ClusterTree& current) const
{
  delete static_cast<AxisAlignedBoundingBox*>(current.cache_);
  current.cache_ = NULL;
}

int
AxisAlignClusteringAlgorithm::largestDimension(const ClusterTree& node, int toAvoid, double avoidRatio)
const
{
  AxisAlignedBoundingBox* bbox = getAxisAlignedBoundingbox(node);
  const int dimension = node.data.coordinates()->dimension();
  std::vector< std::pair<double, int> > sizeDim(dimension);
  for (int i = 0; i < dimension; i++) {
    sizeDim[i].second = i;
    sizeDim[i].first = bbox->bbMax()[i] - bbox->bbMin()[i];
  }
  std::sort(sizeDim.data(), sizeDim.data() + dimension);
  if(toAvoid < 0 || dimension < 2 || sizeDim[dimension - 1].second != toAvoid ||
     sizeDim[dimension - 1].first > avoidRatio * sizeDim[dimension - 2].first)
    return sizeDim[dimension - 1].second;
  else
    return sizeDim[dimension - 2].second;
}

double
AxisAlignClusteringAlgorithm::volume(const ClusterTree& node)
const
{
  AxisAlignedBoundingBox* bbox = getAxisAlignedBoundingbox(node);
  double result = 1.;
  const int dimension = node.data.coordinates()->dimension();
  for (int dim = 0; dim < dimension; dim++) {
    result *= (bbox->bbMax()[dim] - bbox->bbMin()[dim]);
  }
  return result;
}

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

int
ClusteringAlgorithm::getDivider() const
{
  return divider_;
}

void
ClusteringAlgorithm::setDivider(int divider) const
{
  divider_ = divider;
}

int
GeometricBisectionAlgorithm::partition(ClusterTree& current, std::vector<ClusterTree*>& children,
                                       int currentAxis) const
{
  bool x0 = x0_ && current.depth == 0;
  int dim = x0 ? 0 : largestDimension(current, currentAxis);
  sortByDimension(current, dim);
  AxisAlignedBoundingBox* bbox = getAxisAlignedBoundingbox(current);
  current.cache_ = bbox;

  int previousIndex = 0;
  // Loop on 'divider_' = the number of children created
  for (int i=1 ; i<divider_ ; i++) {
    int middleIndex = previousIndex;
    double middlePosition;
    if(x0) {
      middlePosition = 0;
    } else {
      middlePosition = bbox->bbMin()[dim] + (i / (double)divider_) *
        (bbox->bbMax()[dim] - bbox->bbMin()[dim]);
    }
    int* myIndices = current.data.indices() + current.data.offset();
    const DofCoordinates & coord = *current.data.coordinates();
    while (middleIndex < current.data.size() &&
      coord.spanCenter(myIndices[middleIndex], dim) < middlePosition) {
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
        else if (coord.spanCenter(myIndices[upper], dim) + coord.spanCenter(myIndices[lower], dim) < 2 * middlePosition)
          middleIndex = upper;
        else
          middleIndex = lower + 1;
      }
    }
    if (middleIndex > previousIndex)
      children.push_back(current.slice(current.data.offset()+previousIndex, middleIndex-previousIndex));
    previousIndex = middleIndex;
  }
  // Add the last child
  children.push_back(current.slice(current.data.offset()+ previousIndex, current.data.size() - previousIndex));
  return dim;
}

int
MedianBisectionAlgorithm::partition(ClusterTree& current, std::vector<ClusterTree*>& children,
                                    int currentAxis) const
{
  int dim = largestDimension(current, currentAxis);
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
    if (middleIndex > previousIndex)
      children.push_back(current.slice(current.data.offset()+previousIndex, middleIndex-previousIndex));
    previousIndex = middleIndex;
  }
  // Add the last child
  children.push_back(current.slice(current.data.offset()+ previousIndex, current.data.size() - previousIndex));
  return dim;
}

int
HybridBisectionAlgorithm::partition(ClusterTree& current, std::vector<ClusterTree*>& children,
                                    int currentAxis) const
{
  // We first split tree node with an MedianBisectionAlgorithm instance, and compute
  // ratios of volume of children node divided by volume of current node.  If any ratio
  // is larger than a given threshold, this splitting is discarded and replaced by
  // a GeometricBisectionAlgorithm instead.
  int dim = medianAlgorithm_.partition(current, children, currentAxis);
  if (children.size() < 2)
    return dim;
  double currentVolume = volume(current);
  double maxVolume = 0.0;
  for (std::vector<ClusterTree*>::const_iterator cit = children.begin(); cit != children.end(); ++cit)
  {
    if (*cit != NULL)
      maxVolume = std::max(maxVolume, volume(**cit));
  }
  if (maxVolume > thresholdRatio_*currentVolume)
  {
    children.clear();
    dim = geometricAlgorithm_.partition(current, children, currentAxis);
  }
  return dim;
}

void
HybridBisectionAlgorithm::clean(ClusterTree& current) const
{
  medianAlgorithm_.clean(current);
  geometricAlgorithm_.clean(current);
}

int
VoidClusteringAlgorithm::partition(ClusterTree& current, std::vector<ClusterTree*>& children,
                                   int currentAxis) const
{
  if (current.depth % 2 == 0)
  {
    return algo_->partition(current, children, currentAxis);
  } else {
    children.push_back(current.slice(current.data.offset(), current.data.size()));
    for (int i=1 ; i<divider_ ; i++)
      children.push_back(current.slice(current.data.offset() + current.data.size(), 0));
    return -1; // here partition axis is meaning less
  }
}

void
VoidClusteringAlgorithm::clean(ClusterTree& current) const
{
  algo_->clean(current);
}

int
ShuffleClusteringAlgorithm::partition(ClusterTree& current, std::vector<ClusterTree*>& children,
                                      int currentAxis) const
{
  int dim = algo_->partition(current, children, currentAxis);
  ++divider_;
  if (divider_ > toDivider_)
      divider_ = fromDivider_;
  setDivider(divider_);
  return dim;
}

void
ShuffleClusteringAlgorithm::clean(ClusterTree& current) const
{
  algo_->clean(current);
}

int
NTilesRecursiveAlgorithm::subpartition( ClusterTree& father, ClusterTree *current, std::vector<ClusterTree*>& children, int currentAxis ) const
{
    int loffset, lsize, roffset, rsize;
    int offset = current->data.offset();
    int size   = current->data.size();
    int ntiles = ( size + tileSize_ - 1 ) / tileSize_;
    assert( ntiles > 0 );

    if( ntiles == 1 ) {
	/* Register the leaf as a direct child */
	children.push_back( father.slice( offset, size ) );
	return currentAxis;
    }

    /* Sort the subset */
    int dim = largestDimension( *current, currentAxis );
    sortByDimension( *current, dim );

    lsize = tileSize_ * (( ntiles + 1 ) / 2 );
    
    loffset = current->data.offset();
    roffset = loffset + lsize;
    rsize   = size - lsize;
    assert( rsize > 0 );
    
    ClusterTree *slice;
    /* Left */
    slice = current->slice( loffset, lsize );
    subpartition( father, slice, children, dim );
    // avoid dofData_ deletion
    slice->father = slice;
    clean( *slice );
    delete slice;

    /* Right */
    slice = current->slice( roffset, rsize );
    subpartition( father, slice, children, dim );
    // avoid dofData_ deletion
    slice->father = slice;
    clean( *slice );
    delete slice;

    return dim;
}

/* The goal is to create a binary tree with all leaves as children of the root in the final state */
int
NTilesRecursiveAlgorithm::partition(ClusterTree& current, std::vector<ClusterTree*>& children, int currentAxis) const
{
    ClusterTree *slice = current.slice( current.data.offset(), current.data.size() );
    int dim = subpartition( current, slice, children, currentAxis );

    /* Change the father to avoid deleting the data */
    slice->father = slice;
    clean( *slice );
    delete( slice );
    return dim;
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

  divide_recursive(*rootNode, -1);
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
ClusterTreeBuilder::divide_recursive(ClusterTree& current, int currentAxis) const
{
  ClusteringAlgorithm* algo = getAlgorithm(current.depth);
  if (current.data.size() <= algo->getMaxLeafSize())
    return;

  // Sort degrees of freedom and partition current node
  std::vector<ClusterTree*> children;
  int childrenAxis = algo->partition(current, children, currentAxis);
  for (size_t i = 0; i < children.size(); ++i)
  {
    current.insertChild(i, children[i]);
    divide_recursive(*children[i], childrenAxis);
  }
}

SpanClusteringAlgorithm::SpanClusteringAlgorithm(
    const ClusteringAlgorithm &algo, double ratio):
    algo_(algo), ratio_(ratio){
    setMaxLeafSize(algo_.getMaxLeafSize());
}

std::string SpanClusteringAlgorithm::str() const {
    return "SpanClusteringAlgorithm";
}
ClusteringAlgorithm* SpanClusteringAlgorithm::clone() const {
    return new SpanClusteringAlgorithm(algo_, ratio_);
}

int SpanClusteringAlgorithm::partition(
    ClusterTree& current, std::vector<ClusterTree*>& children, int currentAxis) const {
    int offset = current.data.offset();
    int* indices = current.data.indices() + offset;
    const DofCoordinates & coords = *current.data.coordinates();
    int n = current.data.size();
    assert(n + offset <= current.data.coordinates()->numberOfDof());
    AxisAlignedBoundingBox * aabb = getAxisAlignedBoundingbox(current);
    int greatestDim = aabb->greatestDim();
    double threshold = aabb->extends(greatestDim) * ratio_;
    // move large span at the end of the indices array
    LargeSpanComparator comparator(coords, threshold, greatestDim);
    std::stable_sort(indices, indices + n, comparator);
    // create the large span cluster
    int i = n - 1;
    while(i >= 0 && coords.spanDiameter(indices[i], greatestDim) > threshold)
        i--;
    ClusterTree * largeSpanCluster = i < n - 1 ? current.slice(offset + i + 1, n - i - 1) : NULL;
    // Call the delegate algorithm with a temporary cluster
    // containing only small span DOFs.
    ClusterTree * smallSpanCluster = i >= 0 ? current.slice(offset, i + 1) : NULL;
    int dim = -1;
    if(smallSpanCluster != NULL) {
        dim = algo_.partition(*smallSpanCluster, children, currentAxis);
        // avoid dofData_ deletion
        smallSpanCluster->father = smallSpanCluster;
        delete smallSpanCluster;
    }
    if(largeSpanCluster != NULL && !children.empty())
        children.push_back(largeSpanCluster);
    return dim;
}

}  // end namespace hmat
