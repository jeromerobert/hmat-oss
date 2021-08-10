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
  \ingroup Clustering
  \brief Spatial cluster tree for the Dofs.
*/
#ifndef _HMAT_CLUSTERING_HPP
#define _HMAT_CLUSTERING_HPP

#include <vector>
#include <list>
#include <string>

namespace hmat {

// Forward declarations
class ClusterTree;
class DofCoordinates;
class ClusteringAlgorithm;
class AxisAlignedBoundingBox;

class ClusterTreeBuilder {
public:
  explicit ClusterTreeBuilder(const ClusteringAlgorithm& algo);
  ~ClusterTreeBuilder();

  /*! \brief Specify an algorithm for nodes at given depth and below */
  ClusterTreeBuilder& addAlgorithm(int depth, const ClusteringAlgorithm& algo);

  /*! \brief Recursively apply splitting algorithms on points to create a ClusterTree instance.

      \param coordinates node coordinates
      \param group_index optional array containing group numbers of points.  Points of the same group cannot be scattered into different tree leaves.
      \return a ClusterTree instance
   */
  ClusterTree* build(const DofCoordinates& coordinates, int* group_index = NULL) const;

private:
  void divide_recursive(ClusterTree& current, int axis) const;
  void clean_recursive(ClusterTree& current) const;
  ClusteringAlgorithm* getAlgorithm(int depth) const;

private:
  // Sequence of algorithms applied
  std::list<std::pair<int, ClusteringAlgorithm*> > algo_;
};

class ClusteringAlgorithm
{
public:
  /*! \brief Default constructor */
  ClusteringAlgorithm() : maxLeafSize_(-1), divider_(2) {}

  /*! \brief Virtual constructor */
  virtual ClusteringAlgorithm* clone() const = 0;

  /*! \brief Destructor */
  virtual ~ClusteringAlgorithm() {}

  /*! \brief  String representation */
  virtual std::string str() const = 0;

  /*!
   * \brief Split cluster node
   * \param currentAxis the axis used before, to split do the current.
   * This can be -1 for example at the root or if the algorithm does not split along an axis.
   * \return the axis of the partition
   */
  virtual int partition(ClusterTree& current, std::vector<ClusterTree*>& children, int currentAxis) const = 0;

  /*! \brief Called by ClusterTreeBuilder::clean_recursive to free data which may be allocated by partition  */
  virtual void clean(ClusterTree&) const {}

  int getMaxLeafSize() const;
  virtual void setMaxLeafSize(int maxLeafSize);

  int getDivider() const;
  virtual void setDivider(int divider) const;

private:
  int maxLeafSize_;
protected:
  /* the number of children created by division at each level (2 by default) */
  mutable int divider_ ;
};

class AxisAlignClusteringAlgorithm : public ClusteringAlgorithm {
public:
  virtual AxisAlignedBoundingBox* getAxisAlignedBoundingbox(const ClusterTree& node) const;
  void clean(ClusterTree& current) const;
protected:
  void sortByDimension(ClusterTree& node, int dim) const;
  /*!
   * \brief Return the largest dimension of node which is not toAvoid
   * \param toAvoid a dimension which should not be chosen as the largest
   * \param avoidRatio return toAvoid if it's at least avoidRatio longer than the second best
   */
  int largestDimension(const ClusterTree& node, int toAvoid = -1, double avoidRatio = 1.2) const;
  double volume(const ClusterTree& node) const;
};

/*! \brief Creating tree by geometric binary division according to
            the largest dimension.

    For this tree, if a leaf has too many DOFs, the division is done by
    dividing the bounding box in the middle of its largest dimension.
    The boxes of children are resized to the real dimension of the DOFs.
    This method ensures that both 2 children won't be empty, but not the
    equal repartition of DOFs on both sides of the boundary.

    If optional argument axisIndex is set, cyclic splitting is performed
    instead, and this argument is the first axis index.
 */
class GeometricBisectionAlgorithm : public AxisAlignClusteringAlgorithm
{
  /** If true force the first partitioning using the x=0 plane */
  bool x0_;
public:
  explicit GeometricBisectionAlgorithm(bool x0 = false): x0_(x0) {}
  ClusteringAlgorithm* clone() const { return new GeometricBisectionAlgorithm(*this); }
  std::string str() const { return "GeometricBisectionAlgorithm"; }

  int partition(ClusterTree& current, std::vector<ClusterTree*>& children, int currentAxis) const;
};

/*! \brief Creating tree by median division.

  For the division of a leaf, we chose the biggest dimension
  of bounding box and divide it according to this axis. The DOFs are
  sorted by the increasing order on this direction, and are divided
  in the median of this axis. This method ensures that the children
  will have equal number of DOFs, but desn't respect their size criterion.

  If optional argument axisIndex is set, cyclic splitting is performed
  instead, and this argument is the first axis index.

  The bounding boxes of the children are resized to the necessary size.
 */
class MedianBisectionAlgorithm : public AxisAlignClusteringAlgorithm
{
public:
  ClusteringAlgorithm* clone() const { return new MedianBisectionAlgorithm(*this); }
  std::string str() const { return "MedianBisectionAlgorithm"; }

  int partition(ClusterTree& current, std::vector<ClusterTree*>& children, int currentAxis) const;
};

/*! \brief Hybrid algorithm.

  We first split tree node with an MedianBisectionAlgorithm instance, and compute
  ratios of volume of children node divided by volume of current node.  If any ratio
  is larger than a given threshold, this splitting is discarded and replaced by
  a GeometricBisectionAlgorithm instead.
 */
class HybridBisectionAlgorithm : public AxisAlignClusteringAlgorithm
{
public:
  explicit HybridBisectionAlgorithm(double thresholdRatio = 0.8)
    : geometricAlgorithm_(), medianAlgorithm_(), thresholdRatio_(thresholdRatio) {}

  ClusteringAlgorithm* clone() const { return new HybridBisectionAlgorithm(*this); }
  std::string str() const { return "HybridBisectionAlgorithm"; }

  int partition(ClusterTree& current, std::vector<ClusterTree*>& children, int currentAxis) const;
  void clean(ClusterTree& current) const;

private:
  const GeometricBisectionAlgorithm geometricAlgorithm_;
  const MedianBisectionAlgorithm medianAlgorithm_;
  const double thresholdRatio_;
};

/**
 * Isolate DoF with large span to a dedicated cluster
 */
class SpanClusteringAlgorithm: public AxisAlignClusteringAlgorithm {
    const ClusteringAlgorithm &algo_;
    double ratio_;
public:
    /** @param ratio ratio between the number of DOF in a cluster and
     * a span size so a DOF is concidered as large */
    SpanClusteringAlgorithm(const ClusteringAlgorithm &algo, double ratio);
    std::string str() const;
    ClusteringAlgorithm* clone() const;
    int partition(ClusterTree& current, std::vector<ClusterTree*>& children, int currentAxis) const;
};

class VoidClusteringAlgorithm : public ClusteringAlgorithm
{
public:
  explicit VoidClusteringAlgorithm(const ClusteringAlgorithm &algo)
    : ClusteringAlgorithm(algo), algo_(algo.clone()) {}

  ClusteringAlgorithm* clone() const { return new VoidClusteringAlgorithm(*algo_); }
  virtual ~VoidClusteringAlgorithm() { delete algo_; }
  std::string str() const { return "VoidClusteringAlgorithm"; }

  int partition(ClusterTree& current, std::vector<ClusterTree*>& children, int currentAxis) const;
  void clean(ClusterTree& current) const;

private:
  const ClusteringAlgorithm *algo_;
};

class ShuffleClusteringAlgorithm : public AxisAlignClusteringAlgorithm
{
public:
  ShuffleClusteringAlgorithm(const ClusteringAlgorithm &algo, int fromDivider, int toDivider)
    : AxisAlignClusteringAlgorithm(), algo_(algo.clone()), fromDivider_(fromDivider), toDivider_(toDivider) {}

  ClusteringAlgorithm* clone() const { return new ShuffleClusteringAlgorithm(*algo_, fromDivider_, toDivider_); }
  virtual ~ShuffleClusteringAlgorithm() { delete algo_; }
  std::string str() const { return "ShuffleClusteringAlgorithm"; }

  int partition(ClusterTree& current, std::vector<ClusterTree*>& children, int currentAxis) const;
  void clean(ClusterTree& current) const;
  void setMaxLeafSize(int maxLeafSize) { algo_->setMaxLeafSize(maxLeafSize); }
  void setDivider(int divider) const { algo_->setDivider(divider); }

private:
  ClusteringAlgorithm *algo_;
  const int fromDivider_;
  const int toDivider_;
};

class NTilesRecursiveAlgorithm : public AxisAlignClusteringAlgorithm
{
public:
    explicit NTilesRecursiveAlgorithm( int tileSize = 1024 )
        : AxisAlignClusteringAlgorithm(), tileSize_(tileSize) {
    	setMaxLeafSize( tileSize );
    }

    ClusteringAlgorithm* clone() const { return new NTilesRecursiveAlgorithm(*this); }
    std::string str() const { return "NTilesRecursiveAlgorithm"; }

    int subpartition( ClusterTree& father, ClusterTree *current, std::vector<ClusterTree*>& children, int currentAxis ) const;
    int partition(ClusterTree& current, std::vector<ClusterTree*>& children, int currentAxis) const;

private:
    const int tileSize_;
};

}  // end namespace hmat

#endif  /* _HMAT_CLUSTERING_HPP */
