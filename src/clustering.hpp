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
  void divide_recursive(ClusterTree& current) const;
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

  /*! \brief Split cluster node */
  virtual void partition(ClusterTree& current, std::vector<ClusterTree*>& children) const = 0;

  /*! \brief Called by ClusterTreeBuilder::clean_recursive to free data which may be allocated by partition  */
  virtual void clean(ClusterTree&) const {}

  int getMaxLeafSize() const;
  void setMaxLeafSize(int maxLeafSize);

  int getDivider() const;
  void setDivider(int divider);

private:
  int maxLeafSize_;
protected:
  /* the number of children created by division at each level (2 by default) */
  int divider_ ;
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
class GeometricBisectionAlgorithm : public ClusteringAlgorithm
{
public:
  explicit GeometricBisectionAlgorithm(int axisIndex = -1)
    : ClusteringAlgorithm(), axisIndex_(axisIndex), spatialDimension_(-1) {}

  ClusteringAlgorithm* clone() const { return new GeometricBisectionAlgorithm(*this); }
  std::string str() const { return "GeometricBisectionAlgorithm"; }

  void partition(ClusterTree& current, std::vector<ClusterTree*>& children) const;
  void clean(ClusterTree& current) const;

private:
  mutable int axisIndex_;
  mutable int spatialDimension_;
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
class MedianBisectionAlgorithm : public ClusteringAlgorithm
{
public:
  explicit MedianBisectionAlgorithm(int axisIndex = -1)
    : ClusteringAlgorithm(), axisIndex_(axisIndex), spatialDimension_(-1) {}

  ClusteringAlgorithm* clone() const { return new MedianBisectionAlgorithm(*this); }
  std::string str() const { return "MedianBisectionAlgorithm"; }

  void partition(ClusterTree& current, std::vector<ClusterTree*>& children) const;
  void clean(ClusterTree& current) const;

private:
  mutable int axisIndex_;
  mutable int spatialDimension_;
};

/*! \brief Hybrid algorithm.

  We first split tree node with an MedianBisectionAlgorithm instance, and compute
  ratios of volume of children node divided by volume of current node.  If any ratio
  is larger than a given threshold, this splitting is discarded and replaced by
  a GeometricBisectionAlgorithm instead.
 */
class HybridBisectionAlgorithm : public ClusteringAlgorithm
{
public:
  explicit HybridBisectionAlgorithm(double thresholdRatio = 0.8)
    : ClusteringAlgorithm()
    , geometricAlgorithm_(0)
    , medianAlgorithm_(0)
    , thresholdRatio_(thresholdRatio)
    {}

  ClusteringAlgorithm* clone() const { return new HybridBisectionAlgorithm(*this); }
  std::string str() const { return "HybridBisectionAlgorithm"; }

  void partition(ClusterTree& current, std::vector<ClusterTree*>& children) const;
  void clean(ClusterTree& current) const;

private:
  const GeometricBisectionAlgorithm geometricAlgorithm_;
  const MedianBisectionAlgorithm medianAlgorithm_;
  const double thresholdRatio_;
};

class VoidClusteringAlgorithm : public ClusteringAlgorithm
{
public:
  explicit VoidClusteringAlgorithm(const ClusteringAlgorithm &algo)
    : ClusteringAlgorithm(), algo_(algo.clone()) {}

  ClusteringAlgorithm* clone() const { return new VoidClusteringAlgorithm(*this); }
  std::string str() const { return "VoidClusteringAlgorithm"; }

  void partition(ClusterTree& current, std::vector<ClusterTree*>& children) const;
  void clean(ClusterTree& current) const;

private:
  const ClusteringAlgorithm *algo_;
};

}  // end namespace hmat

#endif  /* _HMAT_CLUSTERING_HPP */
