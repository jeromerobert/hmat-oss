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
  \ingroup HMatrix
  \brief Spatial cluster tree for the Dofs.
*/
#ifndef _CLUSTER_TREE_HPP
#define _CLUSTER_TREE_HPP

#include <sstream>
#include "tree.hpp"
#include "clustering.hpp"
#include "coordinates.hpp"
#include "hmat/hmat.h"
#include <vector>
#include <cmath>

namespace hmat {


class IndexSet {
public:
  IndexSet() : offset_(-1), size_(0) {}
  IndexSet(int offset, int size) : offset_(offset), size_(size) {}

  /*! \brief Compare two indices set for equality.
   */
  bool operator==(const IndexSet& o) const;
  /*! \brief Get the number of indices in this set.
   */
  inline int size() const { return size_; }

  /*! \brief Get the offset of first index
   */
  inline int offset() const { return offset_; }

  /*! \brief Return true if this is a subset of o.

      \param o IndexSet
      \return true if this is a subset of o
   */
  bool isSubset(const IndexSet& o) const;
  /*! \brief Return true if this is a strict subset of o.
   */
  bool isStrictSubset(const IndexSet& o) const {
    return isSubset(o) && (!(*this == o));
  };
  /*! \brief Return o.isSubset(*this)
   */
  bool isSuperSet(const IndexSet& o) const;
  /*! \brief Return o.isStrictSubset(*this)
   */
  bool isStrictSuperSet(const IndexSet& o) const {
    return isSuperSet(o) && (!(*this == o));
  };
  /** Return true if two index sets intersect.
   */
  bool intersects(const IndexSet& o) const;

  /** Set this as the intersection of two index set */
  void intersection(const IndexSet& s1, const IndexSet& s2);

  /*! \brief Return a short string describing the content of this IndexSet for debug (like: "[320, 452]")
    */
  std::string description() const {
    std::ostringstream convert;   // stream used for the conversion
    convert << "[" << offset() << ", " << size() << "]" ;
    return convert.str();
  }

protected:
  /// offset of the start of this node's data in indices
  int offset_;
  /// length of this node's data in indices
  int size_;

};

/*! \brief Geometric data.
 */
class DofData {
friend class ClusterData;
public:
  explicit DofData(const DofCoordinates& coordinates, int* group_index = NULL);
  ~DofData();

  DofData* copy() const;
  inline int size() const { return coordinates_->numberOfDof(); }

private:
  /// Indices array
  int *perm_i2e_;
  int *perm_e2i_;
  int *group_index_;
  /// Coordinates
  const DofCoordinates* coordinates_;
};

class ClusterData : public IndexSet {
friend class ClusterTree;
public:
  ClusterData(const DofData* dofData) : IndexSet(0, dofData->size()), dofData_(dofData) {}
  ClusterData(const ClusterData& data) : IndexSet(data), dofData_(data.dofData_) {}
  ClusterData(const ClusterData& data, int offset, int size) : IndexSet(offset, size), dofData_(data.dofData_) {}
  /**
   * @brief indices()[i] is the index in the original coordinate
   * array which match the ith point in the underlying hmat/dofData_
   */
  inline int* indices() const { return dofData_->perm_i2e_; }
  /**
   * @brief indices_rev()[i] is the index in the underlying hmat/dofData_
   * which match the ith point in the original coordinate array
   */
  inline int* indices_rev() const { return dofData_->perm_e2i_; }
  inline const DofCoordinates* coordinates() const { return dofData_->coordinates_; }
  inline int* group_index() const { return dofData_->group_index_; }
  /*! Move a degree of freedom into its right sibling  */
  void moveDoF(int index, ClusterData* right);
  /** @brief debug method that check the underlying indices arrays are valid */
  void assertValid();

private:
  const DofData* dofData_;
};

/*! \brief Abstract Base Class for the Cluster trees.

  This class defines the index set division of a problem. Several divisions
  being possible, this is an abstract class. It however only allows bisection of
  the index set.

  \warning During the tree creation, it is mandatory to call
  ClusterTree::divide() on the newly created tree, to really create the tree
  structure.
 */
class ClusterTree : public Tree<ClusterTree> {
public:
  /*! Data */
  ClusterData data;

  /*! Opaque pointer which may be used by algorithms */
  mutable void* cache_;

public:
  /*! \brief Create a leaf.

      \param _boundingBox bounding box of this leaf
      \param clusterData data held by this leaf
      \param _threshold max number of indices in a leaf. Used for the recursive division.
   */
  explicit ClusterTree(const DofData* dofData);
    explicit ClusterTree(const DofData* dofData, int offset, int size);

  /* Copy constructor */
  ClusterTree(const ClusterTree& other);

  virtual ~ClusterTree();

  ClusterTree* slice(int offset, int size) const;

  /*! \brief Return a copy to this.
   */
  ClusterTree* copy(const ClusterTree* copyFather=NULL) const;

  /*! \brief Swap contents of two ClusterTree
   */
  void swap(ClusterTree* other);

  /*! \brief Return a short string describing the content of this ClusterTree for debug (like: "[320, 452]")
    */
  std::string description() const {
    return data.description();
  }

};

class AxisAlignedBoundingBox
{
    const unsigned dimension_;
    double * bb_;
public:
    explicit AxisAlignedBoundingBox(const ClusterData& node);
    ~AxisAlignedBoundingBox();
    double extends(int dim) const;
    int greatestDim() const;
    double diameter() const;
    double diameterSqr() const;
    double distanceTo(const AxisAlignedBoundingBox& other) const;
    double distanceToSqr(const AxisAlignedBoundingBox& other) const;
    const double * bbMin() const { return bb_; }
    const double * bbMax() const { return bb_ + dimension_; }
    void bbMin(const double * bbMin);
    void bbMax(const double * bbMax);
    /** @brief Square of the distance between the centers of 2 boxes */
    double ccDistSqr(const AxisAlignedBoundingBox& other) const {
      double r = 0;
      for(unsigned int i = 0; i < dimension_; i++) {
        double v = (bbMin()[i] + bbMax()[i]) - (other.bbMin()[i] + other.bbMax()[i]);
        r += v * v;
      }
      return r / 4;
    }
};

}  // end namespace hmat

#endif
