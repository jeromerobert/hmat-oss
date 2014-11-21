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
#include "tree.hpp"
#include <vector>
#include <cmath>


extern size_t maxElementsPerBlock;

/*! \brief Class representing a 3D point.
 */
class Point {
public:
  union {
    struct {
      double x, y, z; /// Coordinates
    };
    double xyz[3]; /// Idem
  };
  Point(double _x = 0., double _y = 0., double _z = 0.) : x(_x), y(_y), z(_z) {}
  /*! \brief Return d(this, other)
   */
  inline double distanceTo(const Point &other) const {
    double result = 0.;
    double difference = 0.;
    difference = x - other.x;
    result += difference * difference;
    difference = y - other.y;
    result += difference * difference;
    difference = z - other.z;
    result += difference * difference;
    return sqrt(result);
  }
};


/*! \brief Data held by a node of a \a ClusterTree.
 */
class ClusterData {
public:
  /// Global indices array
  int *indices;
  /// offset of the start of this node's data in indices
  size_t offset;
  /// length of this node's data in indices
  size_t n;
  const std::vector<Point>* points;

public:
  /*! \brief Compare two indices set for equality.
   */
  bool operator==(const ClusterData& o) const;
  /*! \brief Return true if this is a subset of o.

      \param o ClusterData
      \return true if this is a subset of o
   */
  bool isSubset(const ClusterData& o) const;
  /*! \brief Return true if this is a strict subset of o.
   */
  bool isStrictSubset(const ClusterData& o) const {
    return isSubset(o) && (!(*this == o));
  };
  /*! \brief Return o.isSubset(*this)
   */
  bool isSuperSet(const ClusterData& o) const;
  /*! \brief Return o.isStrictSubset(*this)
   */
  bool isStrictSuperSet(const ClusterData& o) const {
    return isSuperSet(o) && (!(*this == o));
  };
  /** Return true if two index sets intersect.
   */
  bool intersects(const ClusterData& o) const;
  /** Return the intersection of two index sets, NULL if the intersection is empty.
   */
  const ClusterData* intersection(const ClusterData& o) const;

  /** Compute the bounding box of a ClusterData set.
   */
  void computeBoundingBox(Point boundingBox[2]) const;

  ClusterData(int* _indices, size_t _offset, size_t _n, const std::vector<Point>* _points)
    : indices(_indices), offset(_offset), n(_n), points(_points) {}
};

/*! \brief Comparateur pour deux indices selon une de leur coordonnee.

  On utilise les templates pour eviter un trop grand cout a
  l'execution ou le copier-coller, avec pour le parametre du template :
    - 0 pour une comparaison selon x
    - 1 pour une comparaison selon y
    - 2 pour une comparaison selon z
 */
template<int N> class IndicesComparator {
private:
  ClusterData& data;

public:
  IndicesComparator(ClusterData& _data) : data(_data) {}
  bool operator() (int i, int j) {
    return (*data.points)[i].xyz[N] < (*data.points)[j].xyz[N];
  }
};

/*! \brief Abstract Base Class for the Cluster trees.

  This class defines the index set division of a problem. Several divisions
  being possible, this is an abstract class. It however only allows bisection of
  the index set.

  \warning During the tree creation, it is mandatory to call
  ClusterTree::divide() on the newly created tree, to really create the tree
  structure.
 */
class ClusterTree : public Tree<2> {
public:
  /*! min and max points */
  Point boundingBox[2];
  /*! Admissibility criteria for the tree nodes. */
  static double eta;
  /*! Data */
  ClusterData data;

protected:
  /*! Bounding boxes of the children */
  Point childrenBoundingBoxes[2][2];
  /*! Max number of Dofs in a leaf */
  int threshold;

public:
  /*! \brief Create a leaf.

      \param _boundingBox bounding box of this leaf
      \param clusterData data held by this leaf
      \param _threshold max number of indices in a leaf. Used for the recursive division.
   */
  ClusterTree(Point _boundingBox[2], const ClusterData& _data, int _threshold);
  /*! \brief Returns true if 2 nodes are admissible together.

    This is used for the tree construction on which we develop the HMatrix.
    Two leaves are admissible if they satisfy the criterion allowing the
    compression of the resulting matrix block.

    In the default implementation in the base class, the criterion kept is:
       min (diameter (), other-> diameter ()) <= eta * distanceTo (other);

    There is also a criterion of size on the resulting block, which
    should not exceed 10000000 elements (160MB for double complex precision).
    TODO: making the criterion adjustable.

    \param other  the other node of the couple.
    \param eta    a parameter used in the evaluation of the admissibility.
    \return true  if 2 nodes are admissible.

   */
  bool isAdmissibleWith(const ClusterTree* other, double eta = 10.) const;
  /*! \brief Returns the admissibility parameter eta corresponding to two clusters.

    As described in \a ClusterTree::isAdmissibleWith documentation,
    this criteria is defined by:

      eta = min(diameter(), other->diameter()) / distanceTo(other);

    The lower it is, the farther away are the two boxes. It is thus
    linked with the compression ratio one can expect from a block, the
    lower eta is, the more the block can be compressed. It can be used
    as a parameter in a crude a-priori work estimation method.
   */
  double getEta(const ClusterTree* other) const;
  /*! \brief Divide the tree to create its final structure.

    \warning It is mandatory to call this function before using a newly created tree.
   */
  void divide();

  /*! \brief Return a copy to this.
   */
  ClusterTree* copy(const ClusterTree* copyFather=NULL) const;

protected:
  /*! Diameter of the node */
  double diameter() const;
  /*! Distance to an other node */
  double distanceTo(const ClusterTree* other) const;
  /*! Returns true if a point is inside a given children bounding box.

    \param p The points
    \param index the index of the children bounding box to consider.
   */
  inline bool isPointInBox(const Point p, int boxIndex) const {
    Point box[2] = {childrenBoundingBoxes[boxIndex][0],
                    childrenBoundingBoxes[boxIndex][1]};
    return ((p.x >= box[0].x) && (p.x <= box[1].x)
            && (p.y >= box[0].y) && (p.y <= box[1].y)
            && (p.z >= box[0].z) && (p.z <= box[1].z));
  }
  /** Return the largest dimension of the node bounding box.
   */
  int largestDimension() const;
  /** Sort the ClusterData indices of this node according to a given dimension.

      \parak dim The dimension.
   */
  void sortByDimension(int dim);
  /** Find the index of the separation between the two sons.
   */
  virtual int findSeparatorIndex() const = 0;
  /*! \brief Make a new ClusterTree*.

    This function has the same arguments as \a
    ClusterTree::ClusterTree(). It is used to make an instance of the
    right derived type.
   */
  virtual ClusterTree* make(Point _boundingBox[2], const ClusterData& _data, int _threshold) const = 0;
};


/*! \brief Creating tree by geometric binary division according to
            the largest dimension.

    For this tree, if a leaf has too many DOFs, the division is done by
    dividing the bounding box in the middle of its biggest dimension.
    The boxes of children are resized to the real dimension of the DOFs.
    This method ensures that both 2 children won't be empty, but not the
    equal repartition of DOFs on both sides of the boundary.
 */
class GeometricBisectionClusterTree : public ClusterTree {
public:
  GeometricBisectionClusterTree(Point _boundingBox[2],
                                const ClusterData& _data, int _threshold)
    : ClusterTree(_boundingBox, _data, _threshold) {}

protected:
  int findSeparatorIndex() const;
  ClusterTree* make(Point _boundingBox[2], const ClusterData& _data, int _threshold) const;
};

/*! \brief Creating tree by median division.

  For the division of a leaf, we chose the biggest dimension
  of bounding box and divide it according to this axis. The DOFs are
  sorted by the increasing order on this direction, and are divided
  in the median of this axis. This method ensures that the children
  will have equal number of DOFs, but desn't respect their size criterion.

  The bounding boxes of the children are resized to the necessary size.
 */
class MedianBisectionClusterTree : public ClusterTree {
public:
  MedianBisectionClusterTree(Point _boundingBox[2],
                             const ClusterData& _data, int _threshold)
    : ClusterTree(_boundingBox, _data, _threshold) {}

protected:
  int findSeparatorIndex() const;
  ClusterTree* make(Point _boundingBox[2], const ClusterData& _data, int _threshold) const;
};

class HybridBisectionClusterTree : public ClusterTree {
public:
  HybridBisectionClusterTree(Point _boundingBox[2],
                             const ClusterData& _data, int _threshold)
    : ClusterTree(_boundingBox, _data, _threshold) {}

protected:
  int findSeparatorIndex() const;
  ClusterTree* make(Point _boundingBox[2], const ClusterData& _data, int _threshold) const;
};
#endif
