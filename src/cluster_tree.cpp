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
#include "tree.hpp"
#include "cluster_tree.hpp"
#include "common/my_assert.h"
#include "common/context.hpp"

#include <algorithm>
#include <cstring>

using namespace std;

bool ClusterData::operator==(const ClusterData& o) const {
  // Attention ! On ne fait pas de verification sur les indices, pour
  // pouvoir parler d'egalite entre des noeuds ayant ete generes
  // differemment (differentes matrices).
  return (offset == o.offset) && (n == o.n);
}

bool ClusterData::isSubset(const ClusterData& o) const {
  return (offset >= o.offset) && (offset + n <= o.offset + o.n);
}

bool ClusterData::isSuperSet(const ClusterData& o) const {
  return o.isSubset(*this);
}

bool ClusterData::intersects(const ClusterData& o) const {
  size_t start = max(offset, o.offset);
  size_t end = min(offset + n, o.offset + o.n);
  return (end > start);
}

const ClusterData* ClusterData::intersection(const ClusterData& o) const {
  if (!intersects(o)) {
    return NULL;
  }
  size_t start = max(offset, o.offset);
  size_t end = min(offset + n, o.offset + o.n);
  size_t interN = end - start;
  ClusterData* result = new ClusterData(*this);
  result->offset = start;
  result->n = interN;
  return result;
}

void ClusterData::computeBoundingBox(Point boundingBox[2]) const {
  int* myIndices = indices + offset;
  Point tmp = (*points)[myIndices[0]];
  Point minPoint(tmp.x, tmp.y, tmp.z), maxPoint(tmp.x, tmp.y, tmp.z);

  for (int i = 0; i < n; i++) {
    int index = myIndices[i];
    const Point& p = (*points)[index];
    for (int dim = 0; dim < 3; dim++) {
      minPoint.xyz[dim] = min(minPoint.xyz[dim], p.xyz[dim]);
      maxPoint.xyz[dim] = max(maxPoint.xyz[dim], p.xyz[dim]);
    }
  }
  boundingBox[0] = minPoint;
  boundingBox[1] = maxPoint;
}

ClusterTree::ClusterTree(Point _boundingBox[2], const ClusterData& _data,
                         int _threshold = 100) : Tree<2>(NULL),
                                                 data(_data),
                                                 threshold(_threshold) {
  DECLARE_CONTEXT;
  boundingBox[0] = _boundingBox[0];
  boundingBox[1] = _boundingBox[1];
  memset(&childrenBoundingBoxes, 0, sizeof(Point) * 2 * 2);
  myAssert(data.n > 0);
}

/* Implemente la condition d'admissibilite des bounding box.
 */
bool ClusterTree::isAdmissibleWith(const ClusterTree* other, double eta, size_t max_size) const {
  size_t elements = ((size_t) data.n) * other->data.n;
  if (elements > max_size) {
    return false;
  }
  return min(diameter(), other->diameter()) <= eta * distanceTo(other);
}

double ClusterTree::getEta(const ClusterTree* other) const {
  return min(diameter(), other->diameter()) / distanceTo(other);
}

double ClusterTree::diameter() const {
  return boundingBox[0].distanceTo(boundingBox[1]);
}

double ClusterTree::distanceTo(const ClusterTree* other) const {
  double result = 0.;
  double difference = 0.;

  difference = max(0., boundingBox[0].x - other->boundingBox[1].x);
  result += difference * difference;
  difference = max(0., other->boundingBox[0].x - boundingBox[1].x);
  result += difference * difference;

  difference = max(0., boundingBox[0].y - other->boundingBox[1].y);
  result += difference * difference;
  difference = max(0., other->boundingBox[0].y - boundingBox[1].y);
  result += difference * difference;

  difference = max(0., boundingBox[0].z - other->boundingBox[1].z);
  result += difference * difference;
  difference = max(0., other->boundingBox[0].z - boundingBox[1].z);
  result += difference * difference;

  return sqrt(result);
}

void ClusterTree::divide() {
  DECLARE_CONTEXT;
  myAssert(isLeaf());
  if (data.n <= (size_t) threshold) {
    return;
  }

  int dim = largestDimension();
  sortByDimension(dim);
  int separatorIndex = findSeparatorIndex();
  ClusterData leftData(data.indices, data.offset, separatorIndex, data.points);
  ClusterData rightData(data.indices, data.offset + separatorIndex,
			data.n - separatorIndex, data.points);
  leftData.computeBoundingBox(childrenBoundingBoxes[0]);
  rightData.computeBoundingBox(childrenBoundingBoxes[1]);

  ClusterTree *leftChild = this->make(childrenBoundingBoxes[0], leftData,
				      threshold);
  ClusterTree *rightChild = this->make(childrenBoundingBoxes[1], rightData,
				       threshold);
  strongAssert(leftChild);
  strongAssert(rightChild);
  leftChild->depth = depth + 1;
  rightChild->depth = depth + 1;
  insertChild(0, leftChild);
  insertChild(1, rightChild);

  leftChild->divide();
  rightChild->divide();
}

ClusterTree* ClusterTree::copy(const ClusterTree* copyFather) const {
  ClusterTree* result = NULL;
  if (!copyFather) {
    // La racine doit s'occuper le tableau des points et le mapping.
    myAssert(data.n == data.points->size());
    size_t n = data.points->size();
    int* copyIndices = new int[n];
    memcpy(copyIndices, data.indices, sizeof(int) * n);
    vector<Point>* copyPoints = new vector<Point>(n);
    *copyPoints = *data.points;
    ClusterData rootData(copyIndices, 0, n, copyPoints);
    result = this->make((Point*) boundingBox, rootData, threshold);
    copyFather = result;
  } else {
    ClusterData copyData(copyFather->data.indices, data.offset, data.n,
                         copyFather->data.points);
    result = this->make((Point*) boundingBox, copyData, threshold);
  }
  if (!isLeaf()) {
    result->insertChild(0, ((ClusterTree*) getChild(0))->copy(copyFather));
    result->insertChild(1, ((ClusterTree*) getChild(1))->copy(copyFather));
  }
  return result;
}


// GeometricBisectionClusterTree
int GeometricBisectionClusterTree::findSeparatorIndex() const {
  int dim = largestDimension();
  double middle = .5 * (boundingBox[0].xyz[dim] + boundingBox[1].xyz[dim]);
  int middleIndex = 0;
  int* myIndices = data.indices + data.offset;
  while ((*data.points)[myIndices[middleIndex]].xyz[dim] < middle) {
    middleIndex++;
  }
  return middleIndex;
}

ClusterTree* GeometricBisectionClusterTree::make(Point _boundingBox[2],
                                                 const ClusterData& _data,
                                                 int _threshold) const {
  return new GeometricBisectionClusterTree(_boundingBox, _data, _threshold);
}


// MedianBisectionClusterTree
int MedianBisectionClusterTree::findSeparatorIndex() const {
  return data.n / 2;
}

ClusterTree* MedianBisectionClusterTree::make(Point _boundingBox[2],
                                              const ClusterData& _data,
                                              int _threshold) const {
  return new MedianBisectionClusterTree(_boundingBox, _data, _threshold);
}

int ClusterTree::largestDimension() const {
  int maxDim = -1;
  double maxSize = -1;
  for (int i = 0; i < 3; i++) {
    double size = (boundingBox[1].xyz[i] - boundingBox[0].xyz[i]);
    if (size > maxSize) {
      maxSize = size;
      maxDim = i;
    }
  }
  return maxDim;
}

void ClusterTree::sortByDimension(int dim) {
  int* myIndices = data.indices + data.offset;
  switch (dim) {
  case 0:
    sort(myIndices, myIndices + data.n, IndicesComparator<0>(data));
    break;
  case 1:
    sort(myIndices, myIndices + data.n, IndicesComparator<1>(data));
    break;
  case 2:
    sort(myIndices, myIndices + data.n, IndicesComparator<2>(data));
    break;
  default:
    strongAssert(false);
  }
}


static double volume(const Point boundingBox[2]) {
  double result = 1.;
  for (int dim = 0; dim < 3; dim++) {
    result *= (boundingBox[1].xyz[dim] - boundingBox[0].xyz[dim]);
  }
  return result;
}

const double thresholdRatio = .8;
int HybridBisectionClusterTree::findSeparatorIndex() const {
  // Change de version de decoupage en fonction de la taille realtive des
  // boites. On essaie de prendre le decoupage selon la mediane. Si celui-ci
  // donne une reduction trop faible de la taille des boites englobantes, alors
  // on passe sur le critere geometrique.
  int index = data.n / 2;
  Point leftBoundingBox[2], rightBoundingBox[2];
  {
    ClusterData leftData(data.indices, data.offset, index, data.points);
    ClusterData rightData(data.indices, data.offset + index, data.n - index,
			  data.points);
    leftData.computeBoundingBox(leftBoundingBox);
    rightData.computeBoundingBox(rightBoundingBox);
  }
  double currentVolume = volume(boundingBox);
  double leftVolume = volume(leftBoundingBox);
  double rightVolume = volume(rightBoundingBox);
  double maxRatio = max(rightVolume / currentVolume, leftVolume / currentVolume);
  if (maxRatio > thresholdRatio) {
    int dim = largestDimension();
    double middle = .5 * (boundingBox[0].xyz[dim] + boundingBox[1].xyz[dim]);
    int middleIndex = 0;
    int* myIndices = data.indices + data.offset;
    while ((*data.points)[myIndices[middleIndex]].xyz[dim] < middle) {
      middleIndex++;
    }
    return middleIndex;
  } else {
    return index;
  }
}

ClusterTree* HybridBisectionClusterTree::make(Point _boundingBox[2],
					      const ClusterData& _data,
					      int _threshold) const {
  return new HybridBisectionClusterTree(_boundingBox, _data, _threshold);
}
