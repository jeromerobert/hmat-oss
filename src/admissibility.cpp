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

#include "admissibility.hpp"
#include "cluster_tree.hpp"

#include "common/my_assert.h"


#include <cmath>
#include <sstream>

bool
AdmissibilityCondition::isAdmissible(const ClusterTree& rows, const ClusterTree& cols)
{
  return false;
}

std::string
AdmissibilityCondition::str() const
{
  return "Not implemented";
}

StandardAdmissibilityCondition::StandardAdmissibilityCondition(double eta)
  : AdmissibilityCondition()
  , eta_(eta)
{
  // Nothing to do
}

bool
StandardAdmissibilityCondition::isAdmissible(const ClusterTree& rows, const ClusterTree& cols)
{
  return std::min(rows.diameter(), cols.diameter()) <= eta_ * rows.distanceTo(&cols);
}

std::string
StandardAdmissibilityCondition::str() const
{
  std::ostringstream oss;
  oss << "Hackbusch formula, with eta = " << eta_;
  return oss.str();
}

InfluenceRadiusCondition::InfluenceRadiusCondition(int length, double* radii)
  : AdmissibilityCondition()
  , radii_(std::vector<double>(radii, radii + length))
{
  // Nothing to do
}

bool
InfluenceRadiusCondition::isAdmissible(const ClusterTree& rows, const ClusterTree& cols)
{
  if (radiiMap_.empty())
  {
    strongAssert(!radii_.empty() && rows.father == NULL && cols.father == NULL);
    computeRadii(rows);
    computeRadii(cols);
  }
  return radiiMap_[&rows] + radiiMap_[&cols] < rows.distanceTo(&cols);
}

void
InfluenceRadiusCondition::computeRadii(const ClusterTree& tree)
{
  double maxRadius = 0.0;
  for (int i = 0; i < tree.data.n; ++i) {
    int externalIndex = tree.data.indices[tree.data.offset + i];
    maxRadius = std::max( maxRadius, radii_[externalIndex] );
  }
  radiiMap_[&tree] = maxRadius;
  if (!tree.isLeaf())
  {
    computeRadii(* static_cast<ClusterTree*>(tree.getChild(0)));
    computeRadii(* static_cast<ClusterTree*>(tree.getChild(1)));
  }
}

std::string
InfluenceRadiusCondition::str() const
{
  return "influence radius formula";
}

