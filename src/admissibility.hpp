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
#ifndef _ADMISSIBLITY_HPP
#define _ADMISSIBLITY_HPP

#include <cstddef>
#include <string>
#include <map>
#include <vector>

// Forward declarations
class ClusterTree;

class AdmissibilityCondition
{
public:
  AdmissibilityCondition() {}
  virtual ~AdmissibilityCondition() {}
  /*! \brief Returns true if 2 nodes are admissible together.

    This is used for the tree construction on which we develop the HMatrix.
    Two leaves are admissible if they satisfy the criterion allowing the
    compression of the resulting matrix block.

    In the default implementation in the base class, the criterion kept
    is Hackbusch formula :
       min (diameter (), other-> diameter ()) <= eta * distanceTo (other);
    For close interaction matrix computation, the influence radius formula
    is also available :
      influenceRadius() + other->influenceRadius() <= distanceTo (other);
    \param other  the other node of the couple.
    \param eta    a parameter used in the evaluation of the admissibility.
    \param max_size should be used with AcaFull and Svd compression to limit
           memory size of a block. A value of 0 will be ignored and should be
           used with AcaPlus.
    \return true  if 2 nodes are admissible.

   */
  virtual bool isAdmissible(const ClusterTree& rows, const ClusterTree& cols);
  virtual std::string str() const;
};

class StandardAdmissibilityCondition : public AdmissibilityCondition
{
public:
  StandardAdmissibilityCondition(double eta);
  bool isAdmissible(const ClusterTree& rows, const ClusterTree& cols);
  std::string str() const;
private:
  double eta_;
};

class InfluenceRadiusCondition : public AdmissibilityCondition
{
public:
  InfluenceRadiusCondition(int length, double* radii);
  bool isAdmissible(const ClusterTree& rows, const ClusterTree& cols);
  std::string str() const;
private:
  void computeRadii(const ClusterTree& tree);

  std::vector<double> radii_;
  std::map<const ClusterTree*, double> radiiMap_;
};

#endif
