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
  \brief Templated Matrix class for recursive algorithms used by HMatrix.
*/
#ifndef RECURSION_HPP
#define RECURSION_HPP

namespace hmat {

  /*! \brief Templated hierarchical matrix class.

  This class defines recursive algorithms used by H-Matrix.
  It uses the CRTP idiom for enhanced readability.
 */

  template <typename T, typename Mat>
  class RecursionMatrix {
  public:
    RecursionMatrix() {}
    ~RecursionMatrix() {}
    void recursiveLdltDecomposition() ;
    void recursiveSolveUpperTriangularRight(Mat* b, bool unitriangular, bool lowerStored) const;
    void recursiveMdmtProduct(const Mat* m, const Mat* d);
    void recursiveSolveLowerTriangularLeft(Mat* b, bool unitriangular, bool mainSolve = false) const;
    void recursiveLuDecomposition() ;
    void recursiveInverseNosym() ;
    void recursiveLltDecomposition() ;
    void recursiveSolveUpperTriangularLeft(Mat* b, bool unitriangular, bool lowerStored, bool mainSolve = false) const;
    void transposeMeta();

    // https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
    // "me()->" replaces "this->" when calling a method of Mat
    Mat* me() {
        return static_cast<Mat*>(this);
    }
    const Mat* me() const {
        return static_cast<const Mat*>(this);
    }
  };

}  // end namespace hmat

#endif // RECURSION_HPP

