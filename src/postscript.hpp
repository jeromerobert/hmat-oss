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

#ifndef _POSTSCRIPT_HPP
#define _POSTSCRIPT_HPP

#include "tree.hpp"
#include <fstream>

namespace hmat {

template<typename T> class HMatrix;

/*! \brief Create a Postscript file representing the HMatrix.

  The result .ps file shows the matrix structure and the compression ratio. In
  the output, red = full block, green = compressed. The darker the green, the
  worst the compression ration is. There is saturation at black when the block
  size is divided by less than 5.
 */
template<typename T> class PostscriptDumper
{
public:
    void write(const Tree<4> * tree, const std::string& filename) const;
protected:
    virtual const HMatrix<T> * cast(const Tree<4> * tree) const;
    virtual void drawMatrix(const Tree<4> * tree, const HMatrix<T> *,
        std::ofstream& f, int depth, double scale, bool cross=true) const;
private:
    void recursiveDrawing(const Tree<4> * tree, std::ofstream& f, int depth, double scale) const;
};

}  // end namespace hmat

#endif  // _POSTSCRIPT_HPP
