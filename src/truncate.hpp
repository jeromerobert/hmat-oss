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
  \brief Implementation of the truncate function
*/
#ifndef _TRUNCATE_HPP
#define _TRUNCATE_HPP

#include "h_matrix.hpp"
#include <cassert>
#include <fstream>
#include <iostream>


namespace hmat {

/** Class to write user defined method to  truncate the Rk matrix.

    This class is used by truncate.
 */
template<typename T> class Truncate {

public:
	Truncate(){}
	virtual void truncate(HMatrix<T>* node, const char * cas = "");
};

template<typename T>
class EpsilonTruncate : public Truncate<T> {
private:
	double epsilon;
public:
	EpsilonTruncate(double epsilon_):epsilon(epsilon_){}
	virtual void truncate(HMatrix<T>* node, const char * cas = "");
};


}  // end namespace hmat

#endif
