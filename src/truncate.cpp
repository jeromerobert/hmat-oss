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


#include <algorithm>
#include <list>
#include <vector>
#include <cstring>

#include "truncate.hpp"
#include "h_matrix.hpp"
#include "rk_matrix.hpp"


using namespace std;

namespace hmat {

template<typename T> void Truncate<T>::truncate(HMatrix<T>* node, const char * cas){
	return;
}

template<typename T> void EpsilonTruncate<T>::truncate(HMatrix<T>* node, const char * cas){
	RkMatrix<T> * rk_ = node->rk();
	rk_->truncate(epsilon);
	node->rk(rk_);
	return;
}

template class Truncate<S_t>;
template class Truncate<D_t>;
template class Truncate<C_t>;
template class Truncate<Z_t>;

template class EpsilonTruncate<S_t>;
template class EpsilonTruncate<D_t>;
template class EpsilonTruncate<C_t>;
template class EpsilonTruncate<Z_t>;



}  // end namespace hmat
