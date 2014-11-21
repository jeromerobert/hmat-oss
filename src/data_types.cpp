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

#include "data_types.hpp"

template<> S_t Constants<S_t>::zero = 0.;
template<> D_t Constants<D_t>::zero = 0.;
template<> C_t Constants<C_t>::zero(0., 0.);
template<> Z_t Constants<Z_t>::zero(0., 0.);

template<> S_t Constants<S_t>::pone = 1.;
template<> D_t Constants<D_t>::pone = 1.;
template<> C_t Constants<C_t>::pone(1., 0.);
template<> Z_t Constants<Z_t>::pone(1., 0.);

template<> S_t Constants<S_t>::mone = -1.;
template<> D_t Constants<D_t>::mone = -1.;
template<> C_t Constants<C_t>::mone(-1., 0.);
template<> Z_t Constants<Z_t>::mone(-1., 0.);

template<> const size_t Multipliers<S_t>::add = 1;
template<> const size_t Multipliers<S_t>::mul = 1;
template<> const size_t Multipliers<D_t>::add = 1;
template<> const size_t Multipliers<D_t>::mul = 1;
template<> const size_t Multipliers<C_t>::add = 2;
template<> const size_t Multipliers<C_t>::mul = 6;
template<> const size_t Multipliers<Z_t>::add = 2;
template<> const size_t Multipliers<Z_t>::mul = 6;

template<> int Constants<S_t>::code = 0;
template<> int Constants<D_t>::code = 1;
template<> int Constants<C_t>::code = 2;
template<> int Constants<Z_t>::code = 3;
