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
#include <cstdint>
#include <cstring>

namespace hmat {

template<> const size_t Multipliers<S_t>::add = 1;
template<> const size_t Multipliers<S_t>::mul = 1;
template<> const size_t Multipliers<D_t>::add = 1;
template<> const size_t Multipliers<D_t>::mul = 1;
template<> const size_t Multipliers<C_t>::add = 2;
template<> const size_t Multipliers<C_t>::mul = 6;
template<> const size_t Multipliers<Z_t>::add = 2;
template<> const size_t Multipliers<Z_t>::mul = 6;

template<> const int Constants<S_t>::code = 0;
template<> const int Constants<D_t>::code = 1;
template<> const int Constants<C_t>::code = 2;
template<> const int Constants<Z_t>::code = 3;

template<> bool swIsFinite(double x) {
  uint64_t bits;
  // casting break strict aliasing rules and std::bit_cast is C++20 only
  std::memcpy(&bits, &x, sizeof(double));
  // FIXME: This trick fool gcc up to 14.x and clang up to 18.x. Clang 19 will
  // optimize and always return true when using -ffast-math
  return (bits & 0x7ff0000000000000) != 0x7ff0000000000000;
}

template<> bool swIsFinite(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(float));
  return (bits & 0x7f800000) != 0x7f800000;
}

}  // end namespace hmat
