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

#ifndef _DATA_TYPES_HPP
#define _DATA_TYPES_HPP

#include <complex>


namespace hmat {

// Scalar Types
typedef double D_t;
typedef float S_t ;
typedef std::complex<float> C_t;
typedef std::complex<double> Z_t;

typedef enum {
  /*! \brief Simple real type (float in C, REAL in fortran) */
  S_TYPE = 0,
  /*! \brief Double real type (double in C, REAL*8 in fortran) */
  D_TYPE = 1,
  /*! \brief Simple complex type (doesn't exist in C, COMPLEX in fortran) */
  C_TYPE = 2,
  /*! \brief Double complex type (doesn't exist in C, DOUBLE COMPLEX in fortran) */
  Z_TYPE = 3,
  // /*! \brief Number of scalar types available. */
  // nbScalarType = 4
} ScalarTypes;


template<typename T> class Constants {
public:
  static const T zero;
  static const T pone;
  static const T mone;
  static const int code;
};

/** Multipliers used in the operations count.

    The operations are more expansive on complex numbers. This class contains
    the ratios for the various types used here (SDCZ).
 */
template<typename T> class Multipliers {
public:
  static const size_t mul;
  static const size_t add;
};

/** Trait class used to get information about the types.
*/
template<typename T> struct Types {
  // Simple and double precision equivalents to this type.
  typedef void dp;
  typedef void sp;
};

template<> struct Types<S_t> {
  typedef D_t dp;
  typedef S_t sp;
  static const ScalarTypes TYPE = S_TYPE;
};
template<> struct Types<D_t> {
  typedef D_t dp;
  typedef S_t sp;
  static const ScalarTypes TYPE = D_TYPE;
};
template<> struct Types<C_t> {
  typedef Z_t dp;
  typedef C_t sp;
  static const ScalarTypes TYPE = C_TYPE;
};
template<> struct Types<Z_t> {
  typedef Z_t dp;
  typedef C_t sp;
  static const ScalarTypes TYPE = Z_TYPE;
};

template<typename T>
double squaredNorm(const T x) {
  return x * x;
}

// Specializations for complex values
template<>
inline double squaredNorm(const C_t x) {
// std::norm seems deadfully slow on Intel 15
#ifdef __INTEL_COMPILER
  const float x_r = x.real();
const float x_i = x.imag();
return x_r*x_r + x_i*x_i;
#else
  return std::norm(x);
#endif
}

template<>
inline double squaredNorm(const Z_t x) {
#ifdef __INTEL_COMPILER
  const double x_r = x.real();
const double x_i = x.imag();
return x_r*x_r + x_i*x_i;
#else
  return std::norm(x);
#endif
}

}  // end namespace hmat

#endif
