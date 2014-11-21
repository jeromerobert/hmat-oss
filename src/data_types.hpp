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


// Scalar Types
typedef double D_t;
typedef float S_t ;
typedef std::complex<float> C_t;
typedef std::complex<double> Z_t;

typedef enum {
  /*! \brief Simple real type (float in C, REAL in fortran, S_type in MPF) */
  S_TYPE = 0,
  /*! \brief Double real type (double in C, REAL*8 in fortran, D_type in MPF) */
  D_TYPE = 1,
  /*! \brief Simple complex type (doesn't exist in C, COMPLEX in fortran, C_type in MPF) */
  C_TYPE = 2,
  /*! \brief Double complex type (doesn't exist in C, DOUBLE COMPLEX in fortran, Z_type in MPF) */
  Z_TYPE = 3,
  // /*! \brief Number of scalar types available. */
  // nbScalarType = 4
} ScalarTypes;


template<typename T> class Constants {
public:
  static T zero;
  static T pone;
  static T mone;
  static int code;
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
};
template<> struct Types<D_t> {
  typedef D_t dp;
  typedef S_t sp;
};
template<> struct Types<C_t> {
  typedef Z_t dp;
  typedef C_t sp;
};
template<> struct Types<Z_t> {
  typedef Z_t dp;
  typedef C_t sp;
};
#endif
