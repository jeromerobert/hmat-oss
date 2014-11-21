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

/*! Disable MKL and OpenMP threading inside a block using RAII.

  The HMatrix solver doesn't use a multithreaded BLAS, nor OpenMP. It is
  actually important for optimal performance to *not* use any threading.

  This class is not meant to be used by itself, but rather using the \a
  DISABLE_THREADING_IN_BLOCK macro to disable threading in a block, and restore
  it to its original setting at the end.
 */

class DisableThreadingInBlock {
private:
  int mklNumThreads;
  int ompNumThreads;
  int openblasNumThreads;
public:
  DisableThreadingInBlock();
  ~DisableThreadingInBlock();
};

/** Disable OpenMP and MKL (if available) threading in a block, and restore it
    at the end.
 */
#define DISABLE_THREADING_IN_BLOCK DisableThreadingInBlock dummyDisableThreadingInBlock

