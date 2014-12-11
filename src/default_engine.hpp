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

#ifndef _DEFAULT_ENGINE_HPP
#define _DEFAULT_ENGINE_HPP
#include "h_matrix.hpp"

class NullSettings {};
template<typename T> class DefaultEngine
{
public:
  typedef NullSettings Settings;
  Settings settings;
  explicit DefaultEngine(HMatrix<T>* m = NULL): hmat(m){}
  void destroy(){}
  // this attribute could be in HMatInterface, it's here to avoid making it friend
  HMatrix<T>* hmat;
  static int init() { return 0; }
  static void finalize(){}
  void assembly(AssemblyFunction<T>& f, SymmetryFlag sym, bool synchronize);
  void factorization();
  void gemv(char trans, T alpha, FullMatrix<T>& x, T beta, FullMatrix<T>& y) const;
  void gemm(char transA, char transB, T alpha, const DefaultEngine<T> & a, const DefaultEngine<T>& b, T beta);
  void solve(FullMatrix<T>& b) const;
  void solve(DefaultEngine<T>& b) const;
  void copy(DefaultEngine<T> & result) const {}
  void transpose() {}
  void createPostcriptFile(const char* filename) const;
  void dumpTreeToFile(const char* filename) const;
  double norm() const;
};

#endif
