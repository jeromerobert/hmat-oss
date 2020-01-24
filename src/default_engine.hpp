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
#include "uncompressed_block.hpp"
#include "uncompressed_values.hpp"
#include "iengine.hpp"

namespace hmat {

template<typename T> class DefaultEngine : public IEngine<T>
{
  NullSettings settings;
public:
  ~DefaultEngine(){}
  typedef hmat::UncompressedBlock<T> UncompressedBlock;
  typedef hmat::UncompressedValues<T> UncompressedValues;
  void destroy(){}
  EngineSettings& GetSettings(){ return settings;}
  static int init();
  static void finalize(){}
  void assembly(Assembly<T>& f, SymmetryFlag sym, bool ownAssembly);
  void factorization(hmat_factorization_t);
  void inverse();
  void gemv(char trans, T alpha, ScalarArray<T>& x, T beta, ScalarArray<T>& y) const;
  void gemm(char transA, char transB, T alpha, const IEngine<T>& a, const IEngine<T>& b, T beta);
  void addRand(double epsilon);
  void solve(ScalarArray<T>& b, hmat_factorization_t) const;
  void solve(IEngine<T>& b, hmat_factorization_t) const;
  void solveLower(ScalarArray<T>& b, hmat_factorization_t t, bool transpose=false) const;
  void copy(IEngine<T> & result, bool structOnly) const;
  void transpose();
  void applyOnLeaf(const hmat::LeafProcedure<hmat::HMatrix<T> >&f);
  IEngine<T>* clone() const { return new DefaultEngine();};
  HMatrix<T> * getHandle() const { return IEngine<T>::hmat; }
  void scale(T alpha);
  void info(hmat_info_t &i) const;
};

}  // end namespace hmat

#endif
