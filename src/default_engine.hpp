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
#include "hodlr.hpp"

namespace hmat {

template<typename T> class DefaultEngine : public IEngine<T>
{
  NullSettings settings;
  HODLR<T> hodlr;
public:
  ~DefaultEngine(){}
  typedef hmat::UncompressedBlock<T> UncompressedBlock;
  typedef hmat::UncompressedValues<T> UncompressedValues;
  void destroy() override {}
  EngineSettings& GetSettings() override { return settings;}
  static int init();
  static void finalize(){}
  void assembly(Assembly<T>& f, SymmetryFlag sym, bool ownAssembly) override;
  void factorization(Factorization) override;
  void inverse() override ;
  void gemv(char trans, T alpha, ScalarArray<T>& x, T beta, ScalarArray<T>& y) const override;
  void gemm(char transA, char transB, T alpha, const IEngine<T>& a, const IEngine<T>& b, T beta) override;
  void trsm(char side, char uplo, char trans, char diag, T alpha, IEngine<T> &B) const override;
  void trsm(char side, char uplo, char trans, char diag, T alpha, ScalarArray<T> &B) const override;
  void addIdentity(T alpha) override;
  void addRand(double epsilon) override;
  void solve(ScalarArray<T>& b, Factorization) const override;
  void solve(IEngine<T>& b, Factorization) const override ;
  void solveLower(ScalarArray<T>& b, Factorization t, bool transpose=false) const override;
  void solveLower(IEngine<T>& b, Factorization t, bool transpose=false) const override;
  void copy(IEngine<T> & result, bool structOnly) const override;
  void transpose() override;
  void applyOnLeaf(const hmat::LeafProcedure<hmat::HMatrix<T> >&f) override;
  IEngine<T>* clone() const override { return new DefaultEngine();}
  HMatrix<T> * getHandle() const { return IEngine<T>::hmat; }
  void scale(T alpha) override;
  void info(hmat_info_t &i) const override;
  void profile(hmat_profile_t &p, const std::string& filename = "profile.json") const override;
  void ratio(hmat_FPCompressionRatio_t &r) const override;
  void FPcompress() override;
  void FPdecompress() override;
  FPCompressionSettings GetFPCompressionSettings() override;
  void SetFPCompressionSettings(hmat_FPcompress_t compressor, int nb_blocs, float epsilonFP, bool compressFull, bool compressRk) override;
  typename Types<T>::dp logdet() const override;
  double norm() const override;
};

}  // end namespace hmat

#endif
