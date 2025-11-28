#pragma once

/*
  HMat-OSS (HMatrix library, open source software)

  Copyright (C) 2019 Airbus S.A.S.

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

#include "hmat/hmat.h"
#include "h_matrix.hpp"
#include "engine_settings.hpp"

namespace hmat {

  template<typename T>
  class IEngine {
  public:
    // this attribute could be in HMatInterface, it's here to avoid making it friend
    HMatrix<T> *hmat;

    virtual void destroy() = 0;
    IEngine(): progress_(NULL) {}
    virtual ~IEngine(){}

    virtual IEngine<T>* clone() const = 0;

    virtual void assembly(Assembly<T> &f, SymmetryFlag sym, bool ownAssembly) = 0;

    virtual void factorization(Factorization) = 0;

    virtual void inverse() = 0;

    virtual void gemv(char trans, T alpha, ScalarArray<T> &x, T beta, ScalarArray<T> &y) const = 0;

    virtual void gemm(char transA, char transB, T alpha, const IEngine<T>& a, const IEngine<T>& b, T beta) = 0;

    virtual void trsm(char side, char uplo, char trans, char diag, T alpha, IEngine<T>     &B) const = 0;
    virtual void trsm(char side, char uplo, char trans, char diag, T alpha, ScalarArray<T> &B) const = 0;

    virtual void addIdentity(T alpha) = 0;
    virtual void addRand(double epsilon) = 0;

    virtual void solve(ScalarArray<T> &b, Factorization) const = 0;

    virtual void solveLower(ScalarArray<T> &b, Factorization t, bool transpose) const = 0;

    virtual void solveLower(IEngine<T> &b, Factorization t, bool transpose) const = 0;

    virtual void transpose() = 0;

    virtual void applyOnLeaf(const hmat::LeafProcedure<hmat::HMatrix<T> > &f) = 0;

    void progress(hmat_progress_t *p) { progress_ = p; }
    hmat_progress_t * progress() const { return progress_; }
    virtual void info(hmat_info_t &i) const =0;

    virtual void profile(hmat_profile_t &p, const std::string& filename = "profile.json") const =0;

    virtual void ratio(hmat_FPCompressionRatio_t &r) const =0;

    virtual void FPcompress() = 0;

    virtual void FPdecompress() = 0;

    virtual FPCompressionSettings GetFPCompressionSettings() = 0;

    virtual void SetFPCompressionSettings(hmat_FPcompress_t compressor, int nb_blocs, float epsilonFP, bool compressFull, bool compressRk) = 0;

    virtual EngineSettings &GetSettings() = 0;

    virtual void setHMatrix(HMatrix<T>* m = NULL){IEngine<T>::hmat = m;}

    virtual void copy(IEngine <T> &result, bool structOnly) const = 0;

    virtual void solve(IEngine<T>& b, Factorization) const = 0;

    virtual void scale(T alpha) = 0;
    virtual typename hmat::Types<T>::dp logdet() const = 0;
    virtual double norm() const = 0;
  protected:
    hmat_progress_t *progress_;
  };

}
