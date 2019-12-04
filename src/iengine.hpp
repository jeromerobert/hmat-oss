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

    virtual void factorization(hmat_factorization_t) = 0;

    virtual void inverse() = 0;

    virtual void gemv(char trans, T alpha, ScalarArray<T> &x, T beta, ScalarArray<T> &y) const = 0;

    virtual void gemm(char transA, char transB, T alpha, const IEngine<T>& a, const IEngine<T>& b, T beta) = 0;

    virtual void addRand(double epsilon) = 0;

    virtual void solve(ScalarArray<T> &b, hmat_factorization_t) const = 0;

    virtual void solveLower(ScalarArray<T> &b, hmat_factorization_t t, bool transpose) const = 0;

    virtual void transpose() = 0;

    virtual void applyOnLeaf(const hmat::LeafProcedure<hmat::HMatrix<T> > &f) = 0;

    void progress(hmat_progress_t *p) { progress_ = p; }

    void info(hmat_info_t &i) const { hmat->info(i); }

    virtual EngineSettings &GetSettings() = 0;

    virtual void setHMatrix(HMatrix<T>* m = NULL){IEngine<T>::hmat = m;}

    virtual void copy(IEngine <T> &result, bool structOnly) const = 0;

    virtual void solve(IEngine<T>& b, hmat_factorization_t) const = 0;

    virtual void scale(T alpha) = 0;
  protected:
    hmat_progress_t *progress_;
  };

}
