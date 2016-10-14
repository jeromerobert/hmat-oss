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

#include "fromdouble.hpp"
#include <assert.h>

namespace hmat {

template<>
ScalarArray<D_t>* fromDoubleScalarArray(ScalarArray<D_t>* d, bool) {
  return d;
}
template<>
ScalarArray<Z_t>* fromDoubleScalarArray(ScalarArray<Z_t>* d, bool) {
  return d;
}

template<typename T>
ScalarArray<T>* fromDoubleScalarArray(ScalarArray<typename Types<T>::dp>* d, bool del) {
  if (!d) {
    return NULL;
  }
  ScalarArray<T>* result = new ScalarArray<T>(d->rows, d->cols);
  assert(result);
  const size_t size = ((size_t) d->rows) * d->cols;
  T* const r = result->m;
  const typename Types<T>::dp* m = d->m;
  for (size_t i = 0; i < size; ++i) {
    r[i] = T(m[i]);
  }
  if (del)
    delete d;
  return result;
}

template ScalarArray<S_t>* fromDoubleScalarArray(ScalarArray<Types<S_t>::dp>* d, bool);
template ScalarArray<C_t>* fromDoubleScalarArray(ScalarArray<Types<C_t>::dp>* d, bool);

template<>
FullMatrix<D_t>* fromDoubleFull(FullMatrix<D_t>* f) {
  return f;
}
template<>
FullMatrix<Z_t>* fromDoubleFull(FullMatrix<Z_t>* f) {
  return f;
}

template<typename T>
FullMatrix<T>* fromDoubleFull(FullMatrix<typename Types<T>::dp>* f) {
  if (!f)
    return NULL;
  FullMatrix<T>* result = new FullMatrix<T>(fromDoubleScalarArray<T>(&f->data, false), f->rows_, f->cols_); // 'false' because we cant delete f->data (its not a pointer)
  delete f;
  return result;
}

template FullMatrix<S_t>* fromDoubleFull(FullMatrix<Types<S_t>::dp>* f);
template FullMatrix<C_t>* fromDoubleFull(FullMatrix<Types<C_t>::dp>* f);

template<> RkMatrix<D_t>* fromDoubleRk(RkMatrix<D_t>* rk) {
  return rk;
}
template<> RkMatrix<Z_t>* fromDoubleRk(RkMatrix<Z_t>* rk) {
  return rk;
}

template<typename T> RkMatrix<T>* fromDoubleRk(RkMatrix<typename Types<T>::dp>* rk) {
  RkMatrix<T>* result = new RkMatrix<T>(fromDoubleScalarArray<T>(rk->a),
                                        rk->rows,
                                        fromDoubleScalarArray<T>(rk->b),
                                        rk->cols,
                                        rk->method);
  rk->a = NULL; // because rk->a and rk->b have allready been deleted in fromDoubleScalarArray
  rk->b = NULL;
  delete rk;
  return result;
}

template RkMatrix<S_t>* fromDoubleRk(RkMatrix<Types<S_t>::dp>* rk);
template RkMatrix<C_t>* fromDoubleRk(RkMatrix<Types<C_t>::dp>* rk);

}  // end namespace hmat
