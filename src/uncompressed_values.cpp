/*
  HMat-OSS (HMatrix library, open source software)

  Copyright (C) 2014-2016 Airbus Group SAS

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

#include "uncompressed_values.hpp"
#include "blas_overloads.hpp"
#include "rk_matrix.hpp"

namespace hmat {

template <typename T> void UncompressedValues<T>::getRkValues() {
    const HMatrix<T> & m = *this->matrix_;
    int rank = m.rank();
    T * a = m.rk()->a->data.m - m.rows()->offset();
    int lda = m.rk()->a->data.lda;
    T * b = m.rk()->b->data.m - m.cols()->offset();
    int ldb = m.rk()->b->data.lda;
    for(IndiceIt r = this->rowStart_; r != this->rowEnd_; ++r) {
        for(IndiceIt c = this->colStart_; c != this->colEnd_; ++c) {
            getValue(r, c, proxy_cblas::dot(rank, a + r->first, lda,
                                            b + c->first, ldb));
        }
    }
}
template void UncompressedValues<S_t>::getRkValues();
template void UncompressedValues<D_t>::getRkValues();
template void UncompressedValues<C_t>::getRkValues();
template void UncompressedValues<Z_t>::getRkValues();
}
