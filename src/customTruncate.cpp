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

#include "customTruncate.hpp"
#include "rk_matrix.hpp"
#include "h_matrix.hpp"
#include "cluster_tree.hpp"
#include <cstring> // memcpy
#include <cfloat> // DBL_MAX
#include "data_types.hpp"
#include "lapack_operations.hpp"
#include "common/context.hpp"
#include "common/my_assert.h"


namespace hmat {
template<typename T> void mGSTruncate(RkMatrix<T> *Rk, double prec){
  DECLARE_CONTEXT;
  typedef double dp_t;

  int ierr;
  if (Rk->rank() == 0) {
    assert(!(Rk->a || Rk->b));
    return;
  }

  int krank = Rk->rank();

  /* Gram-Schmidt on Rk->a */
  int *permA = new int[krank];
  for(int k=0;k<krank;k++){permA[k] = k;}

  ScalarArray<T> *ra_tmp = new ScalarArray<T>(krank,krank);
  int kA = modifiedGramSchmidt( Rk->a, ra_tmp, permA, prec );

  /* We have to apply permA to ra */
  ScalarArray<T> *ra = new ScalarArray<T>(krank,krank);
  for(int j=0;j<krank;j++){
    for(int i=0;i<krank;i++){
      ra->m[i+permA[j]*krank] = ra_tmp->m[i+j*krank];
    }
  }
  delete ra_tmp;

  /* Gram-Schmidt on Rk->b */
  int *permB = new int[krank];
  for(int k=0;k<krank;k++){permB[k] = k;}
  ScalarArray<T> *rb_tmp = new ScalarArray<T>(krank,krank);
  int kB = modifiedGramSchmidt( Rk->b, rb_tmp, permB, prec );

  /* We have to apply permB to rb */
  ScalarArray<T> *rb = new ScalarArray<T>(krank,krank);
  for(int j=0;j<krank;j++){
    for(int i=0;i<krank;i++){
      rb->m[i+permB[j]*krank] = rb_tmp->m[i+j*krank];
    }
  }
  delete rb_tmp;

  ra->rows = kA; rb->rows = kB;
  /* gemm  because the matrices are not triangular */
  ScalarArray<T> *matR = new ScalarArray<T>(kA,kB);
  matR->gemm('N','T', Constants<T>::pone, ra, rb , Constants<T>::zero);
  ra->rows = krank; rb->rows = krank;
  delete ra; delete rb;

  /* SVD */
  ScalarArray<T>* ur = NULL;
  Vector<dp_t>* sr = NULL;
  ScalarArray<T>* vhr = NULL;
  ierr = truncatedSvd<T>(matR, &ur, &sr, &vhr);
  HMAT_ASSERT(!ierr);
  delete matR;

  /* Remove small singular values */
  int newK = 0;
  for(int i=0; i<std::min(kA, kB); i++){
    if(sr->m[i] > prec*sr->m[0]){
      newK += 1;
    }
    else break;
  }
  assert(newK>0);

  /* Scaling of ur and vhr */
  for(int j=0;j<newK;j++){
    for(int i=0;i<ur->rows;i++){
      ur->m[i+j*ur->lda] *= sqrt(sr->m[j]);
    }
  }

  for(int j=0;j<vhr->cols;j++){
    for(int i=0;i<newK;i++){
      vhr->m[i+j*vhr->lda] *= sqrt(sr->m[i]);
    }
  }

  /* Multiplication by orthogonal matrix Q: no or/un-mqr as
    this comes from Gram-Schmidt procedure not Householder
  */
  ScalarArray<T> *newA = new ScalarArray<T>(Rk->a->rows,newK);
  Rk->a->cols = kA; ur->cols = newK;
  newA->gemm ( 'N', 'N', Constants<T>::pone, Rk->a, ur, Constants<T>::zero);
  Rk->a->cols = krank; ur->cols = newK;

  ScalarArray<T> *newB = new ScalarArray<T>(Rk->b->rows,newK);
  Rk->b->cols = kB; vhr->rows = newK;
  newB->gemm ( 'N', 'T', Constants<T>::pone, Rk->b, vhr, Constants<T>::zero);
  Rk->b->cols = krank; vhr->rows = std::min(kA,kB);

  delete Rk->a;
  Rk->a = newA;
  delete Rk->b;
  Rk->b = newB;

  delete ur;
  delete vhr;
  delete sr;

  if (Rk->rank() == 0) {
    assert(!(Rk->b || Rk->a));
    return;
  }
}

// Declaration of the used templates
template void mGSTruncate(RkMatrix<S_t> *Rk, double prec);
template void mGSTruncate(RkMatrix<D_t> *Rk, double prec);
template void mGSTruncate(RkMatrix<C_t> *Rk, double prec);
template void mGSTruncate(RkMatrix<Z_t> *Rk, double prec);

}// end namespace hmat
