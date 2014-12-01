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

#ifndef _C_WRAPPING_HPP
#define _C_WRAPPING_HPP

#include "common/context.hpp"
#include "full_matrix.hpp"

template<typename T, template <typename> class E> static hmat_matrix_t * create_empty_hmatrix(
        void* rows_tree, void* cols_tree)
{
    return (hmat_matrix_t*) new HMatInterface<T, E>(
            static_cast<ClusterTree*>(rows_tree),
            static_cast<ClusterTree*>(cols_tree));
}

template<typename T>
class SimpleCAssemblyFunction : public SimpleAssemblyFunction<T> {
private:
  simple_interaction_compute_func functor;
  void* functor_extra_args;

  /** Constructor.

      \param _mat The FullMatrix<T> the values are taken from.
   */
public:
  SimpleCAssemblyFunction(void* user_context, simple_interaction_compute_func &f)
    : SimpleAssemblyFunction<T>(), functor(f), functor_extra_args(user_context) {}

  typename Types<T>::dp interaction(int i, int j) const {
    typename Types<T>::dp result;
    (*functor)(functor_extra_args, i, j, &result);
    return result;
  }
};

template<typename T, template <typename> class E>
static int assemble(hmat_matrix_t * holder,
                              void* user_context,
                              prepare_func prepare,
                              compute_func compute,
                              release_func free_data,
                              int lower_symmetric) {
  DECLARE_CONTEXT;
  HMatInterface<T, E>* hmat = (HMatInterface<T, E>*) holder;
  BlockAssemblyFunction<T> f(&hmat->rows->data, &hmat->cols->data, user_context, prepare, compute, free_data);
  hmat->assemble(f, lower_symmetric ? kLowerSymmetric : kNotSymmetric, true);
  std::pair<size_t, size_t> p =  hmat->compressionRatio();
  std::cout << "Compression ratio = " << (100. * p.first) / p.second << "%" << std::endl;
  return 0;
}

template<typename T, template <typename> class E>
static int assemble_factor(hmat_matrix_t * holder,
                              void* user_context,
                              prepare_func prepare,
                              compute_func compute,
                              release_func free_data,
                              int lower_symmetric) {
  DECLARE_CONTEXT;
  HMatInterface<T, E>* hmat = (HMatInterface<T, E>*) holder;
  BlockAssemblyFunction<T> f(&hmat->rows->data, &hmat->cols->data, user_context, prepare, compute, free_data);
  hmat->assemble(f, lower_symmetric ? kLowerSymmetric : kNotSymmetric, false);
  hmat->factorize();
  std::pair<size_t, size_t> p =  hmat->compressionRatio();
  std::cout << "Compression ratio = " << (100. * p.first) / p.second << "%" << std::endl;
  return 0;
}

template<typename T, template <typename> class E>
static int assemble_simple_interaction(hmat_matrix_t * holder,
                          void* user_context,
                          simple_interaction_compute_func compute,
                          int lower_symmetric) {
  DECLARE_CONTEXT;
  HMatInterface<T, E>* hmat = (HMatInterface<T, E>*) holder;
  SimpleCAssemblyFunction<T> f(user_context, compute);
  hmat->assemble(f, lower_symmetric ? kLowerSymmetric : kNotSymmetric);
  std::pair<size_t, size_t> p =  hmat->compressionRatio();
  std::cout << "Compression ratio = " << (100. * p.first) / p.second << "%" << std::endl;
  return 0;
}

template<typename T, template <typename> class E>
static hmat_matrix_t* copy(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  return (hmat_matrix_t*) ((HMatInterface<T, E>*) holder)->copy();
}

template<typename T, template <typename> class E> static int destroy(hmat_matrix_t* holder) {
  delete (HMatInterface<T, E>*)(holder);
  return 0;
}

template<typename T, template <typename> class E> static int factor(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  HMatInterface<T, E>* hmat = (HMatInterface<T, E>*) holder;
  hmat->factorize();
  std::pair<size_t, size_t> p =  hmat->compressionRatio();
  std::cout << "Compression ratio = " << (100. * p.first) / p.second << "%" << std::endl;
  return 0;
}

template<typename T, template <typename> class E> static int finalize() {
  HMatInterface<T, E>::finalize();
  return 0;
}

template<typename T, template <typename> class E>
static int full_gemm(char transA, char transB, int mc, int nc, void* c,
                             void* alpha, void* a, hmat_matrix_t * holder, void* beta) {
  DECLARE_CONTEXT;

  const HMatInterface<T, E>* b = (HMatInterface<T, E>*)holder;
  FullMatrix<T> matC((T*)c, mc, nc);
  FullMatrix<T>* matA = NULL;
  if (transA == 'N') {
    matA = new FullMatrix<T>((T*)a, mc, transB == 'N' ? b->rows->data.n
                             : b->cols->data.n);
  } else {
    matA = new FullMatrix<T>((T*)a, transB == 'N' ? b->rows->data.n
                             : b->cols->data.n, mc);
  }
  HMatInterface<T, E>::gemm(matC, transA, transB, *((T*)alpha), *matA, *b, *((T*)beta));
  delete matA;
  return 0;
}

template<typename T, template <typename> class E>
static int gemm(char trans_a, char trans_b, void *alpha, hmat_matrix_t * holder,
                   hmat_matrix_t * holder_b, void *beta, hmat_matrix_t * holder_c) {
  DECLARE_CONTEXT;
  HMatInterface<T, E>* hmat_a = (HMatInterface<T, E>*)holder;
  HMatInterface<T, E>* hmat_b = (HMatInterface<T, E>*)holder_b;
  HMatInterface<T, E>* hmat_c = (HMatInterface<T, E>*)holder_c;
  hmat_c->gemm(trans_a, trans_b, *((T*)alpha), hmat_a, hmat_b, *((T*)beta));
  return 0;
}

template<typename T, template <typename> class E>
static int gemv(char trans_a, void* alpha, hmat_matrix_t * holder, void* vec_b,
                   void* beta, void* vec_c, int nrhs) {
  DECLARE_CONTEXT;
  HMatInterface<T, E>* hmat = (HMatInterface<T, E>*)holder;
  const ClusterData* bData = (trans_a == 'N' ? &hmat->cols->data : &hmat->rows->data);
  const ClusterData* cData = (trans_a == 'N' ? &hmat->rows->data : &hmat->cols->data);
  FullMatrix<T> mb((T*) vec_b, bData->n, nrhs);
  FullMatrix<T> mc((T*) vec_c, cData->n, nrhs);
  hmat->gemv(trans_a, *((T*)alpha), mb, *((T*)beta), mc);
  return 0;
}

template<typename T, template <typename> class E>
static int init() {
    return HMatInterface<T, E>::init();
}

template<typename T, template <typename> class E> static double norm(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  return ((HMatInterface<T, E>*)holder)->norm();
}

template<typename T, template <typename> class E> static int scale(void *alpha, hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  ((HMatInterface<T, E>*)holder)->scale(*((T*)alpha));
  return 0;
}

template<typename T, template <typename> class E>
static int solve_mat(hmat_matrix_t* hmat, hmat_matrix_t* hmatB) {
  ((HMatInterface<T, E>*)hmat)->solve(*(HMatInterface<T, E>*)hmatB);
    return 0;
}

template<typename T, template <typename> class E>
static int solve_systems(hmat_matrix_t* holder, void* b, int nrhs) {
  DECLARE_CONTEXT;
  HMatInterface<T, E>* hmat = (HMatInterface<T, E>*)holder;
  FullMatrix<T> mb((T*) b, hmat->cols->data.n, nrhs);
  hmat->solve(mb);
  return 0;
}

template<typename T, template <typename> class E>
static int transpose(hmat_matrix_t* hmat) {
  DECLARE_CONTEXT;
  ((HMatInterface<T, E>*)hmat)->transpose();
  return 0;
}

template<typename T, template <typename> class E> static void createCInterface(hmat_interface_t * i)
{
    i->assemble = assemble<T, E>;
    i->assemble_factor = assemble_factor<T, E>;
    i->assemble_simple_interaction = assemble_simple_interaction<T, E>;
    i->copy = copy<T, E>;
    i->create_empty_hmatrix = create_empty_hmatrix<T, E>;
    i->destroy = destroy<T, E>;
    i->factor = factor<T, E>;
    i->finalize = finalize<T, E>;
    i->full_gemm = full_gemm<T, E>;
    i->gemm = gemm<T, E>;
    i->gemv = gemv<T, E>;
    i->init = init<T, E>;
    i->norm = norm<T, E>;
    i->scale = scale<T, E>;
    i->solve_mat = solve_mat<T, E>;
    i->solve_systems = solve_systems<T, E>;
    i->transpose = transpose<T, E>;
    i->internal = NULL;
}

#endif  // _C_WRAPPING_HPP
