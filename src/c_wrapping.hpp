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

#include <string>
#include <cstring>

#include "common/context.hpp"
#include "full_matrix.hpp"
#include "h_matrix.hpp"

namespace
{
template<typename T, template <typename> class E>
hmat_matrix_t * create_empty_hmatrix(void* rows_tree, void* cols_tree)
{
    const hmat::HMatSettings& settings = hmat::HMatSettings::getInstance();
    hmat::SymmetryFlag sym = (settings.useLdlt ? hmat::kLowerSymmetric : hmat::kNotSymmetric);
    return (hmat_matrix_t*) new hmat::HMatInterface<T, E>(
            static_cast<hmat::ClusterTree*>(rows_tree),
            static_cast<hmat::ClusterTree*>(cols_tree),
            sym);
}

template<typename T, template <typename> class E>
hmat_matrix_t * create_empty_hmatrix_admissibility(
  hmat_cluster_tree_t* rows_tree,
  hmat_cluster_tree_t* cols_tree,
  hmat_admissibility_t* condition)
{
    hmat::HMatSettings& settings = hmat::HMatSettings::getInstance();
    settings.setAdmissibilityCondition(static_cast<hmat::AdmissibilityCondition*>((void*)condition));
    hmat::SymmetryFlag sym = (settings.useLdlt ? hmat::kLowerSymmetric : hmat::kNotSymmetric);
    return (hmat_matrix_t*) new hmat::HMatInterface<T, E>(
            static_cast<hmat::ClusterTree*>(static_cast<void*>(rows_tree)),
            static_cast<hmat::ClusterTree*>(static_cast<void*>(cols_tree)),
            sym);
}


template<typename T>
class SimpleCAssemblyFunction : public hmat::SimpleAssemblyFunction<T> {
private:
  simple_interaction_compute_func functor;
  void* functor_extra_args;

  /** Constructor.

      \param _mat The FullMatrix<T> the values are taken from.
   */
public:
  SimpleCAssemblyFunction(void* user_context, simple_interaction_compute_func &f)
    : hmat::SimpleAssemblyFunction<T>(), functor(f), functor_extra_args(user_context) {}

  typename hmat::Types<T>::dp interaction(int i, int j) const {
    typename hmat::Types<T>::dp result;
    (*functor)(functor_extra_args, i, j, &result);
    return result;
  }
};

template<typename T, template <typename> class E>
int assemble(hmat_matrix_t * holder,
                              void* user_context,
                              hmat_prepare_func_t prepare,
                              compute_func compute,
                              int lower_symmetric) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T, E>* hmat = (hmat::HMatInterface<T, E>*) holder;
  hmat::BlockAssemblyFunction<T> f(&hmat->rows->data_, &hmat->cols->data_, user_context, prepare, compute);
  hmat->assemble(f, lower_symmetric ? hmat::kLowerSymmetric : hmat::kNotSymmetric, true);
  std::pair<size_t, size_t> p =  hmat->compressionRatio();
  std::cout << "Compression ratio          = " << (100. * p.first) / p.second << "%" << std::endl;
  p =  hmat->fullrkRatio();
  double total = p.first + p.second;
  std::cout << "Memory balance : (Full,Rk) = (" << (100. * p.first)/total << "%, " 
                                                << (100. * p.second)/total << "%)" << std::endl;
  return 0;
}

template<typename T, template <typename> class E>
int assemble_factor(hmat_matrix_t * holder,
                              void* user_context,
                              hmat_prepare_func_t prepare,
                              compute_func compute,
                              int lower_symmetric) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T, E>* hmat = (hmat::HMatInterface<T, E>*) holder;
  hmat::BlockAssemblyFunction<T> f(&hmat->rows->data_, &hmat->cols->data_, user_context, prepare, compute);
  hmat->assemble(f, lower_symmetric ? hmat::kLowerSymmetric : hmat::kNotSymmetric, false);
  hmat->factorize();
  std::pair<size_t, size_t> p =  hmat->compressionRatio();
  std::cout << "Compression ratio          = " << (100. * p.first) / p.second << "%" << std::endl;
  p =  hmat->fullrkRatio();
  double total = p.first + p.second;
  std::cout << "Memory balance : (Full,Rk) = (" << (100. * p.first)/total << "%, " 
                                                << (100. * p.second)/total << "%)" << std::endl;
  return 0;
}

template<typename T, template <typename> class E>
int assemble_simple_interaction(hmat_matrix_t * holder,
                          void* user_context,
                          simple_interaction_compute_func compute,
                          int lower_symmetric) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T, E>* hmat = (hmat::HMatInterface<T, E>*) holder;
  SimpleCAssemblyFunction<T> f(user_context, compute);
  hmat->assemble(f, lower_symmetric ? hmat::kLowerSymmetric : hmat::kNotSymmetric);
  std::pair<size_t, size_t> p =  hmat->compressionRatio();
  std::cout << "Compression ratio          = " << (100. * p.first) / p.second << "%" << std::endl;
  p =  hmat->fullrkRatio();
  double total = p.first + p.second;
  std::cout << "Memory balance : (Full,Rk) = (" << (100. * p.first)/total << "%, " 
                                                << (100. * p.second)/total << "%)" << std::endl;
  return 0;
}

template<typename T, template <typename> class E>
hmat_matrix_t* copy(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  return (hmat_matrix_t*) ((hmat::HMatInterface<T, E>*) holder)->copy();
}

template<typename T, template <typename> class E>
int destroy(hmat_matrix_t* holder) {
  delete (hmat::HMatInterface<T, E>*)(holder);
  return 0;
}

template<typename T, template <typename> class E>
int factor(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T, E>* hmat = (hmat::HMatInterface<T, E>*) holder;
  hmat->factorize();
  std::pair<size_t, size_t> p =  hmat->compressionRatio();
  std::cout << "Compression ratio          = " << (100. * p.first) / p.second << "%" << std::endl;
  p =  hmat->fullrkRatio();
  double total = p.first + p.second;
  std::cout << "Memory balance : (Full,Rk) = (" << (100. * p.first)/total << "%, " 
                                                << (100. * p.second)/total << "%)" << std::endl;
  return 0;
}

template<typename T, template <typename> class E>
int finalize() {
  hmat::HMatInterface<T, E>::finalize();
  return 0;
}

template<typename T, template <typename> class E>
int full_gemm(char transA, char transB, int mc, int nc, void* c,
                             void* alpha, void* a, hmat_matrix_t * holder, void* beta) {
  DECLARE_CONTEXT;

  const hmat::HMatInterface<T, E>* b = (hmat::HMatInterface<T, E>*)holder;
  hmat::FullMatrix<T> matC((T*)c, mc, nc);
  hmat::FullMatrix<T>* matA = NULL;
  if (transA == 'N') {
    matA = new hmat::FullMatrix<T>((T*)a, mc, transB == 'N' ? b->rows->data_.size()
                             : b->cols->data_.size());
  } else {
    matA = new hmat::FullMatrix<T>((T*)a, transB == 'N' ? b->rows->data_.size()
                             : b->cols->data_.size(), mc);
  }
  hmat::HMatInterface<T, E>::gemm(matC, transA, transB, *((T*)alpha), *matA, *b, *((T*)beta));
  delete matA;
  return 0;
}

template<typename T, template <typename> class E>
int gemm(char trans_a, char trans_b, void *alpha, hmat_matrix_t * holder,
                   hmat_matrix_t * holder_b, void *beta, hmat_matrix_t * holder_c) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T, E>* hmat_a = (hmat::HMatInterface<T, E>*)holder;
  hmat::HMatInterface<T, E>* hmat_b = (hmat::HMatInterface<T, E>*)holder_b;
  hmat::HMatInterface<T, E>* hmat_c = (hmat::HMatInterface<T, E>*)holder_c;
  hmat_c->gemm(trans_a, trans_b, *((T*)alpha), hmat_a, hmat_b, *((T*)beta));
  return 0;
}

template<typename T, template <typename> class E>
int gemv(char trans_a, void* alpha, hmat_matrix_t * holder, void* vec_b,
                   void* beta, void* vec_c, int nrhs) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T, E>* hmat = (hmat::HMatInterface<T, E>*)holder;
  const hmat::ClusterData* bData = (trans_a == 'N' ? &hmat->cols->data_ : &hmat->rows->data_);
  const hmat::ClusterData* cData = (trans_a == 'N' ? &hmat->rows->data_ : &hmat->cols->data_);
  hmat::FullMatrix<T> mb((T*) vec_b, bData->size(), nrhs);
  hmat::FullMatrix<T> mc((T*) vec_c, cData->size(), nrhs);
  hmat->gemv(trans_a, *((T*)alpha), mb, *((T*)beta), mc);
  return 0;
}

template<typename T, template <typename> class E>
int init() {
    return hmat::HMatInterface<T, E>::init();
}

template<typename T, template <typename> class E>
double norm(hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  return ((hmat::HMatInterface<T, E>*)holder)->norm();
}

template<typename T, template <typename> class E>
int scale(void *alpha, hmat_matrix_t* holder) {
  DECLARE_CONTEXT;
  ((hmat::HMatInterface<T, E>*)holder)->scale(*((T*)alpha));
  return 0;
}

template<typename T, template <typename> class E>
int solve_mat(hmat_matrix_t* hmat, hmat_matrix_t* hmatB) {
  ((hmat::HMatInterface<T, E>*)hmat)->solve(*(hmat::HMatInterface<T, E>*)hmatB);
    return 0;
}

template<typename T, template <typename> class E>
int solve_systems(hmat_matrix_t* holder, void* b, int nrhs) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T, E>* hmat = (hmat::HMatInterface<T, E>*)holder;
  hmat::FullMatrix<T> mb((T*) b, hmat->cols->data_.size(), nrhs);
  hmat->solve(mb);
  return 0;
}

template<typename T, template <typename> class E>
int transpose(hmat_matrix_t* hmat) {
  DECLARE_CONTEXT;
  ((hmat::HMatInterface<T, E>*)hmat)->transpose();
  return 0;
}

template<typename T, template <typename> class E>
int hmat_get_info(hmat_matrix_t* holder, hmat_info_t* info) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T, E>* hmat = (hmat::HMatInterface<T, E>*) holder;
  std::pair<size_t, size_t> p = hmat->compressionRatio();
  info->compressed_size       = p.first;
  info->uncompressed_size     = p.second;
  p                           = hmat->fullrkRatio();
  info->full_size             = p.first;
  info->rk_size               = p.second;
  info->nr_block_clusters     = hmat->nodesCount();
  return 0;
}

template<typename T, template <typename> class E>
int hmat_dump_info(hmat_matrix_t* holder, char* prefix) {
  DECLARE_CONTEXT;
  hmat::HMatInterface<T, E>* hmat = (hmat::HMatInterface<T, E>*) holder;
  std::string fileps(prefix);
  fileps += ".ps";
  std::string filejson(prefix);
  filejson += ".json";
  hmat->createPostcriptFile( fileps.c_str());
  hmat->dumpTreeToFile( filejson.c_str() );
  return 0;
}

}  // end anonymous namespace

namespace hmat {

template<typename T, template <typename> class E>
static void createCInterface(hmat_interface_t * i)
{
    i->assemble = assemble<T, E>;
    i->assemble_factor = assemble_factor<T, E>;
    i->assemble_simple_interaction = assemble_simple_interaction<T, E>;
    i->copy = copy<T, E>;
    i->create_empty_hmatrix = create_empty_hmatrix<T, E>;
    i->create_empty_hmatrix_admissibility = create_empty_hmatrix_admissibility<T, E>;
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
    i->hmat_get_info  = hmat_get_info<T, E>;
    i->hmat_dump_info = hmat_dump_info<T, E>;
}

}  // end namespace hmat

#endif  // _C_WRAPPING_HPP
