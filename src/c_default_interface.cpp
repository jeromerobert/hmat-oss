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

#include "hmat/hmat.h"
#include "coordinates.hpp"
#include "hmat_cpp_interface.hpp"
#include "default_engine.hpp"
#include "clustering.hpp"
#include "admissibility.hpp"
#include "c_wrapping.hpp"
#include "common/my_assert.h"

using namespace hmat;

hmat_clustering_algorithm_t * hmat_create_clustering_median()
{
    return (hmat_clustering_algorithm_t*) new MedianBisectionAlgorithm();
}

hmat_clustering_algorithm_t * hmat_create_clustering_geometric()
{
    return (hmat_clustering_algorithm_t*) new GeometricBisectionAlgorithm();
}

hmat_clustering_algorithm_t * hmat_create_clustering_hybrid()
{
    return (hmat_clustering_algorithm_t*) new HybridBisectionAlgorithm();
}

void hmat_delete_clustering(hmat_clustering_algorithm_t* algo)
{
    delete (ClusteringAlgorithm*) algo;
}

void hmat_set_clustering_divider(hmat_clustering_algorithm_t* algo, int divider)
{
    ((ClusteringAlgorithm*) algo)->setDivider(divider);
}

hmat_clustering_algorithm_t*
hmat_create_clustering_max_dof(const hmat_clustering_algorithm_t* algo, int max_dof)
{
  ClusteringAlgorithm* result = static_cast<const ClusteringAlgorithm*>((void*) algo)->clone();
  result->setMaxLeafSize(max_dof);
  return static_cast<hmat_clustering_algorithm_t*>((void*) result);
}

hmat_clustering_algorithm_t*
hmat_create_void_clustering(const hmat_clustering_algorithm_t* algo)
{
  VoidClusteringAlgorithm* result = new VoidClusteringAlgorithm(*(static_cast<const ClusteringAlgorithm*>((void*) algo)));
  return static_cast<hmat_clustering_algorithm_t*>((void*) result);
}

hmat_cluster_tree_t * hmat_create_cluster_tree(double* coord, int dimension, int size, hmat_clustering_algorithm_t* algo)
{
    DofCoordinates dofs(coord, dimension, size, true);
    return (hmat_cluster_tree_t*) createClusterTree(dofs, *((ClusteringAlgorithm*) algo));
}

hmat_cluster_tree_builder_t* hmat_create_cluster_tree_builder(const hmat_clustering_algorithm_t* algo)
{
    ClusterTreeBuilder* result = new ClusterTreeBuilder(*static_cast<const ClusteringAlgorithm*>((void*) algo));
    return static_cast<hmat_cluster_tree_builder_t*>((void*) result);
}

/* Specify an algorithm for nodes at given depth and below */
void hmat_cluster_tree_builder_add_algorithm(hmat_cluster_tree_builder_t* ctb, int level, const hmat_clustering_algorithm_t* algo)
{
    ClusterTreeBuilder* ct_builder = static_cast<ClusterTreeBuilder*>((void*) ctb);
    ct_builder->addAlgorithm(level, *static_cast<const ClusteringAlgorithm*>((void*) algo));
}

void hmat_delete_cluster_tree_builder(hmat_cluster_tree_builder_t* ctb)
{
    delete static_cast<ClusterTreeBuilder*>((void*)ctb);
}

/* Create a ClusterTree from the DoFs coordinates. */
hmat_cluster_tree_t * hmat_create_cluster_tree_from_builder(double* coord, int dimension, int size, const hmat_cluster_tree_builder_t* ctb)
{
    const ClusterTreeBuilder* ct_builder = static_cast<const ClusterTreeBuilder*>((void*) ctb);
    DofCoordinates dofs(coord, dimension, size, true);
    return static_cast<hmat_cluster_tree_t*>((void*)  ct_builder->build(dofs));
}

void hmat_delete_cluster_tree(hmat_cluster_tree_t * tree) {
    delete ((ClusterTree*)tree);
}

hmat_cluster_tree_t * hmat_copy_cluster_tree(hmat_cluster_tree_t * tree) {
    return (hmat_cluster_tree_t*)((ClusterTree*)tree)->copy();
}

int hmat_tree_nodes_count(hmat_cluster_tree_t * tree)
{
    return ((ClusterTree*)tree)->nodesCount();
}

int hmat_cluster_get_info(hmat_cluster_tree_t *tree, hmat_cluster_info_t* info)
{
    ClusterTree* cl          = static_cast<ClusterTree*>((void*) tree);
    info->spatial_dimension  = cl->data.coordinates()->dimension();
    info->dimension          = cl->data.coordinates()->size();
    info->nr_tree_nodes      = cl->nodesCount();
    return 0;
}

void hmat_cluster_tree_same_depth(const hmat_cluster_tree_t *reference, hmat_cluster_tree_t *to_modify) {
    ClusterTree* toModify = reinterpret_cast<ClusterTree*>(to_modify);
    const ClusterTree* ref = reinterpret_cast<const ClusterTree*>(reference);
    toModify->sameDepth(ref);
}

void hmat_init_admissibility_param(hmat_admissibility_param_t * p) {
    p->eta = 2;
    p->max_svd_elements = 5000000;
    p->max_aca_elements = 0;
    p->always = 0;
    p->separator_force_compression = -1;
}

hmat_admissibility_t* hmat_create_admissibility(hmat_admissibility_param_t * p) {
    hmat::StandardAdmissibilityCondition * r = new hmat::StandardAdmissibilityCondition(
         p->eta, p->max_svd_elements, p->max_aca_elements);
    if(p->always)
      r->setAlways(true);
    return reinterpret_cast<hmat_admissibility_t*>(r);
}

hmat_admissibility_t* hmat_create_admissibility_standard(double eta)
{
    return static_cast<hmat_admissibility_t*>((void*) new hmat::StandardAdmissibilityCondition(eta));
}

void hmat_delete_admissibility(hmat_admissibility_t * cond) {
    delete static_cast<AdmissibilityCondition*>((void*)cond);
}

void hmat_init_default_interface(hmat_interface_t * i, hmat_value_t type)
{
    i->value_type = type;
    switch (type) {
    case HMAT_SIMPLE_PRECISION: createCInterface<S_t, DefaultEngine>(i); break;
    case HMAT_DOUBLE_PRECISION: createCInterface<D_t, DefaultEngine>(i); break;
    case HMAT_SIMPLE_COMPLEX: createCInterface<C_t, DefaultEngine>(i); break;
    case HMAT_DOUBLE_COMPLEX: createCInterface<Z_t, DefaultEngine>(i); break;
    default: HMAT_ASSERT(false);
    }
}

void hmat_get_parameters(hmat_settings_t* settings)
{
    HMatSettings& settingsCxx = HMatSettings::getInstance();
    settings->assemblyEpsilon = settingsCxx.assemblyEpsilon;
    settings->recompressionEpsilon = settingsCxx.recompressionEpsilon;
    switch (settingsCxx.compressionMethod) {
    case Svd:
      settings->compressionMethod = hmat_compress_svd;
      break;
    case AcaFull:
      settings->compressionMethod = hmat_compress_aca_full;
      break;
    case AcaPartial:
      settings->compressionMethod = hmat_compress_aca_partial;
      break;
    case AcaPlus:
      settings->compressionMethod = hmat_compress_aca_plus;
      break;
    default:
      std::cerr << "Internal error: invalid value for compression method: \"" << settingsCxx.compressionMethod << "\"." << std::endl;
      std::cerr << "Internal error: using SVD" << std::endl;
      settings->compressionMethod = hmat_compress_svd;
      break;
    }
    settings->compressionMinLeafSize = settingsCxx.compressionMinLeafSize;
    settings->maxLeafSize = settingsCxx.maxLeafSize;
    settings->maxParallelLeaves = settingsCxx.maxParallelLeaves;
    settings->coarsening = settingsCxx.coarsening;
    settings->recompress = settingsCxx.recompress;
    settings->validateCompression = settingsCxx.validateCompression;
    settings->validationErrorThreshold = settingsCxx.validationErrorThreshold;
    settings->validationReRun = settingsCxx.validationReRun;
    settings->validationDump = settingsCxx.validationDump;
}

int hmat_set_parameters(hmat_settings_t* settings)
{
    HMAT_ASSERT(settings != NULL);
    int rc = 0;
    HMatSettings& settingsCxx = HMatSettings::getInstance();
    settingsCxx.assemblyEpsilon = settings->assemblyEpsilon;
    settingsCxx.recompressionEpsilon = settings->recompressionEpsilon;
    switch (settings->compressionMethod) {
    case hmat_compress_svd:
      settingsCxx.compressionMethod = Svd;
      break;
    case hmat_compress_aca_full:
      settingsCxx.compressionMethod = AcaFull;
      break;
    case hmat_compress_aca_partial:
      settingsCxx.compressionMethod = AcaPartial;
      break;
    case hmat_compress_aca_plus:
      settingsCxx.compressionMethod = AcaPlus;
      break;
    default:
      std::cerr << "Invalid value for compression method: \"" << settings->compressionMethod << "\"." << std::endl;
      rc = 1;
      break;
    }
    settingsCxx.compressionMinLeafSize = settings->compressionMinLeafSize;
    settingsCxx.maxLeafSize = settings->maxLeafSize;
    settingsCxx.maxParallelLeaves = settings->maxParallelLeaves;
    settingsCxx.coarsening = settings->coarsening;
    settingsCxx.recompress = settings->recompress;
    settingsCxx.validateCompression = settings->validateCompression;
    settingsCxx.validationErrorThreshold = settings->validationErrorThreshold;
    settingsCxx.validationReRun = settings->validationReRun;
    settingsCxx.validationDump = settings->validationDump;
    settingsCxx.setParameters();
    return rc;
}

void hmat_print_parameters(hmat_settings_t* settings)
{
    hmat_set_parameters(settings);
    HMatSettings& settingsCxx = HMatSettings::getInstance();
    settingsCxx.printSettings();
}

const char * hmat_get_version()
{
    return HMAT_VERSION;
}

const char * hmat_get_build_date()
{
  return "Built on " __DATE__ " at " __TIME__ "" ;
}

void hmat_assemble_context_init(hmat_assemble_context_t * context) {
    context->assembly = NULL;
    context->block_compute = NULL;
    context->factorization = hmat_factorization_none;
    context->lower_symmetric = 0;
    context->prepare = NULL;
    context->simple_compute = NULL;
    context->user_context = NULL;
    context->progress = DefaultProgress::getInstance();
}

void hmat_factorization_context_init(hmat_factorization_context_t *context) {
    context->factorization = hmat_factorization_lu;
    context->progress = DefaultProgress::getInstance();
}

void hmat_delete_procedure(hmat_procedure_t* proc) {
    switch (proc->value_type) {
    case HMAT_SIMPLE_PRECISION: delete static_cast<hmat::TreeProcedure<HMatrix<S_t> >*>(proc->internal); break;
    case HMAT_DOUBLE_PRECISION: delete static_cast<hmat::TreeProcedure<HMatrix<D_t> >*>(proc->internal); break;
    case HMAT_SIMPLE_COMPLEX:   delete static_cast<hmat::TreeProcedure<HMatrix<C_t> >*>(proc->internal); break;
    case HMAT_DOUBLE_COMPLEX:   delete static_cast<hmat::TreeProcedure<HMatrix<Z_t> >*>(proc->internal); break;
    default: HMAT_ASSERT(false);
    }
    delete proc;
}

hmat_procedure_t* hmat_create_procedure_epsilon_truncate(hmat_value_t type, double epsilon) {
    hmat_procedure_t * result = new hmat_procedure_t();
    result->value_type = type;
    switch (type) {
    case HMAT_SIMPLE_PRECISION: result->internal = new hmat::EpsilonTruncate<S_t>(epsilon); break;
    case HMAT_DOUBLE_PRECISION: result->internal = new hmat::EpsilonTruncate<D_t>(epsilon); break;
    case HMAT_SIMPLE_COMPLEX:   result->internal = new hmat::EpsilonTruncate<C_t>(epsilon); break;
    case HMAT_DOUBLE_COMPLEX:   result->internal = new hmat::EpsilonTruncate<Z_t>(epsilon); break;
    default: HMAT_ASSERT(false);
    }
    return result;
}

void hmat_tracing_dump(char *filename) {
  tracing_dump(filename);
}

