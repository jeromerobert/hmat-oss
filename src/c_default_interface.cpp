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

#include "config.h"
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

hmat_clustering_algorithm_t * hmat_create_clustering_ntilesrecursive(int nTiles)
{
    return (hmat_clustering_algorithm_t*) new NTilesRecursiveAlgorithm( nTiles );
}

hmat_clustering_algorithm_t * hmat_create_clustering_geometric()
{
  bool x0 = getenv("HMAT_GEOM_BISECT_X0") != nullptr;
  return reinterpret_cast<hmat_clustering_algorithm_t*>(new GeometricBisectionAlgorithm(x0));
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

hmat_clustering_algorithm_t* hmat_create_clustering_span(
    const hmat_clustering_algorithm_t* algo, double ratio) {
    SpanClusteringAlgorithm* result = new SpanClusteringAlgorithm(
        *(reinterpret_cast<const ClusteringAlgorithm*>(algo)), ratio);
    return reinterpret_cast<hmat_clustering_algorithm_t*>(result);
}

hmat_clustering_algorithm_t*
hmat_create_void_clustering(const hmat_clustering_algorithm_t* algo)
{
  VoidClusteringAlgorithm* result = new VoidClusteringAlgorithm(*(static_cast<const ClusteringAlgorithm*>((void*) algo)));
  return static_cast<hmat_clustering_algorithm_t*>((void*) result);
}

hmat_clustering_algorithm_t*
hmat_create_shuffle_clustering(const hmat_clustering_algorithm_t* algo, int from_divider, int to_divider)
{
  ShuffleClusteringAlgorithm* result = new ShuffleClusteringAlgorithm(*(static_cast<const ClusteringAlgorithm*>((void*) algo)), from_divider, to_divider);
  return static_cast<hmat_clustering_algorithm_t*>((void*) result);
}

hmat_cluster_tree_t * hmat_create_cluster_tree(double* coord, int dimension, int size, hmat_clustering_algorithm_t* algo)
{
    struct hmat_cluster_tree_create_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.coordinates = coord;
    ctx.dimension = dimension;
    ctx.number_of_points = size;
    ctx.number_of_dof = ctx.number_of_points;
    ctx.span_offsets = NULL;
    ctx.spans = NULL;
    ctx.group_index = NULL;
    ClusterTreeBuilder builder(*reinterpret_cast<ClusteringAlgorithm*>(algo));
    ctx.builder = reinterpret_cast<hmat_cluster_tree_builder_t*>(&builder);
    return hmat_create_cluster_tree_generic(&ctx);
}

hmat_cluster_tree_t * hmat_create_cluster_tree_generic(struct hmat_cluster_tree_create_context_t * ctx) {
    DofCoordinates dofs(ctx->coordinates, ctx->dimension, ctx->number_of_points, true,
                        ctx->number_of_dof, ctx->span_offsets, ctx->spans);
    ClusterTree * r = reinterpret_cast<const ClusterTreeBuilder*>(ctx->builder)->build(dofs, ctx->group_index);
    return reinterpret_cast<hmat_cluster_tree_t *>(r);
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
    struct hmat_cluster_tree_create_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.coordinates = coord;
    ctx.dimension = dimension;
    ctx.number_of_points = size;
    ctx.number_of_dof = ctx.number_of_points;
    ctx.span_offsets = NULL;
    ctx.spans = NULL;
    ctx.group_index = NULL;
    ctx.builder = ctb;
    return hmat_create_cluster_tree_generic(&ctx);
}

void hmat_delete_cluster_tree(const hmat_cluster_tree_t * tree) {
    delete ((ClusterTree*)tree);
}

hmat_cluster_tree_t * hmat_copy_cluster_tree(const hmat_cluster_tree_t * tree) {
    return reinterpret_cast<hmat_cluster_tree_t *>(
        reinterpret_cast<const ClusterTree*>(tree)->copy());
}

void hmat_swap_cluster_tree(hmat_cluster_tree_t *first, hmat_cluster_tree_t *second) {
  reinterpret_cast<ClusterTree*>(first)->swap(reinterpret_cast<ClusterTree*>(second));
}

int hmat_tree_nodes_count(const hmat_cluster_tree_t * tree)
{
    return ((ClusterTree*)tree)->nodesCount();
}

hmat_cluster_tree_t *hmat_cluster_get_son( hmat_cluster_tree_t * tree, int index )
{
    ClusterTree *son = reinterpret_cast<ClusterTree*>(tree)->getChild(index);
    return (hmat_cluster_tree_t*)son;
}

int hmat_cluster_get_info(hmat_cluster_tree_t *tree, hmat_cluster_info_t* info)
{
    ClusterTree* cl          = static_cast<ClusterTree*>((void*) tree);
    info->spatial_dimension  = cl->data.coordinates()->dimension();
    info->dimension          = cl->data.coordinates()->numberOfDof();
    info->nr_tree_nodes      = cl->nodesCount();
    return 0;
}

const int * hmat_cluster_get_indices(const hmat_cluster_tree_t *tree) {
    return reinterpret_cast<const ClusterTree*>(tree)->data.indices();
}

void hmat_init_admissibility_param(hmat_admissibility_param_t * p) {
    p->eta = 2;
    p->ratio = 0.0;
    p->max_width = (size_t)-1L;
    p->scale = -1;
}

hmat_admissibility_t* hmat_create_admissibility(hmat_admissibility_param_t * p) {
    hmat::StandardAdmissibilityCondition * r =
         new hmat::StandardAdmissibilityCondition(p->eta, p->ratio);
    r->setMaxWidth(p->max_width);
    return reinterpret_cast<hmat_admissibility_t*>(r);
}

void hmat_update_admissibility(hmat_admissibility_t* cond, hmat_admissibility_param_t *p) {
    hmat::AdmissibilityCondition * r = reinterpret_cast<AdmissibilityCondition*>(cond);
    r->setRatio(p->ratio);
    r->setMaxWidth(p->max_width);
}

hmat_admissibility_t* hmat_create_admissibility_standard(double eta)
{
    return static_cast<hmat_admissibility_t*>((void*) new hmat::StandardAdmissibilityCondition(eta));
}

hmat_admissibility_t* hmat_create_admissibility_hodlr() {
  return reinterpret_cast<hmat_admissibility_t*>(new hmat::HODLRAdmissibilityCondition());
}

hmat_admissibility_t* hmat_create_admissibility_always(
        size_t max_size, unsigned int min_block, int split_rows, int split_cols) {
    return reinterpret_cast<hmat_admissibility_t*>(
        new hmat::AlwaysAdmissibilityCondition(max_size, min_block, split_rows, split_cols));
}

hmat_admissibility_t* hmat_create_admissibility_never(
        size_t max_size, unsigned int min_block, int split_rows, int split_cols) {
    hmat::AlwaysAdmissibilityCondition * r = new hmat::AlwaysAdmissibilityCondition(
        max_size, min_block, split_rows, split_cols);
    r->never(true);
    return reinterpret_cast<hmat_admissibility_t*>(r);
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
    settings->compressionMinLeafSize = settingsCxx.compressionMinLeafSize;
    settings->coarseningEpsilon = settingsCxx.coarseningEpsilon;
    settings->maxLeafSize = settingsCxx.maxLeafSize;
    settings->coarsening = settingsCxx.coarsening;
    settings->validateNullRowCol = settingsCxx.validateNullRowCol;
    settings->validateCompression = settingsCxx.validateCompression;
    settings->validateRecompression = settingsCxx.validateRecompression;
    settings->validationErrorThreshold = settingsCxx.validationErrorThreshold;
    settings->validationReRun = settingsCxx.validationReRun;
    settings->dumpTrace = settingsCxx.dumpTrace;
    settings->validationDump = settingsCxx.validationDump;
}

int hmat_set_parameters(hmat_settings_t* settings)
{
    HMAT_ASSERT(settings != NULL);
    int rc = 0;
    HMatSettings& settingsCxx = HMatSettings::getInstance();
    settingsCxx.compressionMinLeafSize = settings->compressionMinLeafSize;
    settingsCxx.coarseningEpsilon = settings->coarseningEpsilon;
    settingsCxx.maxLeafSize = settings->maxLeafSize;
    settingsCxx.coarsening = settings->coarsening;
    settingsCxx.validateNullRowCol = settings->validateNullRowCol;
    settingsCxx.validateCompression = settings->validateCompression;
    settingsCxx.validateRecompression=settings->validateRecompression;
    settingsCxx.validationErrorThreshold = settings->validationErrorThreshold;
    settingsCxx.validationReRun = settings->validationReRun;
    settingsCxx.dumpTrace = settings->dumpTrace;
    settingsCxx.validationDump = settings->validationDump;
    settingsCxx.setParameters();
    return rc;
}

const char * hmat_get_version()
{
    return HMAT_VERSION;
}

void hmat_get_build_date(const char **date, const char **time)
{
#ifdef HMAT_EXPORT_BUILD_DATE
  *date=(const char*)&__DATE__;
  *time=(const char*)&__TIME__;
#else
  *date="N/A";
  *time="N/A";
#endif
}

void hmat_assemble_context_init(hmat_assemble_context_t * context) {
    context->compression = NULL;
    context->assembly = NULL;
    context->simple_compute = NULL;
    context->block_compute = NULL;
    context->advanced_compute = NULL;
    context->user_context = NULL;
    context->prepare = NULL;
    context->lower_symmetric = 0;
    context->factorization = hmat_factorization_none;
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

void hmat_delete_leaf_procedure(hmat_leaf_procedure_t* proc) {
    switch (proc->value_type) {
    case HMAT_SIMPLE_PRECISION: delete static_cast<const hmat::LeafProcedure<HMatrix<S_t> >*>(proc->internal); break;
    case HMAT_DOUBLE_PRECISION: delete static_cast<const hmat::LeafProcedure<HMatrix<D_t> >*>(proc->internal); break;
    case HMAT_SIMPLE_COMPLEX:   delete static_cast<const hmat::LeafProcedure<HMatrix<C_t> >*>(proc->internal); break;
    case HMAT_DOUBLE_COMPLEX:   delete static_cast<const hmat::LeafProcedure<HMatrix<Z_t> >*>(proc->internal); break;
    default: HMAT_ASSERT(false);
    }
    delete proc;
}

hmat_compression_algorithm_t* hmat_create_compression_svd(double epsilon) {
    return static_cast<hmat_compression_algorithm_t*>((void*) new hmat::CompressionSVD(epsilon));
}

hmat_compression_algorithm_t* hmat_create_compression_aca_full(double epsilon) {
    return static_cast<hmat_compression_algorithm_t*>((void*) new hmat::CompressionAcaFull(epsilon));
}

hmat_compression_algorithm_t* hmat_create_compression_aca_partial(double epsilon) {
    return static_cast<hmat_compression_algorithm_t*>((void*) new hmat::CompressionAcaPartial(epsilon));
}

hmat_compression_algorithm_t* hmat_create_compression_aca_plus(double epsilon) {
    return static_cast<hmat_compression_algorithm_t*>((void*) new hmat::CompressionAcaPlus(epsilon));
}

hmat_compression_algorithm_t* hmat_create_compression_aca_random(double epsilon) {
    return static_cast<hmat_compression_algorithm_t*>((void*) new hmat::CompressionAcaRandom(epsilon));
}
hmat_compression_algorithm_t* hmat_create_compression_rrqr(double epsilon) {
    return reinterpret_cast<hmat_compression_algorithm_t*>(new hmat::CompressionRRQR(epsilon));
}

void hmat_delete_compression(const hmat_compression_algorithm_t* algo) {
    delete static_cast<hmat::CompressionAlgorithm*>((void*)algo);
}

void hmat_tracing_dump(char *filename) {
  tracing_dump(filename);
}

hmat_progress_t * hmat_default_progress() {
    return DefaultProgress::getInstance();
}

void hmat_set_worker_index_function(int (*f)()) {
  tracing_set_worker_index_func(f);
}
