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
#include "hmat_cpp_interface.hpp"
#include "default_engine.hpp"
#include "clustering.hpp"
#include "admissibility.hpp"
#include "c_wrapping.hpp"
#include "common/my_assert.h"

using namespace hmat;

hmat_coordinates_t * hmat_create_coordinates(double* coord, int dim, int size)
{
    return (hmat_coordinates_t*) createCoordinates(coord, dim, size);
}

void hmat_delete_coordinates(hmat_coordinates_t* dls)
{
    delete (DofCoordinates*) dls;
}

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

hmat_cluster_tree_t * hmat_create_cluster_tree(hmat_coordinates_t* dls, hmat_clustering_algorithm_t* algo) {
    return (hmat_cluster_tree_t*) createClusterTree(*((DofCoordinates*) dls), *((ClusteringAlgorithm*) algo));
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

hmat_admissibility_t* hmat_create_admissibility_standard(double eta)
{
    return static_cast<hmat_admissibility_t*>((void*) new hmat::StandardAdmissibilityCondition(eta));
}

void hmat_delete_admissibility(hmat_admissibility_t * cond) {
    delete static_cast<AdmissibilityCondition*>((void*)cond);
}

void hmat_init_default_interface(hmat_interface_t * i, hmat_value_t type)
{
    switch (type) {
    case HMAT_SIMPLE_PRECISION: createCInterface<S_t, DefaultEngine>(i); break;
    case HMAT_DOUBLE_PRECISION: createCInterface<D_t, DefaultEngine>(i); break;
    case HMAT_SIMPLE_COMPLEX: createCInterface<C_t, DefaultEngine>(i); break;
    case HMAT_DOUBLE_COMPLEX: createCInterface<Z_t, DefaultEngine>(i); break;
    default: strongAssert(false);
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
    settings->admissibilityCondition = static_cast<hmat_admissibility_t*>((void*)settingsCxx.admissibilityCondition);
    settings->admissibilityFactor = 0.0;
    settings->clustering = static_cast<hmat_clustering_algorithm_t*>((void*) settingsCxx.clustering);
    settings->compressionMinLeafSize = settingsCxx.compressionMinLeafSize;
    settings->maxLeafSize = settingsCxx.maxLeafSize;
    settings->maxParallelLeaves = settingsCxx.maxParallelLeaves;
    settings->elementsPerBlock = settingsCxx.elementsPerBlock;
    settings->useLu = settingsCxx.useLu;
    settings->useLdlt = settingsCxx.useLdlt;
    settings->coarsening = settingsCxx.coarsening;
    settings->recompress = settingsCxx.recompress;
    settings->validateCompression = settingsCxx.validateCompression;
    settings->validationErrorThreshold = settingsCxx.validationErrorThreshold;
    settings->validationReRun = settingsCxx.validationReRun;
    settings->validationDump = settingsCxx.validationDump;
}

int hmat_set_parameters(hmat_settings_t* settings)
{
    strongAssert(settings != NULL);
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
    if (settings->admissibilityFactor != 0.0)
    {
      StandardAdmissibilityCondition::DEPRECATED_INSTANCE.setEta(settings->admissibilityFactor);
      settingsCxx.admissibilityCondition = &StandardAdmissibilityCondition::DEPRECATED_INSTANCE;
    }
    else
      settingsCxx.admissibilityCondition = static_cast<AdmissibilityCondition*>((void*)settings->admissibilityCondition);
    settingsCxx.clustering = static_cast<ClusteringAlgorithm*>((void*) settings->clustering);
    settingsCxx.maxLeafSize = settings->maxLeafSize;
    settingsCxx.maxParallelLeaves = settings->maxParallelLeaves;
    settingsCxx.elementsPerBlock = settings->elementsPerBlock;
    settingsCxx.useLu = settings->useLu;
    settingsCxx.useLdlt = settings->useLdlt;
    settingsCxx.coarsening = settings->coarsening;
    settingsCxx.recompress = settings->recompress;
    settingsCxx.validateCompression = settings->validateCompression;
    settingsCxx.validationErrorThreshold = settings->validationErrorThreshold;
    settingsCxx.validationReRun = settings->validationReRun;
    settingsCxx.validationDump = settings->validationDump;
    settingsCxx.setParameters();
    settingsCxx.printSettings();
    return rc;
}

const char * hmat_get_version()
{
    return HMAT_VERSION;
}

const char * hmat_get_build_date()
{
  return "Built on " __DATE__ " at " __TIME__ "" ;
}
