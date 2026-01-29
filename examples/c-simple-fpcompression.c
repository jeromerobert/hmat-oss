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

// Cylinder
#include <stdio.h>
#include <math.h>
#ifdef __cplusplus
#include <complex>
typedef std::complex<double> double_complex;
typedef std::complex<float> float_complex;
#define make_complex(realPart, imagPart) \
std::complex<double>(realPart, imagPart)
#define make_fcomplex(realPart, imagPart) \
std::complex<double>(realPart, imagPart)
#else
#include <complex.h>
#ifdef _MSC_VER
typedef _Dcomplex double_complex;
typedef _Fcomplex float_complex;
#define make_complex(realPart, imagPart) _Cbuild(realPart, imagPart)
#define make_fcomplex(realPart, imagPart) _Fbuild(realPart, imagPart)
#else
typedef double complex double_complex;
typedef float complex float_complex;
#define make_complex(realPart, imagPart) \
    realPart + imagPart * _Complex_I
#define make_fcomplex(realPart, imagPart) \
realPart + imagPart * _Complex_I
#endif
#endif

#include <time.h>
#include <string.h>

#include "hmat/hmat.h"
#include "examples.h"

/** This is a simple example showing how to use the HMatrix library.

    In this example, we assemble and do a decomposition of a Matrix such that:
    \f[A_{ij} = \frac{e^{i\kappa |x_i - x_j|}}{4 \pi |x_i - x_j|}\f]
    with the points \f$(x_i)\f$ on a cylinder.
    In the real case we use 1 / r instead.
 */

/**
  Define interaction between 2 degrees of freedoms  (real case)
*/
double interaction_real(double* points, int i, int j)
{
  double dx = points[3*i+0] - points[3*j+0];
  double dy = points[3*i+1] - points[3*j+1];
  double dz = points[3*i+2] - points[3*j+2];
  return 1. / (sqrt(dx*dx + dy*dy + dz*dz) + 1e-10);
}

/**
  Define interaction between 2 degrees of freedoms  (complex case)
*/
double_complex
interaction_complex(double* points, double k, int i, int j)
{
  double dx = points[3*i+0] - points[3*j+0];
  double dy = points[3*i+1] - points[3*j+1];
  double dz = points[3*i+2] - points[3*j+2];
  double distance = sqrt(dx*dx + dy*dy + dz*dz) + 1e-10;
  double realPart = cos(k * distance) / (4 * M_PI * distance);
  double imagPart = sin(k * distance) / (4 * M_PI * distance);
  return make_complex(realPart, imagPart);
}

/**
  This structure contains data related to our problem.
*/
typedef struct {
  int type;
  int n;
  double* points;
  double k;
} problem_data_t;

/**
  This structure is a glue between hmat library and application.
*/
typedef struct {
  int row_start;
  int col_start;
  int* row_hmat2client;
  int* col_hmat2client;
  problem_data_t* user_context;
} block_data_t;

/**
  Function to free our block_data_t structure.  As we only store pointers, there is no
*/
void
free_hmat(void *data)
{
  free((block_data_t*) data);
}

/**
  prepare_hmat is called by hmat library to prepare assembly of
  a cluster block.  We allocate a block_data_t, which is then
  passed to the compute_hmat function.  We have to store all
  datas needed by compute_hmat into this block_data_t structure.
*/
void
prepare_hmat(int row_start,
             int row_count,
             int col_start,
             int col_count,
             int *row_hmat2client,
             int *row_client2hmat,
             int *col_hmat2client,
             int *col_client2hmat,
             void *user_context,
             hmat_block_info_t * block_info)
{
  /* Silence C/C++ compiler warnings */
  (void) row_count; (void) col_count; (void) row_client2hmat; (void) col_client2hmat;

  block_info->user_data = calloc(1, sizeof(block_data_t));
  block_info->release_user_data = free_hmat;
  block_data_t* bdata = (block_data_t*) block_info->user_data;

  bdata->row_start = row_start;
  bdata->col_start = col_start;
  bdata->row_hmat2client = row_hmat2client;
  bdata->col_hmat2client = col_hmat2client;
  bdata->user_context = (problem_data_t*) user_context;
}

/**
  Compute all values of a block and store them into an array,
  which had already been allocated by hmat.  There is no
  padding, all values computed in this block are contiguous,
  and stored in a column-major order.
  This block is not necessarily identical to the one previously
  processed by prepare_hmat, it may be a sub-block.  In fact,
  it is either the full block, or a full column, or a full row.
*/
void
compute_hmat(void *data,
             int rowBlockBegin,
             int rowBlockCount,
             int colBlockBegin,
             int colBlockCount,
             void *values)
{
  int i, j;
  double *dValues = (double *) values;
  double_complex *zValues = (double_complex *) values;
  block_data_t* bdata = (block_data_t*) data;

  int type = bdata->user_context->type;
  int pos = 0;
  for (j = 0; j < colBlockCount; ++j) {
      int col = bdata->col_hmat2client[j + colBlockBegin + bdata->col_start];
      switch (type) {
	case HMAT_SIMPLE_PRECISION:
	case HMAT_DOUBLE_PRECISION:
	  for (i = 0; i < rowBlockCount; ++i, ++pos)
	    dValues[pos] = interaction_real(bdata->user_context->points, bdata->row_hmat2client[i + rowBlockBegin + bdata->row_start], col);
	  break;
	case HMAT_SIMPLE_COMPLEX:
	case HMAT_DOUBLE_COMPLEX:
	  for (i = 0; i < rowBlockCount; ++i, ++pos)
	    zValues[pos] = interaction_complex(bdata->user_context->points, bdata->user_context->k, bdata->row_hmat2client[i + rowBlockBegin + bdata->row_start], col);
	  break;
      }
  }
}


int main(int argc, char **argv) {
  clock_t start = clock();
  //INITIALISATION
  double radius, step, k;
  double* points;
  hmat_settings_t settings;
  hmat_interface_t hmat;
  hmat_value_t scalar_type;
  hmat_info_t mat_info;
  hmat_profile_t mat_profile;
  hmat_FPCompressionRatio_t mat_ratio;
  hmat_fp_settings_t fp_settings;
  int n;
  char arithmetic;
  char* comp_code;
  hmat_clustering_algorithm_t* clustering, * clustering_algo;
  hmat_cluster_tree_t* cluster_tree;
  hmat_matrix_t* hmatrix;
  hmat_matrix_t* hmatrix_c;
  problem_data_t problem_data;
  hmat_admissibility_t * admissibilityCondition = hmat_create_admissibility_standard(3.0);

  if (argc != 5 && argc != 4 && argc != 6) {
    fprintf(stderr, "Usage: %s n_points (S|D|C|Z) eps [nblocs] [Compressor]\n", argv[0]);
    return 1;
  }

  n = atoi(argv[1]);
  arithmetic = argv[2][0];
  fp_settings.epsilonFP = atof(argv[3]);

  fp_settings.nb_blocs = 0; //Default value : no compression if <= 0;
  if (argc == 5 || argc == 6)
    {fp_settings.nb_blocs = atoi(argv[4]);}

  if (argc == 6)
  {
    comp_code = argv[5];
    if(strcmp(comp_code,"SZ2")==0) {
      fp_settings.compressor = SZ_COMPRESSOR;

    } else if(strcmp(comp_code,"SZ")==0) { //"SZ2" and "SZ" both works
      fp_settings.compressor = SZ_COMPRESSOR;

    } else if(strcmp(comp_code,"SZ3")==0) {
      fp_settings.compressor = SZ3_COMPRESSOR;

    } else if(strcmp(comp_code,"ZFP")==0) {
      fp_settings.compressor = ZFP_COMPRESSOR;

    } else {
      fprintf(stderr, "Unknown compressor : %s. Leave blank for default value\n", comp_code);
      return 1;
    }
  }

  fp_settings.compressFull = true;
  fp_settings.compressRk = true;

  printf("\n =================== Tests Compression on Cylinder ===================\n");
  printf("With parameters epsilon = %.2e, n_blocs = %d,\non a problem of size %d with arithmetic %c\n\n", fp_settings.epsilonFP, fp_settings.nb_blocs, n, arithmetic);


  hmat_get_parameters(&settings);

  
  switch (arithmetic) {
  case 'S':
    scalar_type = HMAT_SIMPLE_PRECISION;
    break;
  case 'D':
    scalar_type = HMAT_DOUBLE_PRECISION;
    break;
  case 'C':
    scalar_type = HMAT_SIMPLE_COMPLEX;
    break;
  case 'Z':
    scalar_type = HMAT_DOUBLE_COMPLEX;
    break;
  default:
    fprintf(stderr, "Unknown arithmetic code %c\n", arithmetic);
    return 1;
  }


  hmat_init_default_interface(&hmat, scalar_type);

  hmat_set_parameters(&settings);
  if (0 != hmat.init())
  {
    fprintf(stderr, "Unable to initialize HMat library\n");
    return 1;
  }
  clock_t init = clock();
  //POINTS GENERATION
  printf("Generating the point cloud...\n");

  radius = 1.;
  step = 1.75 * M_PI * radius / sqrt((double)n);
  k = 2 * M_PI / (10. * step); // 10 points / lambda
  points = createCylinder(radius, step, n);
  printf("done.\n");

  pointsToFile(points, n, "points.txt");

  problem_data.n = n;
  problem_data.points = points;
  problem_data.k = k;
  problem_data.type = scalar_type;

  clock_t generation = clock();

  //HMATRIX ASSEMBLING
  printf("\nAssembling...\n");
  clustering_algo = hmat_create_clustering_median();
  clustering = hmat_create_clustering_max_dof(clustering_algo, 80);
  cluster_tree = hmat_create_cluster_tree(points, 3, n, clustering);
  hmat_delete_clustering(clustering);
  hmat_delete_clustering(clustering_algo);
  printf("ClusterTree node count = %d\n", hmat_tree_nodes_count(cluster_tree));
  hmatrix = hmat.create_empty_hmatrix_admissibility(
            cluster_tree, cluster_tree, 0, admissibilityCondition);
  hmat.set_low_rank_epsilon(hmatrix, 1e-5);
  hmat_delete_admissibility(admissibilityCondition);
  
  hmat_assemble_context_t ctx_assemble;
  hmat_assemble_context_init(&ctx_assemble);
  ctx_assemble.compression = hmat_create_compression_aca_plus(1e-5);
  ctx_assemble.user_context = &problem_data;
  ctx_assemble.prepare = prepare_hmat;
  ctx_assemble.block_compute = compute_hmat;
  ctx_assemble.lower_symmetric = 0;
  int rc = hmat.assemble_generic(hmatrix, &ctx_assemble);
  if (0 != rc) {
      fprintf(stderr, "Error during assembly, aborting\n");
      exit(rc);
  }
  hmat_delete_compression(ctx_assemble.compression);

  hmat.get_info(hmatrix, &mat_info);
  printf("HMatrix node count = %d\n", mat_info.nr_block_clusters);
  printf("HMatrix size = %gM, uncompressed size = %gM\n",
    1e-6 * mat_info.compressed_size, 1e-6 * mat_info.uncompressed_size);

  



  //Copie de hmatrix dans hmatrix_c
  hmatrix_c = hmat.copy(hmatrix);

  clock_t assembling = clock();
  clock_t compression = assembling;
  clock_t decompression = compression;

    //Compression de hmatrix_c
  if(fp_settings.compressRk || fp_settings.compressFull) 
  {
    printf("\nCompressing...\n");
    //Setting compression parameters
    hmat.SetFPCompressionSettings(hmatrix_c, fp_settings); //Also possible : hmat.SetFPCompressionSettingsParams(hmatrix_c, epsilon, nb_blocs, compressor_type, true, true);
    //Applying compression
    hmat.FPcompress(hmatrix_c);
    
    compression = clock();
  }
  
  hmat.get_ratio(hmatrix_c, &mat_ratio);

  printf("- Compression Ratios : \n");
  printf(" # Full Blocs : %7.3f, Rk Blocs %7.3f: , Global : %7.3f\n", mat_ratio.fullRatio, mat_ratio.rkRatio, mat_ratio.ratio);
  printf("- Sizes : \n");
  printf(" # Full Blocs : %fM, Rk Blocs %fM: , Global : %fM\n", 1e-6*mat_ratio.size_Full, 1e-6*mat_ratio.size_Rk, 1e-6*(mat_ratio.size_Full + mat_ratio.size_Rk));
  
  hmat.get_profile(hmatrix_c, &mat_profile);

  printf("- Compression Ratios : \n");
  printf(" # Full Blocs : %7.3f, Rk Blocs %7.3f: , Global : %7.3f\n", mat_ratio.fullRatio, mat_ratio.rkRatio, mat_ratio.ratio);
  printf("- Sizes : \n");
  printf(" # Full Blocs : %fM, Rk Blocs %fM: , Global : %fM\n", 1e-6*mat_ratio.size_Full, 1e-6*mat_ratio.size_Rk, 1e-6*(mat_ratio.size_Full + mat_ratio.size_Rk));
  printf("- Sizes compressed: \n");
  printf(" # Full Blocs : %fM, Rk Blocs %fM: , Global : %fM\n", 1e-6*mat_ratio.size_Full_compressed, 1e-6*mat_ratio.size_Rk_compressed, 1e-6*(mat_ratio.size_Full_compressed + mat_ratio.size_Rk_compressed));
  


    //Décompression de hmatrix_c
  if(fp_settings.compressRk || fp_settings.compressFull) 
  {
    
    printf("\nUncompressing...\n");
    hmat.FPdecompress(hmatrix_c);
    decompression = clock();
  }

  
  
#ifndef _MSC_VER

  //Calcul du résidu
  printf("\nComputing...\n");
  
  switch (arithmetic) {
    case 'S':
    {float alpha = (float) -1.;
      hmat.axpy(&alpha, hmatrix, hmatrix_c);}      
      break;
    case 'D':
    {double alpha = (double) -1.;
      hmat.axpy(&alpha, hmatrix, hmatrix_c);}
      break;
    case 'C':
      {float_complex alpha = make_fcomplex(-1., 0.);
      hmat.axpy(&alpha, hmatrix, hmatrix_c);}
      break;
    case 'Z':
      {double_complex alpha = make_complex(-1., 0.);
      hmat.axpy(&alpha, hmatrix, hmatrix_c);}
      break;
    default:
      fprintf(stderr, "Unknown arithmetic code %c\n", arithmetic);
      return 1;
    }


  double normM = hmat.norm(hmatrix);
  double normDelta = hmat.norm(hmatrix_c);
  printf("Residu : %.3e\n", normDelta/normM);

#endif

  clock_t computing = clock();
  //FINALISATION
  hmat.destroy(hmatrix);
  hmat.destroy(hmatrix_c);
  hmat_delete_cluster_tree(cluster_tree);
  hmat.finalize();
  free(points);
  clock_t end = clock();

  float totalTime = (float)(end - start) / CLOCKS_PER_SEC;
  float initTime = (float)(init - start) / CLOCKS_PER_SEC;
  float generationTime = (float)(generation - init) / CLOCKS_PER_SEC;
  float assemblingTime = (float)(assembling - generation) / CLOCKS_PER_SEC;
  float compressionTime = (float)(compression - assembling) / CLOCKS_PER_SEC;
  float decompressionTIme = (float)(decompression - compression) / CLOCKS_PER_SEC;
  float computingTime = (float)(computing - decompression) / CLOCKS_PER_SEC;
  float finalisationTime = (float)(end - computing) / CLOCKS_PER_SEC;

  printf("\n=== TIMER ===\n");
  printf("Initialization : %8.3f s\n", initTime);
  printf("Generation     : %8.3f s\n", generationTime);
  printf("Assembling     : %8.3f s\n", assemblingTime);
  if(fp_settings.compressRk || fp_settings.compressFull){
    printf("Compression    : %8.3f s\n", compressionTime);
    printf("Decompression  : %8.3f s\n", decompressionTIme);
  }
  printf("Computing      : %8.3f s\n", computingTime);
  printf("Finalization   : %8.3f s\n", finalisationTime);
  printf("Total          : %8.3f s\n", totalTime);
  printf("===  END  ===\n\n");
  printf("\n =========================== End of Tests ============================\n\n\n\n");

  return 0;
}
