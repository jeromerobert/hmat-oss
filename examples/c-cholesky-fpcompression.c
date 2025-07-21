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

#include <stdio.h>
#include <math.h>
#ifdef __cplusplus
#include <complex>
typedef std::complex<double> double_complex;
#define make_complex(realPart, imagPart) \
    std::complex<double>(realPart, imagPart)
#else
#include <complex.h>
#ifdef _MSC_VER
typedef _Dcomplex double_complex;
#define make_complex(realPart, imagPart) _Cbuild(realPart, imagPart)
#else
typedef double complex double_complex;
#define make_complex(realPart, imagPart) \
    realPart + imagPart * _Complex_I
#endif
#endif

#include <string.h>
#include <time.h>

#include "hmat/hmat.h"
#include "examples.h"

/** This is a simple example showing how to use the HMatrix library.  */

typedef struct {
  int n;
  double* points;
  double l;
} problem_data_t;

/**
  Define interaction between 2 degrees of freedoms  (real case)
 */
void interaction_real(void* data, int i, int j, void* result)
{
  problem_data_t* pdata = (problem_data_t*) data;
  double* points = pdata->points;
  double r = distanceTo(&points[3*i], &points[3*j]);

  *((double*)result) = exp(-fabs(r) / pdata->l);
}

void double_precision_error(problem_data_t *problem_data, int n, double *rhs, double *rhsCopy, double *result)
{
  int i,j;
  double rhsCopyNorm = 0.;
  double rhsNorm = 0.;
  double diffNorm = 0;
  double diff;
  double a;

  for (i = 0; i < n; i++) {
      rhsCopyNorm += rhsCopy[i] * rhsCopy[i];
  }

  rhsCopyNorm = sqrt(rhsCopyNorm);
  fprintf(stdout, "\n||b|| = %e\n", rhsCopyNorm);

  for (i = 0; i < n; i++) {
      rhsNorm += rhs[i] * rhs[i];
  }

  rhsNorm = sqrt(rhsNorm);
  fprintf(stdout, "||x|| = %e\n", rhsNorm);

  for (i = 0; i < n; i++) {
      diff = rhsCopy[i];
      for (j = 0; j < n; j++) {
	  interaction_real(problem_data, i, j, &a);
	  diff      -= a * rhs[j];
      }
      diffNorm += diff*diff;
  }

  *result = diffNorm/rhsCopyNorm;

}

void simple_precision_error(problem_data_t *problem_data, int n, float *rhs, float *rhsCopy, float *result)
{
  int i,j;
  float rhsCopyNorm = 0.;
  float rhsNorm = 0.;
  float diffNorm = 0;
  float diff;
  double a;

  for (i = 0; i < n; i++) {
      rhsCopyNorm += rhsCopy[i] * rhsCopy[i];
  }

  rhsCopyNorm = sqrt(rhsCopyNorm);
  fprintf(stdout, "\n||b|| = %e\n", rhsCopyNorm);

  for (i = 0; i < n; i++) {
      rhsNorm += rhs[i] * rhs[i];
  }

  rhsNorm = sqrt(rhsNorm);
  fprintf(stdout, "||x|| = %e\n", rhsNorm);

  for (i = 0; i < n; i++) {
      diff = rhsCopy[i];
      for (j = 0; j < n; j++) {
	  interaction_real(problem_data, i, j, &a);
	  diff      -= a * rhs[j];
      }
      diffNorm += diff*diff;
  }

  *result = diffNorm/rhsCopyNorm;

}

int main(int argc, char **argv) {
  clock_t start = clock();
  int i;
  double radius, step;
  double* points;
  int n;
  int nb_blocs = 0;
  double epsilon;
  hmat_FPcompress_t compressor_type = DEFAULT_COMPRESSOR;
  hmat_interface_t hmat;
  hmat_settings_t settings;
  hmat_value_t type;
  hmat_info_t mat_info;
  hmat_profile_t mat_profile;
  hmat_FPCompressionRatio_t mat_ratio;
  char arithmetic;
  char* comp_code;
  hmat_clustering_algorithm_t* clustering;
  hmat_clustering_algorithm_t* clustering_shuffle;
  hmat_clustering_algorithm_t* clustering_max_dof;
  hmat_cluster_tree_t* cluster_tree;
  hmat_matrix_t * hmatrix;
  hmat_matrix_t * hmatrix_c;
  int kLowerSymmetric = 1; /* =0 if not Symmetric */

  problem_data_t problem_data;
  double l;

  int nrhs = 1;
  double *drhs, *drhsCopy1, *drhsCopy2, *drhsCopy3, derr, derr2;
  float  *frhs, *frhsCopy1, *frhsCopy2, *frhsCopy3, ferr, ferr2;
  double diffNorm, normx1;

  
  if (argc != 5 && argc != 4 && argc != 6) {
    fprintf(stderr, "Usage: %s n_points (S|D|C|Z) eps [nblocs] [Compressor]\n", argv[0]);
    return 1;
  }

  n = atoi(argv[1]);
  arithmetic = argv[2][0];
  epsilon = atof(argv[3]);
  
  if (argc == 5 || argc == 6)
    {nb_blocs = atoi(argv[4]);}


  if (argc == 6)
  {
    comp_code = argv[5];
    if(strcmp(comp_code,"SZ2")==0) {
      compressor_type = SZ_COMPRESSOR;

    } else if(strcmp(comp_code,"SZ")==0) { //"SZ2" and "SZ" both work
      compressor_type = SZ_COMPRESSOR;

    } else if(strcmp(comp_code,"SZ3")==0) {
      compressor_type = SZ3_COMPRESSOR;

    }else if(strcmp(comp_code,"ZFP")==0) {
      compressor_type = ZFP_COMPRESSOR;

    } else {
      fprintf(stderr, "Unknown compressor : %s. Leave blank for default value\n", comp_code);
      return 1;
    }
  }


  printf("\n =================== Tests Compression on ''Cholesky'' ===================\n");
  printf("with parameters epsilon = %.2e, n_blocs = %d,\non a problem of size %d with arithmetic %c\n\n", epsilon, nb_blocs, n, arithmetic);


  n = atoi(argv[1]);
  arithmetic = argv[2][0];
  switch (arithmetic) {
    case 'S':
      type = HMAT_SIMPLE_PRECISION;
      break;
    case 'D':
      type = HMAT_DOUBLE_PRECISION;
      break;
    default:
      fprintf(stderr, "Unknown arithmetic code %c, exiting...\n", arithmetic);
      return 1;
  }

  hmat_get_parameters(&settings);
  hmat_init_default_interface(&hmat, type);

  /*settings->recompress = 0;*/
  /*settings->admissibilityFactor = 3.;*/

  hmat_set_parameters(&settings);
  if (0 != hmat.init())
  {
    fprintf(stderr, "Unable to initialize HMat library\n");
    return 1;
  }

  clock_t init = clock();
  printf("Generating the point cloud...\n");
  radius = 1.;
  step = 1.75 * M_PI * radius / sqrt((double)n);
  points = createCylinder(radius, step, n);
  printf("done.\n");
  printf("n = %d\n", n);

  l = correlationLength(points, n);
  printf("correlationLength = %le\n", l);
  problem_data.n = n;
  problem_data.points = points;
  problem_data.l = l;

  drhs = createRhs(points, n, l);
  drhsCopy1 = createRhs(points, n, l);
  drhsCopy2 = createRhs(points, n, l);
  drhsCopy3 = createRhs(points, n, l);
  frhs     =  (float*) calloc(n, sizeof(float));
  frhsCopy1 =  (float*) calloc(n, sizeof(float));
  frhsCopy2 =  (float*) calloc(n, sizeof(float));
  frhsCopy3 =  (float*) calloc(n, sizeof(float));
  for(i=0;i<n;i++) {
    frhs[i] = drhs[i];
    frhsCopy1[i] = drhsCopy1[i];
  }
  clock_t generation = clock();

  clustering = hmat_create_clustering_median();
  /* To test with n-ary trees, change final argument to a value > 2 */
  clustering_shuffle = hmat_create_shuffle_clustering(clustering, 2, 2);
  /* Change maxLeafSize to have a deep tree */
  clustering_max_dof = hmat_create_clustering_max_dof(clustering_shuffle, 10);
  cluster_tree = hmat_create_cluster_tree(points, 3, n, clustering_max_dof);
  hmat_delete_clustering(clustering_max_dof);
  hmat_delete_clustering(clustering_shuffle);
  hmat_delete_clustering(clustering);
  printf("ClusterTree node count = %d\n", hmat_tree_nodes_count(cluster_tree));
  hmat_admissibility_t * admissibilityCondition = hmat_create_admissibility_standard(2.0);
  hmatrix = hmat.create_empty_hmatrix_admissibility(
            cluster_tree, cluster_tree, 1, admissibilityCondition);
  hmat_delete_admissibility(admissibilityCondition);
  hmat.get_info(hmatrix, &mat_info);
  printf("HMatrix node count = %d\n", mat_info.nr_block_clusters);

  fprintf(stdout,"Assembly...");
  hmat_assemble_context_t ctx_assemble;
  hmat_assemble_context_init(&ctx_assemble);
  ctx_assemble.compression = hmat_create_compression_aca_plus(1e-4);
  ctx_assemble.user_context = &problem_data;
  ctx_assemble.simple_compute = interaction_real;
  ctx_assemble.lower_symmetric = kLowerSymmetric;
  hmat.assemble_generic(hmatrix, &ctx_assemble);
  fprintf(stdout, "done.\n");
  hmat.get_info(hmatrix, &mat_info);
  printf("HMatrix size = %gM, uncompressed size = %gM\n",
    1e-6 * mat_info.compressed_size, 1e-6 * mat_info.uncompressed_size);
  hmat_delete_compression(ctx_assemble.compression);


  //Copie de hmatrix dans hmatrix_c
  hmatrix_c = hmat.copy(hmatrix);



  clock_t assembling = clock();
  clock_t compression = assembling;
  clock_t decompression = compression;

  //Compression de hmatrix_c
  if(nb_blocs>0) 
  {
    printf("\nCompressing...\n");
    hmat.FPcompress(hmatrix_c, epsilon, nb_blocs, compressor_type, true, true);
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
  
 


    //Décompression de hmatrix_c
  if(nb_blocs>0) 
  {
    
    printf("\nUncompressing...\n");
    hmat.FPdecompress(hmatrix_c);
    decompression = clock();
  }
   
  fprintf(stdout,"Factorisation...");
  hmat_factorization_context_t ctx_facto;
  hmat_factorization_context_init(&ctx_facto);
  ctx_facto.factorization = hmat_factorization_llt;
  hmat.factorize_generic(hmatrix, &ctx_facto);
  hmat.factorize_generic(hmatrix_c, &ctx_facto);
  fprintf(stdout, "done.\n");
 

 // ===================================================
  fprintf(stdout,"Solve...");
  if(type == HMAT_SIMPLE_PRECISION){
    hmat.vector_reorder(frhs, cluster_tree, 0, NULL, nrhs);
    hmat.solve_dense(hmatrix, frhs, nrhs);
    hmat.vector_restore(frhs, cluster_tree, 0, NULL, nrhs);
  }else{
    hmat.vector_reorder(drhs, cluster_tree, 0, NULL, nrhs);
    hmat.solve_dense(hmatrix, drhs, nrhs);
    hmat.vector_restore(drhs, cluster_tree, 0, NULL, nrhs);
  }
  fprintf(stdout, "done.\n");

  
  // Store solution of AX=B 
  memcpy(frhsCopy2, frhs, n*sizeof(float));
  memcpy(drhsCopy2, drhs, n*sizeof(double));

  fprintf(stdout, "Accuracy...");
  if(type == HMAT_SIMPLE_PRECISION){
    simple_precision_error(&problem_data, n, frhs, frhsCopy1, &ferr);
    fprintf(stdout, "||Ax - b|| / ||b|| = %e\n",  ferr);
  }else{
    double_precision_error(&problem_data, n, drhs, drhsCopy1, &derr);
    fprintf(stdout, "||Ax - b|| / ||b|| = %le\n",  derr);
  }
  // ===================================================

  // Restore right-hand side 
  memcpy(frhs, frhsCopy1, n*sizeof(float));
  memcpy(drhs, drhsCopy1, n*sizeof(double));
  memcpy(frhsCopy3, frhsCopy1, n*sizeof(float));
  memcpy(drhsCopy3, drhsCopy1, n*sizeof(double));
  // ===================================================
  fprintf(stdout,"Solve on decompressed matrix...\n");
  if(type == HMAT_SIMPLE_PRECISION){
    hmat.vector_reorder(frhs, cluster_tree, 0, NULL, nrhs);
    hmat.solve_dense(hmatrix_c, frhs, nrhs);
    hmat.vector_restore(frhs, cluster_tree, 0, NULL, nrhs);
  }else{
    hmat.vector_reorder(drhs, cluster_tree, 0, NULL, nrhs);
    hmat.solve_dense(hmatrix_c, drhs, nrhs);
    hmat.vector_restore(drhs, cluster_tree, 0, NULL, nrhs);
  }
  fprintf(stdout, "done.\n");

  
  // Store solution of A_cX=B 
  memcpy(frhsCopy3, frhs, n*sizeof(float));
  memcpy(drhsCopy3, drhs, n*sizeof(double));

  fprintf(stdout, "Accuracy...");
  if(type == HMAT_SIMPLE_PRECISION){
    simple_precision_error(&problem_data, n, frhs, frhsCopy1, &ferr2);
    fprintf(stdout, "||Ax - b|| / ||b|| = %e\n",  ferr2);
  }else{
    double_precision_error(&problem_data, n, drhs, drhsCopy1, &derr2);
    fprintf(stdout, "||Ax - b|| / ||b|| = %le\n",  derr2);
  }
  // ===================================================

 // ===================================================
  fprintf(stdout, "\nCompare AX=b and A_cX=b solutions...");
  diffNorm = 0.;
  normx1 = 0.;

  if(type == HMAT_SIMPLE_PRECISION){
    for (i = 0; i < n; i++)
      {
        diffNorm += (frhsCopy3[i] - frhsCopy2[i])*(frhsCopy3[i] - frhsCopy2[i]);
        normx1 += frhsCopy2[i] * frhsCopy2[i];
      }
  }else{
    for (i = 0; i < n; i++)
    {
      diffNorm += (drhsCopy3[i] - drhsCopy2[i])*(drhsCopy3[i] - drhsCopy2[i]);
      normx1 += drhsCopy2[i] * drhsCopy2[i];
    }
      
  }
  fprintf(stdout, "\n||x1-x2|| = %e\n", sqrt(diffNorm));
  fprintf(stdout, "||x1-x2||/||x1|| = %e\n", sqrt(diffNorm/normx1));
  /*
 float a = -1;

  switch (arithmetic) {
    case 'S':
    {float alpha = (float) a;
      hmat.axpy(&alpha, hmatrix, hmatrix_c);}      
      break;
    case 'D':
      {double alpha = (double) a;
      hmat.axpy(&alpha, hmatrix, hmatrix_c);}
      break;
    case 'C':
      //{float complex alpha = (float complex) a;
      //hmat.axpy(&alpha, hmatrix, hmatrix_c);}
      //break;
    case 'Z':
      //{double complex alpha = (double complex) a;
      //hmat.axpy(&alpha, hmatrix, hmatrix_c);}
      //break;
    default:
      fprintf(stderr, "Unknown arithmetic code %c\n", arithmetic);
      return 1;
    }



    double normM = hmat.norm(hmatrix);
 double normDelta = hmat.norm(hmatrix_c);
 printf("Residu : %.3e\n", normDelta/normM);*/
  clock_t computing = clock();

  free(frhs);
  free(frhsCopy1);
  free(frhsCopy2);
  free(frhsCopy3);
  free(drhs);
  free(drhsCopy1);
  free(drhsCopy2);
  free(drhsCopy3);
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
  if(nb_blocs>0){
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


