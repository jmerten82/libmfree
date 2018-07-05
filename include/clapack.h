#ifndef CLAPACK_H

#define CLAPACK_H

#ifdef __cplusplus
extern "C"
{
#include "cblas.h"
#ifndef ATL_INT
   #define ATL_INT int
#endif
#ifndef ATL_CINT
   #define ATL_CINT const ATL_INT
#endif
#ifndef ATLAS_ORDER
   #define ATLAS_ORDER CBLAS_ORDER
#endif
#ifndef ATLAS_UPLO
   #define ATLAS_UPLO CBLAS_UPLO
#endif
#ifndef ATLAS_DIAG
   #define ATLAS_DIAG CBLAS_DIAG
#endif

int clapack_sgesv(CBLAS_ORDER Order, const int N, const int NRHS,
                  float *A, const int lda, int *ipiv,
                  float *B, const int ldb);
int clapack_sgetrf(CBLAS_ORDER Order, const int M, const int N,
                   float *A, const int lda, int *ipiv);
int clapack_sgetrs
   (CBLAS_ORDER Order, CBLAS_TRANSPOSE Trans,
    const int N, const int NRHS, const float *A, const int lda,
    const int *ipiv, float *B, const int ldb);
int clapack_sgetri(CBLAS_ORDER Order, const int N, float *A,
                   const int lda, const int *ipiv);
int clapack_sposv(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                  const int N, const int NRHS, float *A, const int lda,
                  float *B, const int ldb);
int clapack_spotrf(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                   const int N, float *A, const int lda);
int clapack_spotrs(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
                   const int N, const int NRHS, const float *A, const int lda,
                   float *B, const int ldb);
int clapack_spotri(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                   const int N, float *A, const int lda);
int clapack_slauum(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                   const int N, float *A, const int lda);
int clapack_strtri(ATLAS_ORDER Order,ATLAS_UPLO Uplo,
                   ATLAS_DIAG Diag, const int N, float *A,
                   const int lda);
int clapack_sgels(CBLAS_ORDER Order,
                  CBLAS_TRANSPOSE TA,
                  ATL_CINT M, ATL_CINT N, ATL_CINT NRHS, float *A,
                  ATL_CINT lda, float *B, const int ldb);
int clapack_sgelqf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   float *A, ATL_CINT lda, float *TAU);
int clapack_sgeqlf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   float *A, ATL_CINT lda, float *TAU);
int clapack_sgerqf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   float *A, ATL_CINT lda, float *TAU);
int clapack_sgeqrf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   float *A, ATL_CINT lda, float *TAU);

int clapack_dgesv(CBLAS_ORDER Order, const int N, const int NRHS,
                  double *A, const int lda, int *ipiv,
                  double *B, const int ldb);
int clapack_dgetrf(CBLAS_ORDER Order, const int M, const int N,
                   double *A, const int lda, int *ipiv);
int clapack_dgetrs
   (CBLAS_ORDER Order, CBLAS_TRANSPOSE Trans,
    const int N, const int NRHS, const double *A, const int lda,
    const int *ipiv, double *B, const int ldb);
int clapack_dgetri(CBLAS_ORDER Order, const int N, double *A,
                   const int lda, const int *ipiv);
int clapack_dposv(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                  const int N, const int NRHS, double *A, const int lda,
                  double *B, const int ldb);
int clapack_dpotrf(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda);
int clapack_dpotrs(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
                   const int N, const int NRHS, const double *A, const int lda,
                   double *B, const int ldb);
int clapack_dpotri(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda);
int clapack_dlauum(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                   const int N, double *A, const int lda);
int clapack_dtrtri(ATLAS_ORDER Order,ATLAS_UPLO Uplo,
                   ATLAS_DIAG Diag, const int N, double *A,
                   const int lda);
int clapack_dgels(CBLAS_ORDER Order,
                  CBLAS_TRANSPOSE TA,
                  ATL_CINT M, ATL_CINT N, ATL_CINT NRHS, double *A,
                  ATL_CINT lda, double *B, const int ldb);
int clapack_dgelqf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   double *A, ATL_CINT lda, double *TAU);
int clapack_dgeqlf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   double *A, ATL_CINT lda, double *TAU);
int clapack_dgerqf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   double *A, ATL_CINT lda, double *TAU);
int clapack_dgeqrf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   double *A, ATL_CINT lda, double *TAU);

int clapack_cgesv(CBLAS_ORDER Order, const int N, const int NRHS,
                  void *A, const int lda, int *ipiv,
                  void *B, const int ldb);
int clapack_cgetrf(CBLAS_ORDER Order, const int M, const int N,
                   void *A, const int lda, int *ipiv);
int clapack_cgetrs
   (CBLAS_ORDER Order, CBLAS_TRANSPOSE Trans,
    const int N, const int NRHS, const void *A, const int lda,
    const int *ipiv, void *B, const int ldb);
int clapack_cgetri(CBLAS_ORDER Order, const int N, void *A,
                   const int lda, const int *ipiv);
int clapack_cposv(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                  const int N, const int NRHS, void *A, const int lda,
                  void *B, const int ldb);
int clapack_cpotrf(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                   const int N, void *A, const int lda);
int clapack_cpotrs(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
                   const int N, const int NRHS, const void *A, const int lda,
                   void *B, const int ldb);
int clapack_cpotri(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                   const int N, void *A, const int lda);
int clapack_clauum(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                   const int N, void *A, const int lda);
int clapack_ctrtri(ATLAS_ORDER Order,ATLAS_UPLO Uplo,
                   ATLAS_DIAG Diag, const int N, void *A,
                   const int lda);
int clapack_cgels(CBLAS_ORDER Order,
                  CBLAS_TRANSPOSE TA,
                  ATL_CINT M, ATL_CINT N, ATL_CINT NRHS, void *A,
                  ATL_CINT lda, void *B, const int ldb);
int clapack_cgelqf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);
int clapack_cgeqlf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);
int clapack_cgerqf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);
int clapack_cgeqrf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);

int clapack_zgesv(CBLAS_ORDER Order, const int N, const int NRHS,
                  void *A, const int lda, int *ipiv,
                  void *B, const int ldb);
int clapack_zgetrf(CBLAS_ORDER Order, const int M, const int N,
                   void *A, const int lda, int *ipiv);
int clapack_zgetrs
   (CBLAS_ORDER Order, CBLAS_TRANSPOSE Trans,
    const int N, const int NRHS, const void *A, const int lda,
    const int *ipiv, void *B, const int ldb);
int clapack_zgetri(CBLAS_ORDER Order, const int N, void *A,
                   const int lda, const int *ipiv);
int clapack_zposv(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                  const int N, const int NRHS, void *A, const int lda,
                  void *B, const int ldb);
int clapack_zpotrf(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                   const int N, void *A, const int lda);
int clapack_zpotrs(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
                   const int N, const int NRHS, const void *A, const int lda,
                   void *B, const int ldb);
int clapack_zpotri(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                   const int N, void *A, const int lda);
int clapack_zlauum(ATLAS_ORDER Order, ATLAS_UPLO Uplo,
                   const int N, void *A, const int lda);
int clapack_ztrtri(ATLAS_ORDER Order,ATLAS_UPLO Uplo,
                   ATLAS_DIAG Diag, const int N, void *A,
                   const int lda);
int clapack_zgels(CBLAS_ORDER Order,
                  CBLAS_TRANSPOSE TA,
                  ATL_CINT M, ATL_CINT N, ATL_CINT NRHS, void *A,
                  ATL_CINT lda, void *B, const int ldb);
int clapack_zgelqf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);
int clapack_zgeqlf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);
int clapack_zgerqf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);
int clapack_zgeqrf(CBLAS_ORDER Order, ATL_CINT M, ATL_CINT N,
                   void *A, ATL_CINT lda, void *TAU);

}
#endif
#endif
