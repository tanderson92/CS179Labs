#include <cstdlib>
#include <assert.h>

#include "build_cheb_precond_csr.hpp"

void BuildChebPrecondCSR(int n, const float *x,
                         int **row_offset_p, int **col_p, float **val_p,
                         float *p, float *q) {

  float *h = (float *) malloc( (n - 1) * sizeof(float) );
  for (int i = 0; i < n - 1; i++) {
    h[i] = x[i+1] - x[i];
  }

  // Construct the CSR preconditioning matrix, using 2nd order finite differences
  int nnz = 2 + (n - 2) * 3;

  *row_offset_p = (int *) malloc( (n + 1) * sizeof(int) );
  assert( *row_offset_p != NULL );
  int *row_offset = *row_offset_p;
  row_offset[0] = 0;
  row_offset[1] = 1;
  for (int i = 2; i < n; i++) {
    row_offset[i] = row_offset[i - 1] + 3;
  }
  row_offset[n] = nnz;

  *col_p = (int *) malloc( nnz * sizeof(int) );
  assert( *col_p != NULL );
  int *col = *col_p;
  col[0] = 0;
  for (int i = 1; i < n - 1; i++) {
    col[3*(i - 1) + 1] = i - 1;
    col[3*(i - 1) + 2] = i;
    col[3*(i - 1) + 3] = i + 1;
  }
  col[nnz - 1] = n - 1;

  *val_p = (float *) malloc( nnz * sizeof(float) );
  assert( *val_p != NULL );
  float *val = *val_p;
  val[0] = 1.0; // left boundary condition
  for (int i = 0; i < n - 1; i++) {
    val[3*i + 1] = -p[i]*(h[i+1] / h[i] / (h[i] + h[i+1])) - q[i]*(2.0/h[i]/(h[i] + h[i+1]));
    val[3*i + 2] = 1.0 - p[i] * (1.0/h[i] - 1.0/h[i+1]) - q[i]*(-2.0/h[i]/h[i+1]);
    val[3*i + 3] = -p[i]*(h[i]/h[i+1]/(h[i] + h[i+1])) - q[i]*(2.0/h[i+1]/(h[i] + h[i+1]));
  }
  val[nnz - 1] = 1.0; // right boundary condition

}
