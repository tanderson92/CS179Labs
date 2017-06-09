#ifndef BUILD_CSR_HPP
#define BUILD_CSR_HPP

void BuildChebPrecondCSR(int n, const float *x,
                         int **row_offset, int **col, float **val,
                         float *p, float *q);

#endif
