#include <iostream>
#include <cstdlib>

#define _USE_MATH_DEFINES
#include <math.h>

#include "paralution.hpp"
#include "build_cheb_precond_csr.hpp"
#include "cuda.h"
#include "cuda_runtime.h"

using namespace paralution;

// Evaluates p(x)
inline float pfcn(float x) {
  return 24.0*x/(1.0 + 4.0*pow(x, 2.0));
}

// Evaluates q(x)
inline float qfcn(float x) {
  return (1.0 + 8*pow(x, 3.0))/(1 + 4.0*pow(x, 2.0));
}

// Evaluates f(x)
inline float ffcn(float x) {
  return 48.0*pow(x, 2.0)/(1.0 + 4.0*pow(x, 2.0)) * sin(2.0 + pow(x, 2.0)) +
         (1.0 + 8.0*pow(x, 3.0))/(1.0 + 4.0*pow(x, 2.0)) *
         (2.0*sin(2.0 + pow(x, 2.0)) + 4.0*pow(x, 2.0)*cos(2.0 + pow(x, 2.0)))
         + cos(2.0 + pow(x, 2.0));
}

// Solution u(x)
inline float ufcn(float x) {
    return cos(2.0 + pow(x, 2.0));
}

// Set up the grid functions on the scaled grid cheb_grid, a 1D chebyshev domain
void init_grid_CPU(int N, float **cheb_grid_p, float **p_p, float **q_p, const float x_a, const float x_b) {
  int npts = N + 1;

  *cheb_grid_p = (float *) malloc( npts * sizeof(float) );
  *p_p = (float *) malloc( (npts - 2) * sizeof(float) );
  *q_p = (float *) malloc( (npts - 2) * sizeof(float) );
  float *cheb_grid = *cheb_grid_p;
  float *p = *p_p;
  float *q = *q_p;

  for (int i = 0; i < npts; i++) {
    float chebpt = cos( (N - i) * M_PI / ( (float) N ));
    cheb_grid[i] = (x_b - x_a)*(chebpt + 1.0)/2.0 + x_a;
  }
  for (int i = 1; i < npts - 1; i++) {
    p[i-1] = pfcn(cheb_grid[i]);
    q[i-1] = qfcn(cheb_grid[i]);
  }
}

int main(int argc, char* argv[]) {
  double tick, tack, start, end;
  int N, npts;
  float x_a = -1.0;
  float x_b =  5.0;

  start = paralution_time();

  if (argc == 1) {
    std::cerr << argv[0] << " <npts> [Acceleration <0> or <1>]" << std::endl;
    exit(1);
  }

  init_paralution();
  N = atoi(argv[1]);
  npts = N + 1;
  set_omp_threads_paralution(1);
  bool accel = false;
  if (argc > 2)
    accel = atoi(argv[2]);

  info_paralution();

  LocalVector<float> x;
  LocalVector<float> rhs;
  LocalMatrix<float> mat;

  tick = paralution_time();
  float *cheb_grid, *p, *q;
  init_grid_CPU(N, &cheb_grid, &p, &q, x_a, x_b);

  float *dev_cheb_grid, *dev_p, *dev_q, *dev_x_a, *dev_x_b;
  cudaMalloc(&dev_cheb_grid, npts * sizeof(float));
  cudaMemcpy( dev_cheb_grid, cheb_grid, npts * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&dev_p, (npts - 2) * sizeof(float));
  cudaMemcpy( dev_p, p, (npts - 2) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&dev_q, (npts - 2) * sizeof(float));
  cudaMemcpy( dev_q, q, (npts - 2) * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_x_a, sizeof(float));
  cudaMemcpy( dev_x_a, &x_a, sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&dev_x_b, sizeof(float));
  cudaMemcpy( dev_x_b, &x_b, sizeof(float), cudaMemcpyHostToDevice);
  // Build a finite difference approximation to the ODE operator
  // using the grid functions. The matrix is tridiagonal because we use 2nd
  // order discretizations. The preconditioner is used to help an iterative
  // method converge, and is generally cheaper than the forward map itself
  // which in this case is the "Stencil" which is computed with FFTs
  {
    int *row_offset, *col;
    float *val;
    int nnz = 2 + (npts - 2) * 3;
    BuildChebPrecondCSR(npts, cheb_grid, &row_offset, &col, &val, p, q);
    mat.SetDataPtrCSR(&row_offset, &col, &val, "chebprec", nnz, npts, npts);
  }
  if (accel) mat.MoveToAccelerator();
  GMRES<LocalStencil<float>, LocalVector<float>, float > ls;
  LocalStencil<float> stencil(Chebyshev1D, accel);
  stencil.SetGrid(npts);
  // Build FFTW plans
  if (accel) {
    stencil.Plan(dev_p, dev_q, dev_x_a, dev_x_b);
  } else {
    stencil.Plan(p, q, &x_a, &x_b);
  }
  // Matrix preconditioner
  LU<LocalMatrix<float>, LocalVector<float>, float > lu;
  Solver<LocalStencil<float>, LocalVector<float>, float > *solv;
  lu.SetOperator(mat);
  lu.Verbose(2);
  lu.Build();

  ls.SetOperator(stencil);

  /* Unfortunately, the inability to use a LU<LocalMatrix<float>...> with a
   * Solver<LocalStencil<float>...> means that the preconditioner we have built
   * cannot be used */
  //solv = reinterpret_cast<LU<LocalStencil<float>, LocalVector<float>, float > *> (&lu);
  //ls.SetPreconditioner(*solv);

  ls.Build();

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());
  x.Zeros(); // Initial guess
  rhs[0] = ufcn(cheb_grid[0]); // Dirichlet Boundary condition
  for (int i = 1; i < npts - 1; i++) {
    rhs[i] = ffcn(cheb_grid[i]);
  }
  rhs[npts - 1] = ufcn(cheb_grid[npts - 1]);

  if (accel) {
    x.MoveToAccelerator();
    rhs.MoveToAccelerator();
  }

  tack = paralution_time();
  stencil.info();
  std::cout << "Build time:" << (tack - tick)/pow(10, 6) << " sec" << std::endl;

  ls.Init(1e-6, 1e-6, 1e+15, 300);
  ls.Verbose(2);

  ls.Solve(rhs, &x);
  x.WriteFileASCII("ODE_Solution.txt");
}
