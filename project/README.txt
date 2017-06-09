Final Project Description
==========================

1. Installation.

One must have FFTW (with --enable-float and --enable-shared) and CUDA installed,
with CUFFT and CUBLAS. I have built my solvers on top of the Paralution platform,
which is an extensible (but sadly not extensible enough, more on that later)
framework for iterative solvers. I have implemented an ODE boundary value problem
solver using Chebyshev-Spectral methods, and the linear system is only accessible
through the action of a "forward map" operator which yields the action of the
operator on a given vector. Specialized methods like GMRES are well equipped
for this class of problems. To be concrete, we are solving this particular
2-point ODE boundary value problem:

    {u(x) - p(x) * u'(x) - q(x) * u''(x) = f(x), x_l < x < x_r
    {u(x_l) = u_l
    {u(x_r) = u_r

To install paralution, use the attached code. Note that to allow this solver
to fit I had to modify the paralution source code, in some places heavily.
Most of my work is in {host,gpu}_stencil_chebyshev1d.{cpp,hpp} but other modifications
were necessary. For a complete list of changes download the original source at
 http://www.paralution.com/downloads/paralution-1.1.0.tar.gz
and run a recursive diff. 

To build, using CMake:
    cd paralution-1.1.0/
    mkdir build && cd build
    cmake -DFFTW_INCLUDES=<path to FFTW includes> -DCMAKE_LIBRARY_PATH=<path to FFTW libs> ..
    make -j
    cd ../../host
    g++ -c build_cheb_precond_csr.cpp
    cd ../
    g++ -o cheb_ode cheb_ode.cpp host/build_cheb_precond_csr.o -Iparalution-1.1.0/build/inc \
        -Ihost/ -lparalution -Lparalution-1.1.0/build/lib/ -I/usr/local/cuda/include -lcudart \
        -L/usr/local/cuda/lib64


2. Program Results / Analysis

The original intention was to write a full PDE solver with time-stepping (see original
document). While I did indeed implement such a solver in serial, in parallel this
had a confluence of ultimately fatal flaws, and I was only able to write a parallel
ODE solver. The idea of the PDE solver was to launch task-parallel kernels which would
each live in streams and be synchronized at the end of each time step. The first flaw
is that cuFFT does not allow FFTs to be executed from kernel code, so all must be run
through Streams for parallelization. The second problem is that the GMRES solver has
blocking operations which make asynchronous calls unworkable (I tried this to no avail).

The answer to this problem can be resolved by launching each solver in a separate host
thread, either with POSIX pthreads or with C++11 async features. On advice of the TAs
this was not done and instead I restricted my work simply to the ODE case which is
a necessary component to the larger problem and is parallelized here.

The program can be run by executing e.g.:

    ./cheb_ode 1024 0

Which uses a 1024-pt. 1D grid and attempts to solve (on the CPU since `0' implies no
GPU acceleration) the linear system for the differential equation. Conversely,

    ./cheb_ode 1024 1

uses the CUDA kernels I wrote to apply the forward map on the GPU, and still use the
same GMRES solvers to iterate towards a solution.

Now, in truth both of these will not actually converge to the proper solution (even
though the forward maps and the preconditioner have been properly debugged and
verified for correctness). The reason is that Paralution uses templates for its
various types. I am using a LocalStencil type for the forward map operator, and the
preconditioner is a tridiagonal matrix of type LocalMatrix. Even though I only need to
execute one routine, it is not possible to use casts across template parameters, even
when there is no real type incompatibility! I'm unsure how to actually resolve this,
apart from a wrapper which simulates a LocalMatrix using a LocalStencil which seems messy.
See the commented block in cheb_ode.cpp for precisely the kind of mask I want to do,
which results in a segmentation fault.

So with this architectural flaw, what one can do is test both the preconditioner and the
operator independently as solvers. Neither is effective alone but they at least execute.

This can be done by selecting either lu.Solve(rhs, &x) or ls.Solve(rhs, &x) at the end of
cheb_ode.cpp.

Performance
===========
Since the full algorithm does not run as originally conceived, it is difficult to analyze
the performance. A longer project (indeed I'm continuing this as part of my research!)
will correct the architectural deficiencies here and true performance tests will be
possible.

***
Note: I was also limited in time by some outages by the CMS mx & minuteman machines which
which were both down for extended periods (often overlapping with Haru outages). I only used
Haru at the beginning of the project before the instability of Haru made it impossible to
use it as a dev environment. Nevertheless I am turning in what I have complete at this time.
Without mx/minuteman this project would have been impossible, even with the outages mx
and minuteman experienced.

