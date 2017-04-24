CS 179
Assignment 2

Due: Wednesday, April 19, 2017 - 3:00 PM.

Time taken to complete: 12 hours (estimated)
Time to complete Part 1: 4 hours

PART 1

Question 1.1: Latency Hiding (5 points)
---------------------------------------

> Approximately how many arithmetic instructions does it take to hide the latency
> of a single arithmetic instruction on a GK110?

> Assume all of the arithmetic instructions are independent (ie have no
> instruction dependencies).

> You do not need to consider the number of execution cores on the chip.

> Hint: What is the latency of an arithmetic instruction? How many instructions
> can a GK110 begin issuing in 1 clock cycle (assuming no dependencies)?

The latency of a single arithmetic instruction on a GPU is ~10 seconds. A 1Ghz GPU executes
1 clock cycle per nanosecond (10^-9 s), so 10 clock cycles per unit of arithmetic instruction
latency. Now, each Kepler SMX (which the GK110 has) has 4 warp schedulers, each with dual
instruction dispatch units[1]. This means that every clock cycle they execute up to 8
instructions per clock cycle, assuming no warp divergence.

So we need approximately
    80 arithmetic instructions / 10 nanoseconds =
        (8 arithmetic instructions / clock cycle)(10 clock cycles / 10 nanoseconds)

So, ~80 arithmetic instructions to hide this latency.

[1]: https://www.nvidia.com/content/PDF/kepler/NVIDIA-Kepler-GK110-Architecture-Whitepaper.pdf

Question 1.2: Thread Divergence (6 points)
------------------------------------------

> Let the block shape be (32, 32, 1).

> (a)
> int idx = threadIdx.y + blockSize.y * threadIdx.x;
> if (idx % 32 < 16)
>     foo();
> else
>     bar();
>
> Does this code diverge? Why or why not?
>
> This code will not diverge; The faster varying index is `threadIdx.x', meaning
> that in each warp it runs from 0-31; as a result the value of `idx' in each
> thread will differ by blockSize.y = 32. Therefore `idx % 32' will be the same
> for every thread in the warp, and so either foo() or bar() will be executed for
> *every* thread in the same warp, and so since all threads follow the same
> code path, there is no thread divergence.
>
> (b)
> const float pi = 3.14;
> float result = 1.0;
> for (int i = 0; i < threadIdx.x; i++)
>     result *= pi;
>
> Does this code diverge? Why or why not? (This is a bit of a trick question,
> either "yes" or "no can be a correct answer with appropriate explanation.)

This code does diverge. This should  be easy to see since threadIdx.x again runs
from 0-31 in a single warp and so the later threads in the warp must do extra work
for the extra for-loop iterations. This implies the earlier threads will be paused
and so there is warp divergence.

/// Aside:
It is also possible but unlikely that the compiler could optimize this code away
by using the fact that `pi' is const (so doesn't vary across loop iterations, reducing
the code to:

const float pi = 3.14;
float result = pow(pi, threadIdx.x);

This would not result in thread divergence if, as presumably the case, pow() performance
is independent of the exponent. (is this what is meant by the 'trick question'?)


Question 1.3: Coalesced Memory Access (9 points)
------------------------------------------------

> Let the block shape be (32, 32, 1). Let data be a (float *) pointing to global
> memory and let data be 128 byte aligned (so data % 128 == 0).
>
> Consider each of the following access patterns.
>
> (a)
> data[threadIdx.x + blockSize.x * threadIdx.y] = 1.0;
>
> Is this write coalesced? How many 128 byte cache lines does this write to?
>
> The fastest index here is threadIdx.x, which runs from 0-31. Since data is a
> (float *), we have that a warp writes 32 4-byte blocks into data. The fact
> that data is 128 byte-aligned (data % 128 == 0) means that writing into data[0]
> is the beginning of a cache line. Since 32*4 = 128, this means each warp will
> fill a single cache line. The line will be filled sequentially as threadIdx.x, so
> the write is coalesced. The access pattern will fill 1 (one) 128 byte cache line.
>
> (b)
> data[threadIdx.y + blockSize.y * threadIdx.x] = 1.0;
>
> Is this write coalesced? How many 128 byte cache lines does this write to?

This write is non-coalesced. Once again the fastest index is threadIdx.x ranging
from 0-31, and for any single warp threadIdx.y will be identical. This means that
each write to data is non-consecutive and because data is a (float *) the writes
are separated by 4*blockSize.y = 128 bytes. As a result each warp writes 4 bytes
each to 32 128-byte cache lines, one cache line per thread in the warp. This is
non-coalesced access.

> (c)
> data[1 + threadIdx.x + blockSize.x * threadIdx.y] = 1.0;

Is this write coalesced? How many 128 byte cache lines does this write to?

The situation here is much more similar to case (a) above, since threadIdx.x is the
fastest varying index in a warp and there are sequential writes to data. However
the writes to data are not coalesced because we are not minimizing the number of cache
lines access for the amount of data written (4*blockSize.x = 128 bytes per warp).
This is because the offset `1' means the writes are no longer 128-byte aligned.
Therefore a warp will begin writing at ((data + 1) % 128 == 1) and finish writing
at ((data + 1) % 128 == 1), which necessarily includes writing to 2 (two) cache lines.
Since there are two cache lines accessed per 128-bytes of written data, the access
pattern is non-coalesced.


Question 1.4: Bank Conflicts and Instruction Dependencies (15 points)
---------------------------------------------------------------------

> Let's consider multiplying a 32 x 128 matrix with a 128 x 32 element matrix.
> This outputs a 32 x 32 matrix. We'll use 32 ** 2 = 1024 threads and each thread
> will compute 1 output element. Although its not optimal, for the sake of
> simplicity let's use a single block, so grid shape = (1, 1, 1),
> block shape = (32, 32, 1).
>
> For the sake of this problem, let's assume both the left and right matrices have
> already been stored in shared memory are in column major format. This means the
> element in the ith row and jth column is accessible at lhs[i + 32 * j] for the
> left hand side and rhs[i + 128 * j] for the right hand side.
>
> This kernel will write to a variable called output stored in shared memory.
>
> Consider the following kernel code:
>
> int i = threadIdx.x;
> int j = threadIdx.y;
> for (int k = 0; k < 128; k += 2) {
>     output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
>     output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
> }
>
> (a)
> Are there bank conflicts in this code? If so, how many ways is the bank conflict
> (2-way, 4-way, etc)?

Remembering that threadIdx.x varies fastest in a block, each warp has a fixed
threadIdx.y value with threadIdx.x in [0,31], which means that i varies from
0-31 with j being fixed for each warp. Also we are interested in bank conflicts
at the instruction level so we treat k as fixed as well for the purposes of
finding simultaneous reads to a bank. If the type of output is (float *) then
we are reading 4 bytes per element.

1st line of loop: lhs[i + 32*k] reads 32 (for i) elements from different banks.
rhs[k + 128*j] will access the same address from a single bank for all threads
in the warp (k and j are fixed in the warp).

2nd line of loop: The same reasoning applies here. lhs[i + 32*(k + 1)] reads 32
elements from different banks. rhs[(k + 1) + 128 * j] again reads the same address
from a single bank for all threads in the warp.

> (b)
> Expand the inner part of the loop (below)
>
> output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
> output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
>
> into "psuedo-assembly" as was done in the coordinate addition example in lecture
> 4.
>
> There's no need to expand the indexing math, only to expand the loads, stores,
> and math. Notably, the operation a += b * c can be computed by a single
> instruction called a fused multiply add (FMA), so this can be a single
> instruction in your "psuedo-assembly".
>
> Hint: Each line should expand to 5 instructions.


Code Line 1:

0; x0 = lhs[i + 32 * k]; // load
1; y0 = rhs[k + 128 * j]; // load
2; z0 = output[i + 32 * j]; // load
3; z0 += x0 * y0; // math: FMA
4; output[i + 32 * j] = z0; // store

Code Line 2:

5; x1 = lhs[i + 32 * (k + 1)]; // load
6; y1 = rhs[(k + 1) + 128 * j]; // load
7; z1 = output[i + 32 * j]; // load
8; z1 += x1 * y1; // math: FMA
9; output[i + 32 * j] = z1; // store

> (c)
> Identify pairs of dependent instructions in your answer to part b.

* Instruction 3 relies on Instructions 0-2.
* Instruction 4 depends on Instruction 3.
* Instruction 7 depends on Instruction 4.
* Instruction 8 depends on Instruction 5-7.
* Instruction 9 depends on Instruction 8.

> (d)
> Rewrite the code given at the beginning of this problem to minimize instruction
> dependencies. You can add or delete instructions (deleting an instruction is a
> valid way to get rid of a dependency!) but each iteration of the loop must still
> process 2 values of k.

I rewrite the kernel to remove the interdependence of computation-heavy sequential
steps. While the store obviously retains its dependence on computation of x, y below,
now we can perform all loads in parallel. Note that x and y are independent now at the
instruction level.

int i = threadIdx.x;
int j = threadIdx.y;
float x, y;
for (int k = 0; k < 128; k += 2) {
    x = lhs[i + 32 * k] * rhs[k + 128 * j];
    y = lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
    output[i + 32 * j] += x + y;
}

But see also my solution to part (e) where I take this further for a more optimal
implementation to minimize all instruction dependencies.

> (e)
> Can you think of any other anything else you can do that might make this code
> run faster?
>

The first obvious thing is to directly express the parallel load I mentioned
in part (d) that the new structure allows. Specifically:

int i = threadIdx.x;
int j = threadIdx.y;
float x, y;
float x0, x1;
float y0, y1;
for (int k = 0; k < 128; k += 2) {
    x0 = lhs[i + 32 * k];
    y0 = rhs[k + 128 * j];
    x1 = lhs[i + 32 * (k + 1)];
    y1 = rhs[(k + 1) + 128 * j];
    x = x0 * y0;
    y = x1 * y1;
    output[i + 32 * j] += x + y;
}

Second, the obvious thing to do is to partially unroll the loop, provided we have
enough available registers to avoid register spilling. This is effectively the same
as enabling ILP at the loop level to remove the interdependency of instructions
on the increment `k'. Then we might increment by 4 or 8 (so we still are
processing at least two values of k, and are within the allowed bounds of the
question). I show here the case when we increment by 4, but the case of incrementing
by 8 is analogous.

int i = threadIdx.x;
int j = threadIdx.y;
float x, y, z, a;
float x0, x1;
float y0, y1;
for (int k = 0; k < 128; k += 4) {
    x0 = lhs[i + 32 * k];
    y0 = rhs[k + 128 * j];
    x1 = lhs[i + 32 * (k + 1)];
    y1 = rhs[(k + 1) + 128 * j];

    x2 = lhs[i + 32 * (k + 2)];
    y2 = rhs[(k + 2) + 128 * j];
    x3 = lhs[i + 32 * (k + 3)];
    y3 = rhs[(k + 3) + 128 * j];

    x = x0 * y0;
    y = x1 * y1;
    z = x2 * y2;
    a = x3 * y3;

    output[i + 32 * j] += x + y + z + a;
}


PART 2 - Matrix transpose optimization (65 points)
--------------------------------------------------
>
> Optimize the CUDA matrix transpose implementations in transpose_cuda.cu. Read
> ALL of the TODO comments. Matrix transpose is a common exercise in GPU
> optimization, so do not search for existing GPU matrix transpose code on the
> internet.
>
> Your transpose code only need to be able to transpose square matrices where the
> side length is a multiple of 64.
>
> The initial implementation has each block of 1024 threads handle a 64x64 block
> of the matrix, but you can change anything about the kernel if it helps obtain
> better performance.
>
> The main method of transpose.cc already checks for correctness for all transpose
> results, so there should be an assertion failure if your kernel produces incorrect
> output.
>
> The purpose of the shmemTransposeKernel is to demonstrate proper usage of global
> and shared memory. The optimalTransposeKernel should be built on top of
> shmemTransposeKernel and should incorporate any "tricks" such as ILP, loop
> unrolling, vectorized IO, etc that have been discussed in class.

[Haru:tanderson]~/lab2>./transpose
Index of the GPU with the lowest temperature: 2 (43 C)
Time limit for this program set to 10 seconds
Size 512 naive CPU: 0.003712 ms
Size 512 GPU memcpy: 0.034624 ms
Size 512 naive GPU: 0.097312 ms
Size 512 shmem GPU: 0.032320 ms
Size 512 optimal GPU: 0.030432 ms

Size 1024 naive CPU: 1.146944 ms
Size 1024 GPU memcpy: 0.085472 ms
Size 1024 naive GPU: 0.317952 ms
Size 1024 shmem GPU: 0.091552 ms
Size 1024 optimal GPU: 0.086144 ms

Size 2048 naive CPU: 31.870625 ms
Size 2048 GPU memcpy: 0.267264 ms
Size 2048 naive GPU: 1.153856 ms
Size 2048 shmem GPU: 0.319424 ms
Size 2048 optimal GPU: 0.307264 ms

Size 4096 naive CPU: 152.841156 ms
Size 4096 GPU memcpy: 1.005760 ms
Size 4096 naive GPU: 4.127392 ms
Size 4096 shmem GPU: 1.216288 ms
Size 4096 optimal GPU: 1.162016 ms

BONUS (+5 points, maximum set score is 100 even with bonus)
--------------------------------------------------------------------------------

> Mathematical scripting environments such as Matlab or Python + Numpy often
> encourage expressing algorithms in terms of vector operations because they offer
> a convenient and performant interface. For instance, one can add 2 n-component
> vectors (a and b) in Numpy with c = a + b.
>
> This is often implemented with something like the following code:
>
> void vec_add(float *left, float *right, float *out, int size) {
>     for (int i = 0; i < size; i++)
>         out[i] = left[i] + right[i];
> }
>
> Consider the code
>
> a = x + y + z
>
> where x, y, z are n-component vectors.
>
> One way this could be computed would be
>
> vec_add(x, y, a, n);
> vec_add(a, z, a, n);
>
> In what ways is this code (2 calls to vec_add) worse than the following?
>
> for (int i = 0; i < n; i++)
>   a[i] = x[i] + y[i] + z[i];
>
> List at least 2 ways (you don't need more than a sentence or two for each way).

1. The second call to vec_add requires the output of the first vec_add to have been
completed and stored in `a', which means that ILP is impossible. Compared to the
second piece of code where this is easily handled. In fact the second code can be
loop unrolled as well while only the individual components can be in the first.

2. There is overhead to function calls, and inlining the code directly will eliminate
that burden, at the cost of less easily-maintained and understood code.

3. There are twice as many stores in the two calls to vec_add as there are in the
second code example.
