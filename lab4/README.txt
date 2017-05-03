1.1
--------------------

In general this can be an effective strategy. The reason being that data
stored in shared memory is accessible by every thread in the block so
threads may not have to re-access the global memory. This is advantageous
because shared memory is much faster than global memory. Reads and writes
to global memory can then be batched by block, in this manner.

That being said, it's unlikely to be of much use for BFS: there is not the
reuse per block that is mentioned in the preceding paragraph.

Referring to the GPU-side kernel pseudocode presented in lecture, note that
X is read perhaps several times and C / F are written to perhaps several times:
but this __only__ occurs if two or more vertices per frontier under consideration
share a neighbor. This would be challenging to navigate in a programmatic sense
at the block-level (where we are considering doing this for shared memory
purposes) since there is a level of indirection between the threadID and the
for-loop counter. And in an algorithmic sense there could arise a situation where
a frontier contains two or more vertices which have an edge to every vertex in the graph.
This is a nightmare to work with at block scope.


1.2
--------------------

First: for obvious reasons one should prefer bitwise-or to summation; the logic
is the same as below but perhaps is faster than summing integers, or longs.
But the actual sum's value is irrelevant in this question since we throw away
all that information after comparing to 0.

If we store "false" as 0 and "true" as 1 within F, then the array's sum
being nonzero is equivalent to F not being all false. Therefore, the problem
reduces to that of computing the sum of F. But this is a solved problem (see
last assignment switching max() for +=) and so we use parallel reduction to
compute the global sum of F. We can reduce in parallel within each block
and then at the last step perform an atomic add across blocks. Then the result
is compared against 0 and a decision is reached.

1.3
--------------------

Store a bool in global memory (pinned to CPU memory?). Set the bool to 0.
Then each thread executes as normal except that if F[j] is set then the
global bool is written to and set as 1. There is no race condition issue
since CUDA ensures that all writes go through. Even if it didn't though,
a write only occurs if it is being set to 1, so we would be fine, for the
same reason one only cares about bitwise-or vs. summing.

This may be slower than the suggestion in 1.2 if the graph is dense. Even
though the solution in 1.2 uses atomic sums, the solution presented here
involves a write for every element added to the F[], the frontier. This
is quite slow, so for dense graphs we would expect a penalty.

2
--------------------

I expect a GPU-accelerated PET reconstruction performance to yield significantly
less speedup than the GPU-accelerate CT reconstruction did. There are several
reasons for this:

1) By measuring emission from a single point, one loses the spatial locality of
CT sinogram computations; from a computational perspective this implies that
texture memory (which I did not complete here due to a bizarre compiler error)
and the interpolatory functions it provides will be of no use to us. Texture
memory is faster than global memory, of course, so is one source of degraded
relative performance.

2) Because our input file is a list over time, it is entirely possible that
multiple detected emissions will result from the same source location. This
is a stochastic process so it is possible this won't happen but we can quantify
in expectation the extent to which this will occur. In any event, this calls
attention the need for atomic operations (atomic sum of contributions) since
parallelization over the the temporal observations may result in race
conditions on the reconstructed tomography pixels.

3) It is also far from clear that an efficient high-pass filter can be performed
on the data, though more details about the PET reconstruction process are needed.
One does not have the final sinogram data up-front in the way that an X-ray CT
does, so applying the high-pass filter is likely not possible. Of course, other
techniques are perhaps (likely!) possible, and would be needed for the same
reasons as in CT.

3
--------------------

See xray_ct_cuda.cu{,h} and xray_ct_recon.cc with associated Makefile.
All comments are in the routine. Note that I used batched in-place R2C/C2R
transforms for added efficiency, which changed some of the index computations
in the high-pass filter from what a C2C transform would have.
