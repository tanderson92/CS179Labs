* See the commented code, in classify.cc and classify_cuda.cu{,h}. Analysis
  follows from here (all tests run on Haru).

Portion of runtime spent on I/O and parsing data:
--------------------------------------------------------------------------------
To do this we measure the amount of time the entire classification
program takes, and compare to the time spent when not running the
kernel on the streamed data. We still call readLSAReview(). The
wall times are (batch size of 2048 here -- a moderate batch size
to be representative; see the full table below):
    Total time: 17.2s
    I/O Parsing time: 16.0s

That is, about 93% of the time is spent on I/O with slightly less
than 7% in the kernel.

Latency timing:
--------------------------------------------------------------------------------

Two cudaEvents were created and the cost of a single call to the kernel was
measured to determine the latency. Note that when the batch size is `1', the
classification does not run through all batches before being terminated due
to excess runtime. We can compute the throughput by dividing the batch size
by the latency.

Batch size :  Latency:    Total Time:       I/O Parsing Time:
    - 00001:    0.11ms,          N/A                     N/A
    - 00032;    0.28ms,         28.0s,                  16.9s
    - 01024;    0.90ms,         17.9s,                  16.0s
    - 02048;    0.90ms,         17.2s,                  16.0s
    - 16384:    2.22ms,         16.6s,                  16.0s
    - 65536:    6.12ms,         16.5s,                  16.3s

Batch size : Throughput:
    - 00001:       9.1 floats/ms
    - 00032:     114.3 floats/ms
    - 01024:    1137.8 floats/ms
    - 02048:    2275.6 floats/ms
    - 16384:    7380.2 floats/ms
    - 65536:   10708.5 floats/ms

