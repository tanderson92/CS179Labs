#ifndef GILLESPIE_CUDA_CUH
#define GILLESPIE_CUDA_CUH

// Whether the system is on or off
enum SysStat { On, Off };

#define SAMPLE_SIZE 1000
#define SAMPLE_T    100

#define B         10.0
#define G         1.0
#define KON       0.1
#define KOFF      0.9

// 2.1     Gillespie timestep implementation (25 pts)
void GillespieCallTimestepKernel(const unsigned int blocks,
    const unsigned int threadsPerBlock, const float *rndReactions,
    const float *rndTimes, float *simTimes, float *simConcentrations,
    SysStat *simStates, const unsigned int nSims);

// 2.2     Data resampling and stopping condition (25 pts)
void GillespieCallResampleKernel( const unsigned nBlocks,
    const unsigned nThreads, int *prevSampleInd, float *simSamples,
    const float *simTimes, float *simMinTimes, float *min_time,
    const float *simConcentrations, int * completed, const unsigned nSims);

// 2.3a    Calculation of system mean (10 pts)
void GillespieCallMeanKernel(const unsigned nBlocks,
    const unsigned nThreads, const unsigned i, float *simSamples, float *sample_mean_dev,
    const unsigned nSims);

// 2.3b    Calculation of system varience (10 pts)
void GillespieCallVarianceKernel(const unsigned nBlocks,
    const unsigned nThreads, const unsigned i, float *simSamples, float *sample_var_dev,
    float *sample_mean_dev, const unsigned nSims);

#endif // GILLESPIE_CUDA_CUH
