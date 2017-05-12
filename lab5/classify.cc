#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "classify_cuda.cuh"
#include "ta_utilities.hpp"

using namespace std;

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code,
    const char *file,
    int line,
    bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
    gpuErrChk(cudaEventCreate(&start));         \
    gpuErrChk(cudaEventCreate(&stop));          \
    gpuErrChk(cudaEventRecord(start));          \
}

#define STOP_RECORD_TIMER(name) {                           \
    gpuErrChk(cudaEventRecord(stop));                       \
    gpuErrChk(cudaEventSynchronize(stop));                  \
    gpuErrChk(cudaEventElapsedTime(&name, start, stop));    \
    gpuErrChk(cudaEventDestroy(start));                     \
    gpuErrChk(cudaEventDestroy(stop));                      \
}

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

// Fills output with standard normal data
void gaussianFill(float *output, int size) {
    // seed generator to 2015
    std::default_random_engine generator(2015);
    std::normal_distribution<float> distribution(0.0, 0.1);
    for (int i=0; i < size; i++) {
        output[i] = distribution(generator);
    }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM + 1 floats.
void readLSAReview(string review_str, float *output, int stride) {
    stringstream stream(review_str);
    int component_idx = 0;

    for (string component; getline(stream, component, ','); component_idx++) {
        output[stride * component_idx] = atof(component.c_str());
    }
    assert(component_idx == REVIEW_DIM + 1);
}

void classify(istream& in_stream, int batch_size) {
    // Randomly initialize weights. allocate and initialize buffers on
    //       host & device. allocate and initialize streams

    // Allocate memory
    size_t weights_len = REVIEW_DIM * sizeof(float);
    size_t data_len = (REVIEW_DIM + 1) * batch_size * sizeof(float);

    float *host_weights = (float *) malloc(weights_len);
    float *dev_weights;

    float *host0_data = (float *) malloc(data_len);
    float *host1_data = (float *) malloc(data_len);
    float *dev0_data, *dev1_data;

    gpuErrChk(cudaMalloc( (void **) &dev_weights, weights_len));
    gpuErrChk(cudaMalloc( (void **) &dev0_data, data_len));
    gpuErrChk(cudaMalloc( (void **) &dev1_data, data_len));

    // Randomly initialize weights
    gaussianFill(host_weights, REVIEW_DIM);

    // Move initialized weights to GPU
    gpuErrChk(cudaMemcpy(dev_weights, host_weights, weights_len, cudaMemcpyHostToDevice));

    // Allocate and initialize streams, use two streams for async
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Utility pointer arrays for streaming I/O for each stream
    // When adding more streams, allocate devX_data, hostX_data with data_len
    // bytes and change the length of devs_data, hosts_data. Also
    // modify the # of streams declared and created above
    float *devs_data[2];
    devs_data[0] = dev0_data;
    devs_data[1] = dev1_data;

    float *hosts_data[2];
    hosts_data[0] = host0_data;
    hosts_data[1] = host1_data;

    cudaEvent_t event_latency_Start, event_latency_Stop;
    cudaEventCreate(&event_latency_Start);
    cudaEventCreate(&event_latency_Stop);

    // The time for kernel to execute on data
    float latency = -1;
    bool latency_flag = true; // only run the latency timer once

    // index identifying current stream, since we wrote this to currently
    // have only two streams we will identify by 0, 1 and flip between
    // (bitwise xor works instead of annoying if-else statements!)
    int stream_idx = 0;

    // main loop to process input lines (each line corresponds to a review)
    int review_idx = 0;
    int batch_counter = 0;
    for (string review_str; getline(in_stream, review_str); review_idx++) {
        // current stream pointers
        float *cur_dev_data = devs_data[stream_idx];
        float *cur_host_data = hosts_data[stream_idx];
        // process review_str with readLSAReview, unit stride data
        readLSAReview(review_str, cur_host_data + review_idx*(REVIEW_DIM + 1), 1);

        // If we have filled up a batch, copy H->D, call kernel and copy
        //      D->H all in a stream

        // NOTE: We do *not* copy D->H per-stream, this seems unneeded; all
        // weights are synced D->H after the main loop is terminated

        // We have filled a batch when review_idx compares to batch_size
        // (but review_idx is 0-indexed so offset by 1)
        if (review_idx + 1 == batch_size) {
            // H->D async copy, into stream_idx ^ 1; notice staggered execution
            gpuErrChk(cudaMemcpyAsync(cur_dev_data, cur_host_data, data_len,
                cudaMemcpyHostToDevice, streams[stream_idx ^ 1]));
            // We need to update the weights after each batch (but not within!),
            // so synchronize the streams and execute the kernel.
            gpuErrChk(cudaStreamSynchronize(streams[stream_idx]));
            if (latency_flag) cudaEventRecord(event_latency_Start);
            // Call the classify kernel on the stream
            float error = cudaClassify(cur_dev_data, batch_size, 1.0, dev_weights,
                streams[stream_idx]);
            // stub for when suppressing kernel execution.
            //float error = 0;
            if (latency_flag) {
                cudaEventRecord(event_latency_Stop);
                cudaEventSynchronize(event_latency_Start);
                cudaEventSynchronize(event_latency_Stop);
                cudaEventElapsedTime(&latency, event_latency_Start, event_latency_Stop);
                cudaEventDestroy(event_latency_Start);
                cudaEventDestroy(event_latency_Stop);
                latency_flag = false;
                // Needed for small batch sizes because one exceeds max time limit
                //printf("Latency of single batch: %f (ms)\n", latency);
                //exit(-1);
            }
            // (bit)flip to other stream now that we've copied the data and
            // started the kernel on the stream_idx stream.
            stream_idx = stream_idx ^ 1;
            review_idx = -1;

            printf("Batch #%d Misclassification (%%): %f \n", batch_counter++, 100.0 * error);
        }

    }
    printf("made it here\n");

    cudaStreamSynchronize(streams[0]);
    cudaStreamDestroy(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamDestroy(streams[1]);

    // print out weights (& copy to host since we avoided that above)
    gpuErrChk(cudaMemcpy(host_weights, dev_weights, weights_len, cudaMemcpyDeviceToHost));
    for (int i = 0; i < REVIEW_DIM; i++) {
        printf("Weight idx %d: %f\n", i + 1, host_weights[i]);
    }

    printf("Latency of single batch: %f (ms)\n", latency);

    // free all memory
    free(host0_data);
    free(host1_data);
    free(host_weights);
    cudaFree(dev0_data);
    cudaFree(dev1_data);
    cudaFree(dev_weights);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("./classify <path to datafile>\n");
        return -1;
    }
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 100;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    // Init timing
    float time_initial, time_final;

    //int batch_size = 1;
    int batch_size = 32;
    //int batch_size = 1024;
    //int batch_size = 2048;
    //int batch_size = 16384;
    //int batch_size = 65536;

    // begin timer
    time_initial = clock();

    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    classify(buffer, batch_size);

    // End timer
    time_final = clock();
    printf("Total time to run classify: %f (s)\n", (time_final - time_initial) / CLOCKS_PER_SEC);


}
