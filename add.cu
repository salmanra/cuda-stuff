#define cudaDeviceScheduleBlockingSync 0x04

#include <iostream>
#include <math.h>
#include <string.h>

// we use nvcc to compile .cu files
// we use nsight compute to profile executables?

// it is time to put this on github. even vector addition is rich
// enough to present multiple enlightening implementations (even
// if none optimize the other)

// kernel to add the elements of two arrays
__global__ void add(int N, float *x, float *y)
{
    // grid-stride iteration
    // now let's try it with half as many threads
    int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x); // start all across the grid
    // int stride = blockDim.x * gridDim.x;               // jump over the entire grid
    // here's the fun part: since we set the number of threads to be equal to N,
    // stride will always be equal to N, and each thread only needs to execute one addition.
    // let's try it.
    // last detail is to check the bound. applies whenever N is not a multiple of blockDim
    // for (int i = index; i < N; i += stride)
    // {
    //     y[i] += x[i];
    // }
    if (index < N - 1)
    {
        y[index] += x[index];
        y[index + 1] += x[index + 1];
    }
}

__global__ void init_arrs(int N, float *x, float *y)
{
    // why not use a grid-stride loop?
    // because we don't need to: we tend to call this
    // with as many threads as elements N
    int index = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
    // int stride = blockDim.x * gridDim.x;
    x[index] = 1.0;
    x[index + 1] = 1.0;
    y[index] = 2.0;
    y[index + 1] = 2.0;
}

int main(void)
{

    // maxing unified data cache to Shared Memory gives better speed over maxing to L1 cache,
    // and is ever so slightly better than the default (1.2 ms, 670 us, 682 us).
    // SM throughput is the same (2%) for both default and shared mem options
    //
    // just kidding! there's some non-determinism going on here!!!
    // the MaxShared, run it three times and you'll get three different SM throughputs ranging from 1-2%
    //
    // my mental model was that the unified data cache would hold the entries to the two vectors x and y,
    // and that the low SM throughput is due to these vectors being too large (~2MB) to fit in the ud cache.
    // and maybe the vecs are only loaded into the L1 portion of the ud cache, or only the shared mem portion,
    // and that maxing one or the other would give better performance. But performance looks independent of
    // ud cache carbeout!!
    //
    // fact: all (yes all) memory operations were between kernel and system memory, meaning all the stores and
    // loads to x and y were happening on CPU. every kernel had to load x and y from CPU memory, not GPU memory.
    // if we initialize x and y on device (ie, init them in a kernel), we should get around this and see different
    // results
    // cudaFuncSetAttribute(add, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);

    int N = 1 << 20; // a bunch of elts

    // we specify device at the time of memory allocation
    // bahahaha!!! cMM is like vm_map but even less committal: it doesn't even create PTEs for
    // these arrays. once they're written/accessed, then the device and the pte are chosen
    float *x, *y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize the arrays on host (here)
    // for (int i = 0; i < N; i++)
    // {
    //     x[i] = 1.0;
    //     y[i] = 2.0;
    // }

    // we invoke kernels (cuda functions as marked by "__global__") with <<<>>> notation that run on devices, not host
    int threadsPerBlock = 256; // is a multiple of 32, meaning the SM can split it evenly into warps

    // say we halve the number of threads.
    // then each block computes an addition for 2*threadsPerBlock threads,
    // so we need this many blocks
    int numBlocks = (N + 2 * threadsPerBlock - 1) / (2 * threadsPerBlock); // what a stupid bug

    // std::cout << "N: " << N << "\ngridDim.x*blockDim.x: " << numBlocks * threadsPerBlock << std::endl;
    // kernel call! initialize arrays on device
    init_arrs<<<numBlocks, threadsPerBlock>>>(N, x, y);

    // add will find both arrays on the device and not suffer any page faults
    add<<<numBlocks, threadsPerBlock>>>(N, x, y);

    // we join on all the kernel threads
    // well that doesn't work the way I expected...
    cudaDeviceSynchronize();

    // std::cout << "synced to host" << std::endl;

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // we free memory according to device
    cudaFree(x);
    cudaFree(y);

    return 0;
}