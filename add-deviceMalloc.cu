#include <iostream>
#include <math.h>
#include <string>

// maybe this is uninteresting, but when you malloc directly to the device,
// you save a LOT of time. it's uninteresting only because we might want to
// start with the assumption that our arrays start on the device

__global__ void add(int n, float *x, float *y)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
    {
        y[i] += x[i];
    }
}

__global__ void init_arrs(int n, float *x, float *y)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
}

int main()
{
    float *x, *y, *res;
    int N = 1 << 20;

    // gonna need to load this onto the host after the computation
    res = (float *)malloc(N * sizeof(float));
    cudaMalloc(&x, N * sizeof(float));
    cudaMalloc(&y, N * sizeof(float));

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    init_arrs<<<numBlocks, threadsPerBlock>>>(N, x, y);

    add<<<numBlocks, threadsPerBlock>>>(N, x, y);

    cudaMemcpy(res, y, N * sizeof(float), cudaMemcpyDeviceToHost);
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = max(maxError, fabs(res[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    free(res);
    cudaFree(x);
    cudaFree(y);

    return 0;
}