#include <iostream>
#include <math.h>
#include <string>

__global__ void add(int n, float *x, float *y)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
    {
        y[i] += x[i];
    }
}

__host__ void init_arrs(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
}

int main()
{
    // a million elts
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y;

    // the only reason we're here is to be serious.
    // don't use unified memory :p
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));
    init_arrs(N, x, y);

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    add<<<numBlocks, threadsPerBlock>>>(N, d_x, d_y);

    // can't access d_y's memory from host, so we have to copy it over
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = max(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    free(x);
    cudaFree(d_x);
    free(y);
    cudaFree(d_y);

    return 0;
}