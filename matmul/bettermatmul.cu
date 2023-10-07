#include <string>
#include <math.h>
#include <iostream>

constexpr int TILE_WIDTH = 32;
constexpr int COARSE_FACTOR = 4;
struct Matrix
{
    int w;
    int h;
    float *elts;
};

void initMat(Matrix M)
{
    for (int col = 0; col < M.w; col++)
    {
        for (int row = 0; row < M.h; row++)
        {
            M.elts[row * M.w + col] = 3.0f;
        }
    }
}

void naiveMatmul(Matrix M, Matrix N, Matrix P)
{
    for (int col = 0; col < P.w; col++)
    {
        for (int row = 0; row < P.h; row++)
        {
            float outVal = 0.0f;
            for (int ph = 0; ph < M.w; ph++)
            {
                outVal += M.elts[row * M.w + ph] * N.elts[ph * N.w + col];
            }
            P.elts[row * P.w + col] = outVal;
        }
    }
}

__global__ void matmul(Matrix M, Matrix N, Matrix P)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= P.h || col >= P.w)
        return;

    float outVal = 0.0f;
    for (int ph = 0; ph < M.w; ph++)
    {
        outVal += M.elts[row * M.w + ph] * N.elts[ph * N.w + col];
    }
    P.elts[row * P.w + col] = outVal;
}

// assumes blocks are TILE_WIDTHxTILE_WIDTH
__global__ void tiledMatmul(Matrix M, Matrix N, Matrix P)
{
    __shared__ float M_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_tile[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_WIDTH + tx;
    int row = blockIdx.y * TILE_WIDTH + ty;

    float outVal = 0.0f;
    for (int ph = 0; ph < ceil(M.w / (float)TILE_WIDTH); ph++)
    {
        int M_col = ph * TILE_WIDTH + tx;
        int N_row = ph * TILE_WIDTH + ty;
        if (row < M.h && M_col < M.w)
        {
            M_tile[ty][tx] = M.elts[row * M.w + M_col];
        }
        else
        {
            M_tile[ty][tx] = 0;
        }
        if (col < N.w && N_row < N.h)
        {
            N_tile[ty][tx] = N.elts[N_row * N.w + col];
        }
        else
        {
            N_tile[ty][tx] = 0;
        }
        __syncthreads();

        for (int idx = 0; idx < TILE_WIDTH; idx++)
        {
            outVal += M_tile[ty][idx] * N_tile[idx][tx];
        }
        __syncthreads();
    }
    if (row < P.h && col < P.w)
        P.elts[row * P.w + col] = outVal;
}

__global__ void coarseTiledMatmul(Matrix M, Matrix N, Matrix P)
{
    __shared__ float M_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_tile[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int startCol = COARSE_FACTOR * TILE_WIDTH * blockIdx.x + tx;
    int row = TILE_WIDTH * blockIdx.y + ty;

    // one thread calcs COARSE_FACTOR output values
    float outVals[COARSE_FACTOR];
    for (int i = 0; i < COARSE_FACTOR; i++)
    {
        outVals[i] = 0.0f;
    }
    for (int p = 0; p < (M.w + TILE_WIDTH - 1) / TILE_WIDTH; p++)
    {
        // load a tile of M
        int M_col = p * TILE_WIDTH + tx;
        if (row < M.h && M_col < M.w)
        {
            M_tile[ty][tx] = M.elts[row * M.w + M_col];
        }
        else
        {
            M_tile[ty][tx] = 0.0f;
        }
        // "reuse" a tile of M COARSE_FACTOR times
        for (int q = 0; q < COARSE_FACTOR; q++)
        {
            // load a tile of N
            int N_row = p * TILE_WIDTH + ty;
            int col = startCol + q * TILE_WIDTH;
            if (N_row < N.h && col < N.w)
            {
                N_tile[ty][tx] = N.elts[N_row * N.w + col];
            }
            else
            {
                N_tile[ty][tx] = 0.0f;
            }

            __syncthreads();

            // matmul
            for (int j = 0; j < TILE_WIDTH; j++)
            {
                outVals[q] += M_tile[ty][j] * N_tile[j][tx];
            }
            __syncthreads();
        }
    }
    for (int i = 0; i < COARSE_FACTOR; i++)
    {
        if (row < P.h && (startCol + i * TILE_WIDTH) < P.w)
            P.elts[row * P.w + startCol + i * TILE_WIDTH] = outVals[i];
    }
}

int main()
{

    int i, j, k;

    i = 320;
    j = 322;
    k = 500;

    Matrix M, N, P, d_M, d_N, d_P;
    M.h = i;
    M.w = j;
    M.elts = (float *)malloc(i * j * sizeof(float));

    N.h = j;
    N.w = k;
    N.elts = (float *)malloc(j * k * sizeof(float));

    P.h = i;
    P.w = k;
    P.elts = (float *)malloc(i * k * sizeof(float));

    initMat(M);
    initMat(N);
    initMat(P);

    naiveMatmul(M, N, P);

    d_M.h = M.h;
    d_M.w = M.w;
    cudaMalloc(&d_M.elts, d_M.w * d_M.h * sizeof(float));

    d_N.h = N.h;
    d_N.w = N.w;
    cudaMalloc(&d_N.elts, d_N.h * d_N.w * sizeof(float));

    d_P.h = P.h;
    d_P.w = P.w;
    cudaMalloc(&d_P.elts, d_P.h * d_P.w * sizeof(float));

    cudaMemcpy(d_M.elts, M.elts, M.w * M.h * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N.elts, N.elts, N.w * N.h * sizeof(float), cudaMemcpyHostToDevice);

    int blockWidth = TILE_WIDTH;
    dim3 threadsPerBlock = dim3(blockWidth, blockWidth);
    dim3 gridShape = dim3((d_P.w + blockWidth - 1) / blockWidth, (d_P.h + blockWidth - 1) / blockWidth);
    coarseTiledMatmul<<<gridShape, threadsPerBlock>>>(d_M, d_N, d_P);

    std::cout << "grid x: " << gridShape.x << " grid y: " << gridShape.y << std::endl;
    std::cout << "block x: " << threadsPerBlock.x << " block y: " << threadsPerBlock.y << std::endl;

    cudaDeviceSynchronize();

    Matrix res;
    res.h = d_P.h;
    res.w = d_P.w;
    res.elts = (float *)malloc(res.h * res.w * sizeof(float));
    cudaMemcpy(res.elts, d_P.elts, d_P.w * d_P.h * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    int rescount = 0;
    for (int u = 0; u < P.h; u++)
    {
        for (int w = 0; w < P.w; w++)
        {
            maxError = max(maxError, abs(P.elts[u * P.w + w] - res.elts[u * res.w + w]));
            if (abs(P.elts[u * P.w + w] - res.elts[u * res.w + w]) > 0.0000001)
            {
                rescount++;
                std::cout << res.elts[u * res.w + w] << ' ' << P.elts[u * P.w + w] << std::endl;
            }
        }
    }
    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "Bad Res: " << rescount << std::endl;

    free(M.elts);
    free(N.elts);
    free(P.elts);
    free(res.elts);

    cudaFree(d_M.elts);
    cudaFree(d_N.elts);
    cudaFree(d_P.elts);
}