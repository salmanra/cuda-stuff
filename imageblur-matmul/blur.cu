#include <iostream>
#include <math.h>
#include <string>

// will this var even be accessible from a kernel?
const int CHANNELS = 3;

__global__ void rgbToGrayScale(unsigned char *Pout, unsigned char *Pin, int width, int height)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < height)
    {
        int grayscaleIdx = row * width + col;
        int rgbIdx = grayscaleIdx * CHANNELS;

        unsigned char r = Pin[rgbIdx];
        unsigned char g = Pin[rgbIdx + 1];
        unsigned char b = Pin[rgbIdx + 2];

        Pout[grayscaleIdx] = 0.21f * r + 0.72f * g + 0.07f * b;
    }
}

__global__ void init_rgb(unsigned char *rgbImg, int m, int n)
{

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < n && row < m)
    {
        for (int k = 0; k < CHANNELS; k++)
        {
            rgbImg[(col + row * n) * CHANNELS + k] = (col % 4 + row % 3 + k) * 10; // i see. a channel's value is exactly one byte.
        }
    }
}

int main()
{
    int n = 76;
    int m = 62;
    unsigned char *rgbImg;
    unsigned char *grayscaleImg; // all elts init to 0

    // don't even need this sizeof(unsigned char) bc it is equal to 1 byte
    cudaMalloc(&rgbImg, n * m * CHANNELS * sizeof(unsigned char));
    cudaMalloc(&grayscaleImg, n * m * sizeof(unsigned char));

    int blockLength = 32;
    dim3 blockShape(blockLength, blockLength, 1);
    dim3 gridShape((n + blockLength - 1) / blockLength, (m + blockLength - 1) / blockLength, 1);

    init_rgb<<<gridShape, blockShape>>>(rgbImg, m, n);

    rgbToGrayScale<<<gridShape, blockShape>>>(grayscaleImg, rgbImg, n, m);

    // gnuplot grayscaleImg

    cudaFree(rgbImg);
    cudaFree(grayscaleImg);
}