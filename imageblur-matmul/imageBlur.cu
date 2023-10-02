#include <iostream>
#include <math.h>
#include <string>

int BLUR_SIZE;
// we're assuming grayscale image, it looks like
__global__ void imageBlur(unsigned char *img, unsigned char *out, int h, int w)
{
    // img -> 3D char*

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < h && col < w) // this thread corresponds to a valid output pixel
    {
        int pixelVal = 0;
        int pixels = 0;
        for (int i = row - BLUR_SIZE; i <= row + BLUR_SIZE; i++)
        {
            for (int j = col - BLUR_SIZE; j <= col + BLUR_SIZE; j++)
            {
                if (j < 0 || j >= w || i < 0 || i >= h)
                {
                    continue;
                }
                pixelVal += img[i * w + j];
                pixels++;
            }
        }
        out[row * w + col] = (unsigned char)pixelVal / pixels;
    }
}