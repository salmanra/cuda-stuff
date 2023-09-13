#include <iostream>
#include <math.h>
#include <string.h>

// function to add the elements of two arrays
void add(int N, float *x, float *y)
{
    for (int i = 0; i < N; i++)
    {
        y[i] += x[i];
    }
}

int main(void)
{
    int N = 1 << 5; // 1M elts

    float *x = new float[N];
    float *y = new float[N];

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0;
        y[i] = 2.0;
    }

    add(N, x, y);
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    delete[] x;
    delete[] y;

    return 0;
}