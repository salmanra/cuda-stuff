#include <iostream>
#include <math.h>
#include <string>
#include <curand.h>
// #include <thrust/fill.h>

constexpr int TILE_WIDTH = 32; // want this to be geq (?) to block-dim
constexpr int COARSE_FACTOR = 4;
__global__ void cornerTurnMatMul(float *A, float *B, float *C)
{
}

// this is definitely incorrect on the boundary checks.
__global__ void CoarseTiledMatMul(float *A, float *B, float *C, int A_width, int A_height, int B_width)
{
    // One block is responsible for COARSE_FACTOR tiles of output.
    // each thread is responsible for COARSE_FACTOR elts of output.
    // this is in the direction of output columns.
    //
    // "While I have this tile of A loaded, let's handle the computation of many tiles of B"

    // 1. Declare tiles, get row and col indices for this thread.
    // 2. Declare array of COARSE_FACTOR output values.
    // 3. For as many tiles as it takes to cover a row of A
    //     a. Load A into shared memory.
    //     b. for COARSE_FACTOR times:
    //         i. Load B into shared memory. this moves by a tile each iter
    //         ii. matmul loaded tiles. each thread now saves COARSE_FACTOR outputs
    // 4. write all the output values to memory
    //

    // 1.
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // TILE_WIDTH is equal to block width.
    // each thread is responsible for COARSE_FACTOR consec rows.
    // Thus indexing is as follows:
    int COL_START = COARSE_FACTOR * TILE_WIDTH * blockIdx.x + tx;
    int ROW = TILE_WIDTH * blockIdx.y + ty;

    // has this scheme lost the plot?
    // this array is going straight to DRAM...
    // 2.
    float outVals[COARSE_FACTOR];
    for (int i = 0; i < COARSE_FACTOR; i++)
    {
        outVals[i] = 0.0f;
    }

    // 3.
    for (int i = 0; i < A_width / TILE_WIDTH; i++)
    {
        // a.
        int A_col = i * TILE_WIDTH + tx;
        bool A_idx_valid = (A_col < A_width && ROW < A_height);
        if (A_idx_valid)
            A_tile[ty][tx] = A[ROW * A_width + A_col];

        // b.
        for (int j = 0; j < COARSE_FACTOR; j++)
        {
            // i.
            int B_col = j * TILE_WIDTH + COL_START;
            int B_row = i * TILE_WIDTH + ty;
            bool B_idx_valid = (B_col < B_width && B_row < A_width);
            if (B_idx_valid)
                B_tile[ty][tx] = B[B_row * B_width + B_col];

            __syncthreads(); // don't want to use B_tile before we've written to it

            // ii.
            if (A_idx_valid && B_idx_valid)
            {
                for (int k = 0; k < TILE_WIDTH; k++)
                {
                    outVals[j] += A_tile[ty][k] + B_tile[k][tx];
                }
            }
            __syncthreads(); // don't want to overwrite B_tile while we're still reading B_tile
        }
    }
    for (int i = 0; i < COARSE_FACTOR; i++)
    {
        int C_col = COL_START + i * TILE_WIDTH;
        if (C_col < B_width && ROW < A_height)
            C[ROW * B_width + C_col] = outVals[i];
    }
}

// actually, what is the tiledMatMul impl?
// a 1D grid of blocks that "sweep" over the input arrays?
// a 2D grid of blocks that each matmul a single tile from each input array?
// let this be a square matrix. I wonder does the optim alg for non-square matmul
// tile it into squares anyway? Probably that's the best way to tile.
// assuming tiling is the optimal alg, then you never really do non-square matmul
__global__ void tiledMatMul(float *A, float *B, float *C, int A_height, int A_width, int B_width)
{
    // 1. declare shared mem tiles (one for each input matrix)
    // 2. get row and col of output matrix for this thread
    // 3. For as many tiles as it takes to cover the input matrices (wym by that?)
    //    a. load a single tile elt for each of the two tiles (collaboratively load the tiles)
    //    b. sync on all threads in the block (which is equiv to tile)
    //    c. accum the value of output at row,col into a register
    // 4. store accumulated val into output matrix
    // there is a "redundancy" in how many tiles are loaded to L1 in total
    // each block will end up tiling an entire row of A and col of C.
    // there is some parallelism that we miss out on here methinks
    // because each row of A will get loaded into L1 cache by width/tile_width
    // blocks and same for each col of B. maybe this is the only way, or maybe
    // we can exploit the fact that different blocks actually load in the same
    // tiles as one another. just a thought. part of the premise of cuda is
    // there is no guarantee on synchronicity between blocks. so there could
    // be a way to do it by expanding what a block does (but then it's not so parallel, right?)

    // i'm having an interesting moment looking at the code in the book
    // i think i need to spell out the algorithm in english before
    // internalizing what even in tiledMatMul

    // yo. in 1D blocks, warps are made of threads with consecutive threadIdx.x
    // what about in 2D blocks? "linearized row major layout" -> consecutive threadIdx.x.

    // 1.
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output indices of this thread
    // 2.
    int COL = TILE_WIDTH * blockIdx.x + tx;
    int ROW = TILE_WIDTH * blockIdx.y + ty;
    if (ROW >= A_height || COL >= B_width)
    {
        return;
    }
    // let the tile dimensions be the same as the block dimensions.
    //
    float outVal = 0.0;
    // 3.
    for (int i = 0; i < (A_width + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        // a.
        // mem coalescing is done on a warp-level. mem coalescing is a statement about what the hardware does on a warp level
        // accesses to A are clearly coalesced (cont. on tx)
        // accesses to B have the same ty val within a warp (assuming tile-width >= 32)
        // ergo accesses to B are coalesced (cont. on COL)
        int A_col = i * TILE_WIDTH + tx;
        int B_row = i * TILE_WIDTH + ty;
        if (A_col >= A_width || B_row >= A_height)
        {
            return;
        }
        A_tile[ty][tx] = A[ROW * A_width + A_col]; // row is const, col within a tile is const, col by tile is not const
        B_tile[ty][tx] = B[B_row * B_width + COL];
        // b.
        __syncthreads();
        // now matmul the tiles
        // c.
        for (int j = 0; j < TILE_WIDTH; j++)
        {
            outVal += A_tile[ty][j] * B_tile[j][tx];
        }
        __syncthreads();
    }
    // 4.
    C[ROW * B_width + COL] = outVal;
}

//  row matmul sucks!!! from the perspective of a single thread, access to A is
// contiguous over time. But that means that on a single iteration, each thread in
// the block is accessing a different row of A, which is a totally non-cont
// access pattern! That means that B is a contiguous access pattern tho,
// and maybe that's more important? Yes, because at least one load will have
// to be performed on be on each iteratin of the inner loop, but
// you only have to load from A on each iter of the outer loop.
// So if mem access is slow once every outer loop, but fast on
// every inner loop, that's preferable. Which means row matmul is
// preferable as written to colmatmul.
__global__ void rowMatmul(float *A, float *B, float *C, int i, int j, int k)
{
    int row = blockDim.x + blockIdx.x + threadIdx.x;

    if (row >= i)
    {
        return;
    }

    for (int x = 0; x < k; x++)
    {
        float val = 0.0f;
        for (int y = 0; y < j; y++)
        {
            // smart is to get this row of A into some registers
            // can't (it's an array)! put it into L1 cache via shared memory
            val += A[row * j + y] + B[y * k + x];
        }
        C[row * k + x] = val;
    }
}

__global__ void colMatmul(float *A, float *B, float *C, int i, int j, int k)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col >= k)
    {
        return;
    }

    for (int z = 0; z < i; z++)
    {
        float val = 0.0f;
        for (int y = 0; y < j; y++)
        {
            val += A[z * j + y] * B[y * k + col];
        }
        C[z * i + col] = val;
    }
}

__global__ void matmul(float *A, float *B, float *C, int i, int j, int k)
{
    // is each thread handling a single elt of the output matrix?

    // let's say yes, and that the output matrix is ixk
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= i || col >= k)
    {
        return;
    }
    float val = 0.0f;
    for (int idx = 0; idx < j; idx++)
    {
        // use registers!!! no other thread accesses C at this index. we can localize this.
        val += A[row * j + idx] * B[idx * k + col];
        // ^^ this part stays in memory, ^^ and ^^ get loaded in as needed
        // what if
        //                      ^^ this stays (bc row major?)
        // and ^^               and             ^^ get loaded in as needed?
    }
    C[row * k + col] = val;
}

void naiveMatmul(float *A, float *B, float *C, int i, int j, int k)
{

    // "row" and "col" are row and col of output.
    // input shapes are A -> ixj, B->jxk
    for (int row = 0; row < i; row++)
    {
        for (int col = 0; col < k; col++)
        {
            float tmp = 0.0f;
            for (int y = 0; y < j; y++)
            {
                // j is the shared dim,
                // the dim we are collapsing,
                // so it goes in the inner

                // of note: j to index in A is to jump from row to row
                // (there are j elts in a row of A)
                // k fills this role for B (there are k elts in a row of B)
                tmp += A[row * j + y] * B[y * k + col];
            }
            C[row * k + col] = tmp;
        }
    }
}

void initMat(float *mat, int h, int w)
{
    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            // mat[j * w + i] = (float)j;
            // mat[j * w + i] = (float)(j * w + i) / (i * j + 1);
            mat[j * w + i] = 3.0f;
        }
    }
}

int main()
{
    // today I want to test tiled, coarsened, non-sqaure matmul
    int i, j, k;
    float *a, *b, *c, *d_a, *d_b, *d_c;

    // do some stuff with the constants
    i = 200;
    j = 200;
    k = 200;

    a = (float *)malloc(i * j * sizeof(float));
    b = (float *)malloc(j * k * sizeof(float));
    c = (float *)malloc(i * k * sizeof(float));

    cudaMalloc(&d_a, i * j * sizeof(float));
    cudaMalloc(&d_b, j * k * sizeof(float));
    cudaMalloc(&d_c, i * k * sizeof(float));

    initMat(a, i, j);
    initMat(b, j, k);

    // a thing to remember about memset is that it fills as many bytes with the given value,
    // not as many "array elts" (it's not an array, just a pointer to some bytes!).
    // thus setting to 0 is easy because float(0) is four bytes each with the value 0
    memset((void *)c, 0, i * k * sizeof(float));

    cudaMemcpy(d_a, a, i * j * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, j * k * sizeof(float), cudaMemcpyHostToDevice);

    // TODO: investigate thrust::fill. I do want to learn how to memset in cuda, but for now let's just copy c over
    // cudaMemcpy(d_c, c, i * k * sizeof(float), cudaMemcpyHostToDevice);

    int blockLength = TILE_WIDTH;
    dim3 threadsPerBlock(blockLength, blockLength);
    dim3 gridShape((k + blockLength - 1) / blockLength, (i + blockLength - 1) / blockLength);

    std::cout << "grid x: " << gridShape.x << " grid y: " << gridShape.y << std::endl;
    std::cout << "block x: " << threadsPerBlock.x << " blok y: " << threadsPerBlock.y << std::endl;

    tiledMatMul<<<gridShape, threadsPerBlock>>>(d_a, d_b, d_c, i, j, k);
    // matmul<<<gridShape, threadsPerBlock>>>(d_a, d_b, d_c, i, j, k);
    // CoarseTiledMatMul<<<gridShape, threadsPerBlock>>>(d_a, d_b, d_c, i, j, k);

    cudaDeviceSynchronize();
    // verify C
    // to veryify correctness, I guess we can serially calculate A@B and compare
    naiveMatmul(a, b, c, i, j, k);

    float *res;
    res = (float *)malloc(i * k * sizeof(float));
    cudaMemcpy(res, d_c, i * k * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    int rescount = 0;
    int ccount = 0;

    for (int u = 0; u < i; u++)
    {
        for (int w = 0; w < k; w++)
        {
            maxError = max(maxError, abs(res[u * k + w] - c[u * k + w]));
            if (abs(res[u * k + w] - c[u * k + w]) > 0.0000001)
            {
                rescount++;
                // std::cout << u << ' ' << w << std::endl;
            }
            // std::cout << res[u * k + w] << ' ' << c[u * k + w] << std::endl;
            // if (abs(c[u * i + w] - 360.0f) > 0.00001)
            // {
            //     ccount++;
            // }
        }
        // std::cout << res[u] << std::endl;
        // std::cout << c[u] << std::endl;
    }
    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "Bad Res: " << rescount << std::endl;
    // std::cout << "Bad c: " << ccount << std::endl;
    // std::cout << res[0] << std::endl;
    // std::cout << c[0] << std::endl;

    free(a);
    free(b);
    free(c);
    free(res);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}