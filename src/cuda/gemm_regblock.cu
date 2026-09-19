#include "cuda/gemm_regblock.cuh"

__global__ void regblock_gemm_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {



     constexpr int TILE_SIZE = 16;
     constexpr int THREAD_TILE = 4;

    __shared__ float As[TILE_SIZE][TILE_SIZE+1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];


    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float acc[THREAD_TILE][THREAD_TILE] = {0.0f};

    for (int phase = 0; phase < (K + TILE_SIZE - 1 ) / TILE_SIZE;++phase){
        if (row < M && (phase * TILE_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + phase * TILE_SIZE + threadIdx.x]; 
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (col < N && (phase * TILE_SIZE + threadIdx.y) < K) {
            Bs[threadIdx.x][threadIdx.y] = B[col * K + phase * TILE_SIZE + threadIdx.y];
        }
        else {
            Bs[threadIdx.x][threadIdx.y] = 0.0f;
        }

}
void regblock_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(64, 4);
    dim3 grid((N + 63) / 64, (M + 3) / 4);
    regblock_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}