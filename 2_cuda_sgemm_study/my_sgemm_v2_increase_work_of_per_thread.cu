#include <stdio.h>

#define A(m, n) a[m * N + n]
#define B(m, n) b[m * N + n]

void random_matrix(int M, int N, float *a)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
#if 1
            A(m, n) = 2.0 * (float)drand48() - 1.0;
#else
            A(m, n) = (m - n) % 3;
#endif
}

void cpu_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < N; n++)
        {
            float tmp = 0.f;
            for(int k = 0; k < K; k++)
            {
                tmp += A_ptr[m * K + k] * B_ptr[k * N + n];
            }
            C_ptr[m * N + n] = tmp;
        }
    }
}

float compare_matrices(int M, int N, float *a, float *b)
{
    float max_diff = 0.0, diff;
    int printed = 0;

    for(int m = 0; m < M; m++)
    {
        for(int n = 0; n < N; n++)
        {
            diff = abs(A(m, n) - B(m, n));
            max_diff = (diff > max_diff ? diff : max_diff);
            if(printed == 0)
                if(max_diff > 0.5f || max_diff < -0.5f)
                {
                    printf("\n error: i %d j %d diff %f got %f expect %f\n", m, n, max_diff, A(m, n), B(m, n));
                    printed = 1;
                }
        }
    }

    return max_diff;
}

template <unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    constexpr int STEP = BLOCK_SIZE * STRIDE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // const int row = blockDim.y * blockIdx.y + threadIdx.y;
    // const int col = blockDim.x * blockIdx.x + threadIdx.x;
    float *A_ptr_start = A_ptr + STEP * blockIdx.y * K;
    float *B_ptr_start = B_ptr + STEP * blockIdx.x;

    __shared__ float a_shared[STEP][STEP];
    __shared__ float b_shared[STEP][STEP];

    float temp[STRIDE][STRIDE] = {0.f};
    for(int s = 0; s < K; s += STEP)
    {
        for(int i = 0; i < STRIDE; i++)
        {
            for(int j = 0; j < STRIDE; j++)
            {
                a_shared[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = A_ptr_start[(ty + i * BLOCK_SIZE) * K + tx + j * BLOCK_SIZE + s];
                b_shared[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = B_ptr_start[(ty + i * BLOCK_SIZE + s) * N + tx + j * BLOCK_SIZE];
            }
        }
        __syncthreads();
        
        for(int i = 0; i < STRIDE; i++)
        {
            for(int j = 0; j < STRIDE; j++)
            {
                for(int k = 0; k < STEP; k++)
                {
                    temp[i][j] += a_shared[ty + i * BLOCK_SIZE][k] * b_shared[k][tx + j * BLOCK_SIZE];
                }
            }
        }
        __syncthreads();

    }

    float *C_ptr_start = C_ptr + N * blockIdx.y * STEP + blockIdx.x * STEP;
    for(int i = 0; i < STRIDE; i++)
    {
        for(int j = 0; j < STRIDE; j++)
        {
            C_ptr_start[N * (ty + i * BLOCK_SIZE) + tx + j * BLOCK_SIZE] = temp[i][j];
        }
    }
}

int main()
{
    int m = 512;
    int n = 512;
    int k = 512;
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    float *matrix_A_host = (float *)malloc(mem_size_A);
    float *matrix_B_host = (float *)malloc(mem_size_B);

    float *matrix_C_gpu_calc = (float *)malloc(mem_size_C);
    float *matrix_C_cpu_calc = (float *)malloc(mem_size_C);

    random_matrix(m, k, matrix_A_host);
    random_matrix(k, n, matrix_B_host);
    memset(matrix_C_gpu_calc, 0, mem_size_C);
    memset(matrix_C_cpu_calc, 0, mem_size_C);

    float *matrix_A_device, *matrix_B_device, *matrix_C_device;
    cudaMalloc((void **)&matrix_A_device, mem_size_A);
    cudaMalloc((void **)&matrix_B_device, mem_size_B);
    cudaMalloc((void **)&matrix_C_device, mem_size_C);

    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);

    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_cpu_calc, m, n, k);

    constexpr int BLOCK = 16;
    constexpr int STRIDE = 2;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((n + BLOCK - 1) / BLOCK / STRIDE, (m + BLOCK - 1) / BLOCK / STRIDE);
    cuda_sgemm<BLOCK, STRIDE><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

    cudaMemcpy(matrix_C_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);

    float diff = compare_matrices(m, n, matrix_C_gpu_calc, matrix_C_cpu_calc);
    if(diff > 0.5f || diff < -0.5f)
    {
        printf("diff too big !\n");
        exit(-1);
    }
    else
        printf("right\n");

    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_cpu_calc);
    free(matrix_C_gpu_calc);

    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
    
    return 0;
}