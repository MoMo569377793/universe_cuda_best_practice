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

#define FETCH_FLOAT4(val) (reinterpret_cast<float4 *>(&(val))[0])

template <unsigned int M_NUM_PER_BLOCK,
          unsigned int N_NUM_PER_BLOCK,
          unsigned int K_NUM_PER_BLOCK,
          unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float *A_ptr_start = A_ptr + blockIdx.y * M_NUM_PER_BLOCK * K;
    float *B_ptr_start = B_ptr + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    float tmp[NUM_PER_THREAD] = {0.f};

    for(int s = 0; s < K; s += K_NUM_PER_BLOCK)
    {
        // 使用float4一次搬运四个元素
        FETCH_FLOAT4(a_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A_ptr_start[K * ty + s + tx * NUM_PER_THREAD]);
        // a_shared[ty][tx * NUM_PER_THREAD] = A_ptr_start[K * ty + s + tx * NUM_PER_THREAD];
        // a_shared[ty][tx * NUM_PER_THREAD + 1] = A_ptr_start[K * ty + s + tx * NUM_PER_THREAD + 1];
        // a_shared[ty][tx * NUM_PER_THREAD + 2] = A_ptr_start[K * ty + s + tx * NUM_PER_THREAD + 2];
        // a_shared[ty][tx * NUM_PER_THREAD + 3] = A_ptr_start[K * ty + s + tx * NUM_PER_THREAD + 3];

        FETCH_FLOAT4(b_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[N * (ty + s) + tx * NUM_PER_THREAD]);
        // b_shared[ty][tx * NUM_PER_THREAD] = B_ptr_start[N * (ty + s) + tx * NUM_PER_THREAD];
        // b_shared[ty][tx * NUM_PER_THREAD + 1] = B_ptr_start[N * (ty + s) + tx * NUM_PER_THREAD + 1];
        // b_shared[ty][tx * NUM_PER_THREAD + 2] = B_ptr_start[N * (ty + s) + tx * NUM_PER_THREAD + 2];
        // b_shared[ty][tx * NUM_PER_THREAD + 3] = B_ptr_start[N * (ty + s) + tx * NUM_PER_THREAD + 3];
        
        __syncthreads();

        for(int i = 0; i < NUM_PER_THREAD; i++)
        {
            for(int k = 0; k < K_NUM_PER_BLOCK; k++)
            {
                tmp[i] += a_shared[ty][k] * 
                           b_shared[k][tx * NUM_PER_THREAD + i];                
            }
        }

        __syncthreads();

    }

    float * C_ptr_start = C_ptr + N * blockIdx.y * M_NUM_PER_BLOCK
                                + blockIdx.x * N_NUM_PER_BLOCK;
    for(int i = 0; i < NUM_PER_THREAD; i++)
        C_ptr_start[ty * N + tx * NUM_PER_THREAD + i] = tmp[i];


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

    constexpr int M_NUM_PER_BLOCK = 32;
    constexpr int N_NUM_PER_BLOCK = 32;
    constexpr int K_NUM_PER_BLOCK = 32;
    constexpr int NUM_PER_THREAD = 4;

    dim3 block(8, 32);
    dim3 grid(n / N_NUM_PER_BLOCK, m / M_NUM_PER_BLOCK);

    cuda_sgemm<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

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