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

template <unsigned int BLOCK_SIZE_M,    // height of block of C that each block calculate
          unsigned int BLOCK_SIZE_K,    // width  of block of A that each block load into shared memory
          unsigned int BLOCK_SIZE_N,    // width  of block of C that each block calculate
          unsigned int THREAD_SIZE_Y,   // height of block of C that each thread calculate
          unsigned int THREAD_SIZE_X,   // width  of block of C that each thread calculate
          const bool ENABLE_DOUBLE_BUFFER>  // whether enable double buffering or not
__global__ void cuda_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // thread id in cur Block
    const int tid = ty * blockDim.x + tx;

    __shared__ float a_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float b_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.f};
    float reg_a[THREAD_SIZE_Y] = {0.f};
    float reg_b[THREAD_SIZE_X] = {0.f};
    float ldg_a_reg[4] = {0.f};
    
    float *A_ptr_start = A_ptr + blockIdx.y * BLOCK_SIZE_M * K;
    float *B_ptr_start = B_ptr + blockIdx.x * BLOCK_SIZE_N;

    // block内部索引重排
    const int A_tile_thread_per_row = BLOCK_SIZE_K / 4; // 2
    const int B_tile_thread_per_row = BLOCK_SIZE_N / 4; // 32
    
    const int A_tile_tid_x = tid % A_tile_thread_per_row;
    const int A_tile_tid_y = tid / A_tile_thread_per_row;
    const int B_tile_tid_x = tid % B_tile_thread_per_row;
    const int B_tile_tid_y = tid / B_tile_thread_per_row;

    // 流水线的装载和排空阶段不是满流水的，不能放在循环里
    FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(A_ptr_start[K * A_tile_tid_y + A_tile_tid_x * 4]);
    // 当前线程在block中的横坐标 * 每个线程横向处理几个数据(4)
    // 当前线程在block中的纵坐标 * 每个线程纵向处理几个数据(1)
    a_shared[0][A_tile_tid_x * 4][A_tile_tid_y] = ldg_a_reg[0];
    a_shared[0][A_tile_tid_x * 4 + 1][A_tile_tid_y] = ldg_a_reg[1];
    a_shared[0][A_tile_tid_x * 4 + 2][A_tile_tid_y] = ldg_a_reg[2];
    a_shared[0][A_tile_tid_x * 4 + 3][A_tile_tid_y] = ldg_a_reg[3];
    FETCH_FLOAT4(b_shared[0][B_tile_tid_y][B_tile_tid_x * 4]) = FETCH_FLOAT4(B_ptr_start[N * B_tile_tid_y + B_tile_tid_x * 4]);
    __syncthreads();

    int write_stage_idx = 1;
    for(int s = BLOCK_SIZE_K; s < K; s += BLOCK_SIZE_K)
    {
        FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(A_ptr_start[K * A_tile_tid_y + A_tile_tid_x * 4 + s]);
        a_shared[write_stage_idx][A_tile_tid_x * 4][A_tile_tid_y] = ldg_a_reg[0];
        a_shared[write_stage_idx][A_tile_tid_x * 4 + 1][A_tile_tid_y] = ldg_a_reg[1];
        a_shared[write_stage_idx][A_tile_tid_x * 4 + 2][A_tile_tid_y] = ldg_a_reg[2];
        a_shared[write_stage_idx][A_tile_tid_x * 4 + 3][A_tile_tid_y] = ldg_a_reg[3];
        FETCH_FLOAT4(b_shared[write_stage_idx][B_tile_tid_y][B_tile_tid_x * 4]) = FETCH_FLOAT4(B_ptr_start[N * (B_tile_tid_y + s) + B_tile_tid_x * 4]);
        

        write_stage_idx ^= 1;
        for(int k = 0; k < BLOCK_SIZE_K; k++)
        {
            // 从共享内存搬运到寄存器中服务下一步计算
            FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y]);
            FETCH_FLOAT4(reg_a[4]) = FETCH_FLOAT4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y + 4]);

            FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X]);
            FETCH_FLOAT4(reg_b[4]) = FETCH_FLOAT4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X + 4]);
        
            for(int i = 0; i < THREAD_SIZE_Y; i++)
                for(int j = 0; j < THREAD_SIZE_X; j++)
                    accum[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();

    }
    write_stage_idx = write_stage_idx ^ 1;
    for(int k = 0; k < BLOCK_SIZE_K; k++)
    {
        // 从共享内存搬运到寄存器中服务下一步计算
        FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y]);
        FETCH_FLOAT4(reg_a[4]) = FETCH_FLOAT4(a_shared[write_stage_idx][k][ty * THREAD_SIZE_Y + 4]);

        FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X]);
        FETCH_FLOAT4(reg_b[4]) = FETCH_FLOAT4(b_shared[write_stage_idx][k][tx * THREAD_SIZE_X + 4]);
    
        for(int i = 0; i < THREAD_SIZE_Y; i++)
            for(int j = 0; j < THREAD_SIZE_X; j++)
                accum[i][j] += reg_a[i] * reg_b[j];
    }

    float *C_ptr_start = C_ptr + N * by * BLOCK_SIZE_M + 
                         bx * BLOCK_SIZE_N;
    for(int i = 0; i < THREAD_SIZE_Y; i++)
    {
        FETCH_FLOAT4(C_ptr_start[N * (ty * THREAD_SIZE_Y + i) + tx * THREAD_SIZE_X]) = FETCH_FLOAT4(accum[i][0]);
        FETCH_FLOAT4(C_ptr_start[N * (ty * THREAD_SIZE_Y + i) + tx * THREAD_SIZE_X + 4]) = FETCH_FLOAT4(accum[i][4]);
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

    constexpr int BLOCK_SIZE_M = 128;
    constexpr int BLOCK_SIZE_K = 8;
    constexpr int BLOCK_SIZE_N = 128;

    constexpr int THREAD_SIZE_Y = 8;
    constexpr int THREAD_SIZE_X = 8;
    const bool ENABLE_DOUBLE_BUFFER = true;

    // 等同于dim3 block(16, 16);
    dim3 block(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 grid(n / BLOCK_SIZE_N, m / BLOCK_SIZE_M);

    cuda_sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);

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