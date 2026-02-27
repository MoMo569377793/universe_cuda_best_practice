#include<stdio.h>
#include<cuda.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce(float *d_input, float *d_output)
{
    // // 最初思路： if判断，但需要根据blockDim.x增加if个数
    // float *input_begin = d_input + blockDim.x * blockIdx.x;
    // if(threadIdx.x == 0 or 2 or 4 or 6)
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 1];
    // if(threadIdx.x == 0 or 4)
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 2];
    // if(threadIdx.x == 0)
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 4];

    // // 第一种写法：全局索引直接计算（不推荐，逻辑复杂容易乱）
    // int tid = threadIdx.x;
    // int index = blockDim.x * blockIdx.x + tid;

    // for(int i = 1; i < blockDim.x; i *= 2)
    // {
    //     if(tid % (i * 2) == 0)
    //         d_input[index] += d_input[index + i];
        
    //     __syncthreads();
    // }

    // if(tid == 0)
    //     d_output[blockIdx.x] = d_input[index];

    
    // 第二种写法：基指针 + 局部索引（保证每个block都使用从0开始的索引）
    float *input_begin = d_input + blockDim.x * blockIdx.x;

    float *input_begin = d_input + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    for(int i = 1; i < blockDim.x; i *= 2)
    {
        if(tid % (2 * i) == 0)
            input_begin[tid] += input_begin[tid + i];
        
        __syncthreads();
    }

    if(tid == 0)
        d_output[blockIdx.x] = input_begin[tid];

}

bool check(float *out, float *res, int n)
{
    for(int i = 0; i < n; i++)
    {
        if(abs(out[i] - res[i]) > 1e-3)
            return false;
    }
    return true;
}

int main()
{
    // printf("hello reduce");
    const int N = 32 * 1024 * 1024;
    float *input = (float *)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));

    int block_num = N / THREAD_PER_BLOCK;
    float *output = (float *)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));

    float *result = (float *)malloc(block_num * sizeof(float));
    for(int i = 0; i < N; i++)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }

    // cpu calculate
    for(int i = 0; i < block_num; i++)
    {
        float cur = 0;
        for(int j = 0; j < THREAD_PER_BLOCK; j++)
        {
            cur += input[i * THREAD_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    reduce<<<Grid, Block>>>(d_input, d_output);

    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    if(check(output, result, block_num))
        printf("the ans is right\n");
    else
    {
        printf("the ans is wrong\n");
        for(int i = 0; i < block_num; i++)
        {
            printf("%lf", output[i]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}