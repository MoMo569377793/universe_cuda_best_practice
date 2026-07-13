#ifndef REDUCE_RUNTIME_CHECKS_CUH
#define REDUCE_RUNTIME_CHECKS_CUH

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace reduce_checks {

inline bool cuda_succeeded(cudaError_t status, const char *stage,
                           const char *expression, const char *file, int line)
{
    if(status == cudaSuccess)
        return true;

    std::fprintf(stderr, "%s: %s at %s:%d: %s\n", stage, expression, file,
                 line, cudaGetErrorString(status));
    return false;
}

inline bool results_match(const float *actual, const float *expected, int n,
                          float tolerance)
{
    bool matches = true;
    int reported = 0;
    for(int i = 0; i < n; ++i)
    {
        const double got = actual[i];
        const double want = expected[i];
        const double difference = std::fabs(got - want);
        if(!std::isfinite(got) || !std::isfinite(want) ||
           difference > tolerance)
        {
            matches = false;
            if(reported < 8)
            {
                std::fprintf(stderr,
                             "RESULT_MISMATCH index=%d actual=%.9g expected=%.9g diff=%.9g tolerance=%.9g\n",
                             i, got, want, difference,
                             static_cast<double>(tolerance));
                ++reported;
            }
        }
    }
    if(!matches && reported == 8)
        std::fprintf(stderr, "RESULT_MISMATCH additional differences omitted\n");
    return matches;
}

}  // namespace reduce_checks

#define REDUCE_CUDA_CHECK(call)                                             \
    do                                                                      \
    {                                                                       \
        if(!reduce_checks::cuda_succeeded((call), "CUDA_API_ERROR", #call, \
                                           __FILE__, __LINE__))             \
            return EXIT_FAILURE;                                            \
    } while(0)

#define REDUCE_CUDA_LAUNCH_CHECK()                                          \
    do                                                                      \
    {                                                                       \
        if(!reduce_checks::cuda_succeeded(cudaGetLastError(),               \
                                           "CUDA_LAUNCH_ERROR",             \
                                           "cudaGetLastError()",            \
                                           __FILE__, __LINE__))              \
            return EXIT_FAILURE;                                            \
    } while(0)

#define REDUCE_CUDA_SYNC_CHECK()                                            \
    do                                                                      \
    {                                                                       \
        if(!reduce_checks::cuda_succeeded(cudaDeviceSynchronize(),          \
                                           "CUDA_SYNC_ERROR",               \
                                           "cudaDeviceSynchronize()",       \
                                           __FILE__, __LINE__))              \
            return EXIT_FAILURE;                                            \
    } while(0)

#endif
