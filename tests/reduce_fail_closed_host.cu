#include "reduce_runtime_checks.cuh"

#include <cstdlib>
#include <cstring>
#include <limits>


int main(int argc, char **argv)
{
    if(argc != 2)
        return 2;

    const char *mode = argv[1];
    float expected[] = {1.0f, 2.0f, 3.0f};
    float actual[] = {1.0f, 2.0f, 3.0f};

    if(std::strcmp(mode, "correct") == 0)
        return reduce_checks::results_match(actual, expected, 3, 1e-3)
            ? EXIT_SUCCESS : EXIT_FAILURE;
    if(std::strcmp(mode, "mismatch") == 0)
    {
        actual[1] = 4.0f;
        return reduce_checks::results_match(actual, expected, 3, 1e-3)
            ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if(std::strcmp(mode, "nan") == 0)
    {
        actual[1] = std::numeric_limits<float>::quiet_NaN();
        return reduce_checks::results_match(actual, expected, 3, 1e-3)
            ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if(std::strcmp(mode, "cuda-api") == 0)
        return reduce_checks::cuda_succeeded(
            cudaErrorInvalidValue,
            "CUDA_API_ERROR",
            "injected CUDA API failure",
            __FILE__,
            __LINE__)
            ? EXIT_SUCCESS : EXIT_FAILURE;
    if(std::strcmp(mode, "launch") == 0)
        return reduce_checks::cuda_succeeded(
            cudaErrorInvalidConfiguration,
            "CUDA_LAUNCH_ERROR",
            "injected launch failure",
            __FILE__,
            __LINE__)
            ? EXIT_SUCCESS : EXIT_FAILURE;
    if(std::strcmp(mode, "sync") == 0)
        return reduce_checks::cuda_succeeded(
            cudaErrorLaunchFailure,
            "CUDA_SYNC_ERROR",
            "injected sync failure",
            __FILE__,
            __LINE__)
            ? EXIT_SUCCESS : EXIT_FAILURE;

    return 2;
}
