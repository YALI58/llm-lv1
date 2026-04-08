#include <vector>
#include <cmath>

namespace llm_lv1 {

/**
 * @brief Matrix multiplication helper
 */
std::vector<float> matmul(
    const std::vector<float>& A,
    const std::vector<float>& B,
    int M, int K, int N
) {
    std::vector<float> C(M * N, 0.0f);
    
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
    
    return C;
}

/**
 * @brief Vector addition
 */
std::vector<float> vec_add(
    const std::vector<float>& A,
    const std::vector<float>& B
) {
    std::vector<float> C(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        C[i] = A[i] + B[i];
    }
    return C;
}

} // namespace llm_lv1
