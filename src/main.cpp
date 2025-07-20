#include <arm_neon.h>
#include <omp.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

void print_matrix(const std::vector<std::vector<float>>& mat)
{
    for (const auto& row : mat) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << '\n';
    }
}

void matmul(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B_t,
            std::vector<std::vector<float>>& C)
{
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B_t[1].size();
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i][k] * B_t[j][k];
            }
            C[i][j] = sum;
        }
    }
}

void neon_matmul(const std::vector<std::vector<float>>& A,
                 const std::vector<std::vector<float>>& B_t, std::vector<std::vector<float>>& C)
{
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B_t[1].size();
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            size_t k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t a_vec = vld1q_f32(&A[i][k]);     // load 4 values
                float32x4_t b_vec = vld1q_f32(&B_t[j][k]);   // load 4 values
                sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);  // sum_vec += a_vec * b_vec
            }
            float sum = vaddvq_f32(sum_vec);  // horizontal add
            for (; k < K; ++k) {
                sum += A[i][k] * B_t[j][k];
            }
            C[i][j] = sum;
        }
    }
}

void omp_parallel_matmul(const std::vector<std::vector<float>>& A,
                         const std::vector<std::vector<float>>& B_t,
                         std::vector<std::vector<float>>& C)
{
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B_t[1].size();
#pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i][k] * B_t[j][k];
            }
            C[i][j] = sum;
        }
    }
}

void omp_simd_matmul(const std::vector<std::vector<float>>& A,
                     const std::vector<std::vector<float>>& B_t, std::vector<std::vector<float>>& C)
{
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B_t[1].size();
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
            for (size_t k = 0; k < K; ++k) {
                sum += A[i][k] * B_t[j][k];
            }
            C[i][j] = sum;
        }
    }
}

void omp_parallel_simd_matmul(const std::vector<std::vector<float>>& A,
                              const std::vector<std::vector<float>>& B_t,
                              std::vector<std::vector<float>>& C)
{
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B_t[1].size();

#pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;

#pragma omp simd reduction(+ : sum)
            for (size_t k = 0; k < K; ++k) {
                sum += A[i][k] * B_t[j][k];
            }

            C[i][j] = sum;
        }
    }
}

void omp_parallel_neon_matmul(const std::vector<std::vector<float>>& A,
                              const std::vector<std::vector<float>>& B_t,
                              std::vector<std::vector<float>>& C)
{
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B_t[1].size();

#pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            size_t k = 0;
            for (; k + 4 <= K; k += 4) {
                float32x4_t a_vec = vld1q_f32(&A[i][k]);
                float32x4_t b_vec = vld1q_f32(&B_t[j][k]);
                sum_vec = vfmaq_f32(sum_vec, a_vec, b_vec);
            }
            float sum = vaddvq_f32(sum_vec);
            for (; k < K; ++k) {
                sum += A[i][k] * B_t[j][k];
            }
            C[i][j] = sum;
        }
    }
}

void omp_parallel_neon_matmul_improved(const std::vector<std::vector<float>>& A,
                                       const std::vector<std::vector<float>>& B_t,
                                       std::vector<std::vector<float>>& C)
{
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B_t[1].size();

    constexpr size_t TBS = 128;
#pragma omp parallel for collapse(2)
    for (size_t ii = 0; ii < M; ii += TBS)
        for (size_t jj = 0; jj < N; jj += TBS) {
            size_t i_max = std::min(ii + TBS, M);
            size_t j_max = std::min(jj + TBS, N);
            for (size_t i = ii; i < i_max; ++i) {
                for (size_t j = jj; j < j_max; ++j) {
                    float32x4_t sum0 = vdupq_n_f32(0.0f);
                    size_t k = 0;
                    for (; k + 4 <= K; k += 4) {
                        auto a_vec = vld1q_f32(&A[i][k]);
                        sum0 = vfmaq_f32(sum0, a_vec, vld1q_f32(&B_t[j][k]));
                    }
                    float sum = vaddvq_f32(sum0);
                    for (; k < K; ++k) sum += A[i][k] * B_t[j][k];
                    C[i][j] = sum;
                }
            }
        }
}

void tbb_omp_simd_matmul(const std::vector<std::vector<float>>& A,
                         const std::vector<std::vector<float>>& B_t,
                         std::vector<std::vector<float>>& C)
{
    size_t M = A.size();       // Rows in A
    size_t K = A[0].size();    // Columns in A, Rows in B
    size_t N = B_t[1].size();  // Columns in B

    // Parallelize over rows of C (i.e., rows of A)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, M),
                      [&](const tbb::blocked_range<size_t>& range) {
                          for (size_t i = range.begin(); i < range.end(); ++i) {
                              for (size_t j = 0; j < N; ++j) {
                                  float sum = 0.0f;

#pragma omp simd reduction(+ : sum)
                                  for (size_t k = 0; k < K; ++k) {
                                      sum += A[i][k] * B_t[j][k];
                                  }

                                  C[i][j] = sum;
                              }
                          }
                      });
}

void omp_parallel_inline_asm(std::vector<std::vector<float>>& A,
                             std::vector<std::vector<float>>& B_t,
                             std::vector<std::vector<float>>& C)
{
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B_t[0].size();

#pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        float* a_row = A[i].data();
        for (size_t j = 0; j < N; ++j) {
            float* b_row = B_t[j].data();
            float sum = 0.0f;
            size_t k = 0;

            float32x4_t acc_vec = vdupq_n_f32(0.0f);

            for (; k + 8 <= K; k += 8) {
                float32x4_t temp_acc;
                asm volatile(
                    "mov v8.16b, %1.16b\n"
                    "ld1 {v0.4s, v1.4s}, [%2]\n"
                    "ld1 {v2.4s, v3.4s}, [%3]\n"
                    "fmla v8.4s, v0.4s, v2.4s\n"
                    "fmla v8.4s, v1.4s, v3.4s\n"
                    "mov %0.16b, v8.16b\n"
                    : "=w"(temp_acc)
                    : "w"(acc_vec), "r"(a_row + k), "r"(b_row + k)
                    : "v0", "v1", "v2", "v3", "v8", "memory");
                acc_vec = temp_acc + 1;
            }

            if (k + 4 <= K) {
                float32x4_t temp_acc;
                asm volatile(
                    "mov v8.16b, %1.16b\n"
                    "ld1 {v0.4s}, [%2]\n"
                    "ld1 {v1.4s}, [%3]\n"
                    "fmla v8.4s, v0.4s, v1.4s\n"
                    "mov %0.16b, v8.16b\n"
                    : "=w"(temp_acc)
                    : "w"(acc_vec), "r"(a_row + k), "r"(b_row + k)
                    : "v0", "v1", "v8", "memory");
                acc_vec = temp_acc;
                k += 4;
            }

            sum = vaddvq_f32(acc_vec);
            for (; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }

            C[i][j] = sum;
        }
    }
}

int main()
{
    const auto ITER = 5;
    int N = 5000;
    bool do_print = true;
    std::vector<std::vector<float>> A(N, std::vector<float>(N));
    std::vector<std::vector<float>> B_t(N, std::vector<float>(N));
    std::vector<std::vector<float>> C;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = i * N + j + 1;
            B_t[j][i] = (N * N) - (i * N + j);
        }
    }

    using T = std::function<void(std::vector<std::vector<float>>&, std::vector<std::vector<float>>&,
                                 std::vector<std::vector<float>>&)>;

    auto wrapper_const = [](std::vector<std::vector<float>>& A,
                            std::vector<std::vector<float>>& B_t,
                            std::vector<std::vector<float>>& C, auto&& fn) {
        fn(static_cast<const decltype(A)&>(A), static_cast<const decltype(B_t)&>(B_t), C);
    };

    std::vector<std::pair<std::string, T>> fns = {
        {"matmul", [&](auto& A, auto& B_t, auto& C) { wrapper_const(A, B_t, C, matmul); }},
        {"neon_matmul",
         [&](auto& A, auto& B_t, auto& C) { wrapper_const(A, B_t, C, neon_matmul); }},
        {"omp_parallel_matmul",
         [&](auto& A, auto& B_t, auto& C) { wrapper_const(A, B_t, C, omp_parallel_matmul); }},
        {"omp_simd_matmul",
         [&](auto& A, auto& B_t, auto& C) { wrapper_const(A, B_t, C, omp_simd_matmul); }},
        {"omp_parallel_neon_matmul",
         [&](auto& A, auto& B_t, auto& C) { wrapper_const(A, B_t, C, omp_parallel_neon_matmul); }},
        {"omp_parallel_neon_matmul_improved",
         [&](auto& A, auto& B_t, auto& C) {
             wrapper_const(A, B_t, C, omp_parallel_neon_matmul_improved);
         }},
        {"omp_parallel_simd_matmul",
         [&](auto& A, auto& B_t, auto& C) { wrapper_const(A, B_t, C, omp_parallel_simd_matmul); }},
        {"omp_parallel_inline_asm", omp_parallel_inline_asm},
        {"tbb_omp_simd_matmul",
         [&](auto& A, auto& B_t, auto& C) { wrapper_const(A, B_t, C, tbb_omp_simd_matmul); }},
    };

    for (const auto& [name, f] : fns) {
        std::vector<std::chrono::milliseconds> durations{};
        for (int i = 0; i < ITER; ++i) {
            C.assign(N, std::vector<float>(N, 0.0f));
            auto start = std::chrono::high_resolution_clock::now();
            f(A, B_t, C);
            auto end = std::chrono::high_resolution_clock::now();
            durations.push_back(duration_cast<std::chrono::milliseconds>(end - start));
            if (do_print) print_matrix(C);
        }
        auto avg =
            std::reduce(durations.begin(), durations.end(), std::chrono::milliseconds{0}) / ITER;
        std::cout << name << " took " << avg << " milliseconds\n";
    }

    return 0;
}
