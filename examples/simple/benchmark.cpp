#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip> // 用于控制输出精度
#include <Accelerate/Accelerate.h>
#include "ggml.h"
#include "ggml-cpu.h"

using namespace std;
using namespace chrono;

// GEMM: GEneral Matrix Multiplication（通用矩阵乘法）
// 1. 普通的基础矩阵乘法 (单线程，无向量化优化)
void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    // 设置矩阵维度: 1024 x 1024
    int M = 1024, N = 1024, K = 1024;

    cout << "硬件平台: Mac M2 | 矩阵维度: " << M << "x" << N << "x" << K << "\n";
    cout << "--------------------------------------------------\n";

    // 初始化测试数据 (全 1.0f，仅作性能基准测试)
    vector<float> A(M * K, 1.0f);
    vector<float> B(K * N, 1.0f);
    vector<float> C_naive(M * N, 0.0f);
    vector<float> C_blas(M * N, 0.0f);

    // 用来记录耗时的变量 (单位：微秒)
    int time_naive = 0;
    int time_blas  = 0;
    int time_ggml  = 0;

    // ==========================================
    // 测试 1: 普通循环计算 (Naive)
    // ==========================================
    auto start = high_resolution_clock::now();
    naive_gemm(A.data(), B.data(), C_naive.data(), M, N, K);
    auto end = high_resolution_clock::now();
    time_naive = duration_cast<microseconds>(end - start).count();
    
    cout << "1. 普通循环矩阵乘法 (Naive) 耗时: \t" << time_naive << " us\n";
    cout << "   (校验数据防止优化: " << C_naive[0] << ")\n\n";

    // ==========================================
    // 测试 2: 苹果 Accelerate 框架 (Apple BLAS)
    // ==========================================
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    start = high_resolution_clock::now();
    // cblas_sgemm 专门调用 M2 内部的 AMX 矩阵协处理器
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C_blas.data(), N);
    end = high_resolution_clock::now();
    time_blas = duration_cast<microseconds>(end - start).count();
    cout << "2. Apple Accelerate 框架 (AMX) 耗时: \t" << time_blas << " us\n\n";

    // ==========================================
    // 测试 3: GGML 计算图调度 (多线程)
    // ==========================================
    // 初始化 GGML 内存上下文
    size_t ctx_size = 0;
    ctx_size += M * K * ggml_type_size(GGML_TYPE_F32); 
    ctx_size += K * N * ggml_type_size(GGML_TYPE_F32); 
    ctx_size += M * N * ggml_type_size(GGML_TYPE_F32); 
    ctx_size += 3 * ggml_tensor_overhead(); 
    ctx_size += ggml_graph_overhead(); 
    ctx_size += 1024; 
    
    struct ggml_init_params params = {
        /*.mem_size =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t_A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    struct ggml_tensor * t_B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);

    memcpy(t_A->data, A.data(), ggml_nbytes(t_A));
    memcpy(t_B->data, B.data(), ggml_nbytes(t_B));

    // 定义计算
    struct ggml_tensor * t_C = ggml_mul_mat(ctx, t_A, t_B);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_C);

    // 使用 n_threads 个线程进行计算
    int n_threads = 1;
    start = high_resolution_clock::now();
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    end = high_resolution_clock::now();
    time_ggml = duration_cast<microseconds>(end - start).count();
    
    cout << "3. GGML FP32 计算图 (" << n_threads << " 线程) 耗时: \t" << time_ggml << " us\n";

    // 释放资源
    ggml_free(ctx);

    // ==========================================
    // 性能分析：计算并输出加速比
    // ==========================================
    double speedup_blas = time_naive / time_blas;
    double speedup_ggml = time_naive / time_ggml;

    cout << "\n--------------------------------------------------\n";
    cout << "性能优化结论 (以 Naive 为基准 Base):\n";
    cout << fixed << setprecision(2); // 保留两位小数
    cout << "* Apple Accelerate (AMX) 加速比: \t" << speedup_blas << " 倍\n";
    cout << "* GGML FP32 (" << n_threads << "线程) 加速比: \t" << speedup_ggml << " 倍\n";
    cout << "--------------------------------------------------\n";

    return 0;
}