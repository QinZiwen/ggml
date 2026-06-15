#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip> // 用于控制输出精度
#include <Accelerate/Accelerate.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

using namespace std;
using namespace chrono;

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

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

// 测试 3: GGML 计算图调度 (多线程)
int test_ggml_graph_compute(const vector<float>& A, const vector<float>& B, int M, int N, int K, int n_threads) {
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
    auto start = high_resolution_clock::now();
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    auto end = high_resolution_clock::now();
    
    int time_us = duration_cast<microseconds>(end - start).count();
    
    // 释放资源
    ggml_free(ctx);
    
    return time_us;
}

// 测试 4: GGML backend (多后端调度)
int test_ggml_backend(const vector<float>& A, const vector<float>& B, int M, int N, int K) {
    // 初始化 GGML backend
    ggml_log_set(ggml_log_callback_default, nullptr);
    ggml_backend_load_all();
    
    ggml_backend_t backend = ggml_backend_init_best();
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    
    ggml_backend_t backends[2] = { backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    
    // 创建新的上下文用于 backend 测试
    size_t ctx_size_backend = 0;
    ctx_size_backend += M * K * ggml_type_size(GGML_TYPE_F32); 
    ctx_size_backend += K * N * ggml_type_size(GGML_TYPE_F32); 
    ctx_size_backend += M * N * ggml_type_size(GGML_TYPE_F32); 
    ctx_size_backend += 3 * ggml_tensor_overhead(); 
    ctx_size_backend += ggml_graph_overhead(); 
    
    struct ggml_init_params params_backend = {
        /*.mem_size =*/ ctx_size_backend,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc =*/ true, // 延迟分配，由 scheduler 管理
    };
    struct ggml_context * ctx_backend = ggml_init(params_backend);
    
    // 创建张量
    struct ggml_tensor * t_A_backend = ggml_new_tensor_2d(ctx_backend, GGML_TYPE_F32, K, M);
    struct ggml_tensor * t_B_backend = ggml_new_tensor_2d(ctx_backend, GGML_TYPE_F32, K, N);
    struct ggml_tensor * t_C_backend = ggml_mul_mat(ctx_backend, t_A_backend, t_B_backend);
    
    // 构建计算图
    struct ggml_cgraph * gf_backend = ggml_new_graph(ctx_backend);
    ggml_build_forward_expand(gf_backend, t_C_backend);
    
    // 重置并分配 graph
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf_backend);
    
    // 加载数据到 backend
    ggml_backend_tensor_set(t_A_backend, A.data(), 0, ggml_nbytes(t_A_backend));
    ggml_backend_tensor_set(t_B_backend, B.data(), 0, ggml_nbytes(t_B_backend));
    
    // 执行计算
    auto start = high_resolution_clock::now();
    ggml_backend_sched_graph_compute(sched, gf_backend);
    auto end = high_resolution_clock::now();
    
    int time_us = duration_cast<microseconds>(end - start).count();
    
    // 获取结果进行验证
    vector<float> C_backend(M * N, 0.0f);
    ggml_backend_tensor_get(t_C_backend, C_backend.data(), 0, ggml_nbytes(t_C_backend));
    cout << "   (校验数据防止优化: " << C_backend[0] << ")\n";
    
    // 释放 backend 资源
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_backend_free(cpu_backend);
    ggml_free(ctx_backend);
    
    return time_us;
}

int main() {
    int matrix_size = 8192;
    int M = matrix_size, N = matrix_size, K = matrix_size;

    cout << "硬件平台: Mac M2 | 矩阵维度: " << M << "x" << N << "\n";
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
    int time_ggml_backend = 0;

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
    int n_threads = 1;
    time_ggml = test_ggml_graph_compute(A, B, M, N, K, n_threads);
    
    cout << "3. GGML FP32 计算图 (" << n_threads << " 线程) 耗时: \t" << time_ggml << " us\n";

    // ==========================================
    // 测试 4: GGML backend (多后端调度)
    // ==========================================
    time_ggml_backend = test_ggml_backend(A, B, M, N, K);
    
    cout << "4. GGML Backend (多后端调度) 耗时: \t" << time_ggml_backend << " us\n";

    // ==========================================
    // 性能分析：计算并输出加速比
    // ==========================================
    double speedup_blas = (double)time_naive / time_blas;
    double speedup_ggml = (double)time_naive / time_ggml;
    double speedup_ggml_backend = (double)time_naive / time_ggml_backend;

    cout << "\n--------------------------------------------------\n";
    cout << "性能优化结论 (以 Naive 为基准 Base):\n";
    cout << fixed << setprecision(2); // 保留两位小数
    cout << "* Apple Accelerate (AMX) 加速比: \t" << speedup_blas << " 倍\n";
    cout << "* GGML FP32 (" << n_threads << "线程) 加速比: \t" << speedup_ggml << " 倍\n";
    cout << "* GGML Backend (多后端调度) 加速比: \t" << speedup_ggml_backend << " 倍\n";
    cout << "--------------------------------------------------\n";
    
    // 性能对比分析
    cout << "\n性能对比分析:\n";
    if (speedup_blas > speedup_ggml && speedup_blas > speedup_ggml_backend) {
        cout << "* Apple Accelerate (AMX) 性能最优，充分利用了 M2 芯片的专用硬件加速器\n";
    } else if (speedup_ggml_backend > speedup_blas && speedup_ggml_backend > speedup_ggml) {
        cout << "* GGML Backend 多后端调度性能最优，可能利用了更优的内存管理和任务分配\n";
    } else {
        cout << "* GGML 多线程计算在当前配置下表现最佳\n";
    }
    
    cout << "* 建议：在生产环境中优先使用 Apple Accelerate 或 GGML Backend 以获得最佳性能\n";
    cout << "--------------------------------------------------\n";

    return 0;
}