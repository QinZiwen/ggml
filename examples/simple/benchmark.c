#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

// 引入苹果硬件加速库（CBLAS 接口）
#include <Accelerate/Accelerate.h>

// 引入 GGML 头文件
#include "ggml.h"
#include "ggml-cpu.h"

// 辅助函数：获取当前时间（毫秒）
double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

// --------------------------------------------------
// 1. Naive 传统三层循环测试
// --------------------------------------------------
double run_naive_benchmark(const float *A, const float *B, float *C, int n) {
    printf("正在运行 Test 1: Naive 三层循环... (请稍候)\n");
    double start = get_time_ms();
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
    
    double end = get_time_ms();
    double elapsed = end - start;
    printf(">>> 1. Naive 耗时: %.2f ms\n\n", elapsed);
    return elapsed;
}

// --------------------------------------------------
// 2. Apple Accelerate (CBLAS) 测试
// --------------------------------------------------
double run_cblas_benchmark(const float *A, const float *B, float *C, int n, double time_naive) {
    printf("正在运行 Test 2: Apple Accelerate (CBLAS)...\n");
    double start = get_time_ms();
    
    // cblas_sgemm 内部针对 Apple Silicon 的 AMX 进行了极致优化
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0f, A, n, B, n, 0.0f, C, n);
                
    double end = get_time_ms();
    double elapsed = end - start;
    printf(">>> 2. CBLAS 耗时: %.2f ms  [加速比: %.1fx]\n\n", 
            elapsed, time_naive / elapsed);
    return elapsed;
}

// --------------------------------------------------
// 3. GGML 图计算测试
// --------------------------------------------------
double run_ggml_benchmark(const float *A, const float *B, float *C_out, int n, double time_naive) {
    printf("正在运行 Test 3: GGML mul_mat 图计算...\n");
    
    size_t size = (size_t)n * n;
    
    // 动态估算并留足 GGML context 空间
    size_t ctx_size = 0;
    ctx_size += size * ggml_type_size(GGML_TYPE_F32); // tensor A
    ctx_size += size * ggml_type_size(GGML_TYPE_F32); // tensor B_input
    ctx_size += size * ggml_type_size(GGML_TYPE_F32); // result
    ctx_size += 5 * ggml_tensor_overhead();           
    ctx_size += ggml_graph_overhead();
    ctx_size += 16 * 1024 * 1024;                     // 16MB 缓冲区足够安全

    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context *ctx = ggml_init(params);

    // 创建标准 2D Tensor
    struct ggml_tensor *tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n); 
    struct ggml_tensor *tensor_b_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n); 
    
    // 拷贝 A
    memcpy(tensor_a->data, A, ggml_nbytes(tensor_a));
    
    // 关键优化：仿照大模型加载权重的行为，在数据拷入时提前进行行/列主序重排（不算在计算耗时内）
    float *b_dst = (float *)tensor_b_input->data;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b_dst[i * n + j] = B[j * n + i]; 
        }
    }

    // 构建绝对纯粹的计算图
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    
    // ggml_mul_mat 内部会自动对右矩阵进行转置。
    // 我们传入原本就已经物理转置好的 tensor_b_input，内部再转置一次负负得正，完美对应标准 A * B 的结果，且内存绝对连续！
    struct ggml_tensor *result = ggml_mul_mat(ctx, tensor_a, tensor_b_input);
    
    ggml_build_forward_expand(gf, result);

    double start = get_time_ms();
    // 调度 4 个线程执行纯粹的矩阵乘法
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    double end = get_time_ms();
    
    double elapsed = end - start;
    printf(">>> 3. GGML 耗时: %.2f ms  [加速比: %.1fx]\n\n", 
            elapsed, time_naive / elapsed);

    // 将计算结果拷贝至输出缓冲区，供后续校验
    memcpy(C_out, result->data, size * sizeof(float));

    ggml_free(ctx);
    return elapsed;
}

// --------------------------------------------------
// 4. 精度校验与防御优化的结果处理
// --------------------------------------------------
void verify_results(const float *C_naive, const float *C_blas, const float *C_ggml, size_t size) {
    float max_diff_blas_vs_ggml = 0.0f;
    float max_diff_naive_vs_blas = 0.0f;

    for (size_t i = 0; i < size; i++) {
        float diff1 = fabsf(C_blas[i] - C_ggml[i]);
        if (diff1 > max_diff_blas_vs_ggml) max_diff_blas_vs_ggml = diff1;

        float diff2 = fabsf(C_naive[i] - C_blas[i]);
        if (diff2 > max_diff_naive_vs_blas) max_diff_naive_vs_blas = diff2;
    }

    // 防御性策略：虚假条件分支阻止编译器把整个 C_naive 的计算和内存优化掉
    if (C_naive[0] < -100.0f) {
        printf("Never prints: %f", C_naive[0]);
    }

    printf("精度校验 (Naive vs CBLAS) 最大绝对误差: %e\n", max_diff_naive_vs_blas);
    printf("精度校验 (GGML vs CBLAS)  最大绝对误差: %e\n", max_diff_blas_vs_ggml);
    printf("==================================================\n");
}

// --------------------------------------------------
// 主入口
// --------------------------------------------------
int main(void) {
    const int n = 1024; 
    size_t size = (size_t)n * n;

    float *A = (float *)malloc(size * sizeof(float));
    float *B = (float *)malloc(size * sizeof(float));
    float *C_naive = (float *)malloc(size * sizeof(float));
    float *C_blas = (float *)malloc(size * sizeof(float));
    float *C_ggml = (float *)malloc(size * sizeof(float));

    if (!A || !B || !C_naive || !C_blas || !C_ggml) {
        fprintf(stderr, "内存分配失败！\n");
        return -1;
    }

    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < size; i++) {
        A[i] = (float)((double)rand() / (double)RAND_MAX);
        B[i] = (float)((double)rand() / (double)RAND_MAX);
    }

    printf("==================================================\n");
    printf("         矩阵乘法基准测试 (Matrix Scale: %d x %d)\n", n, n);
    printf("==================================================\n\n");

    double time_naive = run_naive_benchmark(A, B, C_naive, n);
    run_cblas_benchmark(A, B, C_blas, n, time_naive);
    run_ggml_benchmark(A, B, C_ggml, n, time_naive);

    verify_results(C_naive, C_blas, C_ggml, size);

    free(A); free(B); free(C_naive); free(C_blas); free(C_ggml);
    return 0;
}