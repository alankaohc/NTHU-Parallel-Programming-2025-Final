#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include <stdio.h>

// =========================================================
// [GLOBAL CONFIG] 全域設定 (大家共用，不要亂改)
// =========================================================
#define BLOCK_SIZE 256
#define EPSILON 1e-15

struct Pair {
    int i;
    int j;
};

// =========================================================
// [MEMBER C] MATH CORE 區塊
// 負責: 數學運算、Warp Shuffle、Givens Rotation
// =========================================================

// 輔助: Warp 內加總 (不需要改)
__device__ double warp_reduce_sum(double val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 核心運算函式
__device__ void compute_and_rotate(double* s_row_i, double* s_row_j, int N) {
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // TODO [Member C]: 1. 計算局部內積 (alpha, beta, gamma)
    // 每個人算自己負責的部分
    double local_alpha = 0.0;
    double local_beta = 0.0;
    double local_gamma = 0.0;

    // 提示: 用 stride 迴圈累加 s_row_i[k] * s_row_i[k] 等等...
    
    // TODO [Member C]: 2. 執行 Reduction (Warp -> Block)
    // 先用 warp_reduce_sum，然後用 Shared Memory 把各個 Warp 的結果加起來
    // 最終只有 Thread 0 拿到全域的 alpha, beta, gamma
    
    __syncthreads();

    // TODO [Member C]: 3. 計算 c, s (只有 Thread 0 做)
    __shared__ double c, s;
    if (tid == 0) {
        // 這裡填入 sequential code 的數學公式
        // c = ...; s = ...;
    }
    __syncthreads(); // 重要: 等 c, s 算好

    // TODO [Member C]: 4. 更新 Shared Memory (所有 Threads 平行做)
    // 讀取舊值 -> 旋轉 -> 寫回
    // s_row_i[k] = c * val_i - s * val_j;
}


// =========================================================
// [MEMBER B] MEMORY MOVER 區塊
// 負責: Global <-> Shared Memory 搬運 (Float4/Double2 優化)
// =========================================================

__device__ void load_rows_to_shared(double* d_data, int r1, int r2, 
                                    double* s_r1, double* s_r2, int N) {
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // TODO [Member B]: 從 Global Memory 搬運到 Shared Memory
    // 挑戰題: 嘗試使用 reinterpret_cast<double2*> 進行向量化讀取 (一次讀2個double)
    // 如果 N 保證是偶數，可以這樣加速頻寬
    
    // 基本實作 (Stride Loop):
    for (int k = tid; k < N; k += stride) {
        // s_r1[k] = d_data[r1 * N + k]; (注意 Row-Major 的索引算式)
    }
}

__device__ void store_rows_to_global(double* d_data, int r1, int r2, 
                                     double* s_r1, double* s_r2, int N) {
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // TODO [Member B]: 把 Shared Memory 寫回 Global Memory
    // 邏輯跟 Load 一樣，只是方向相反
}


// =========================================================
// [KERNEL] 整合區塊 (由 Member A 寫好框架，通常不需大改)
// =========================================================

__global__ void parallel_jacobi_kernel(double* A, double* V, Pair* pairs, int N) {
    int pair_idx = blockIdx.x;
    int r1 = pairs[pair_idx].i;
    int r2 = pairs[pair_idx].j;

    // 動態配置 Shared Memory: 大小由 Host 端決定
    // 前半段放 Row i，後半段放 Row j
    extern __shared__ double smem[];
    double* s_r1 = smem;
    double* s_r2 = smem + N;

    // 1. 載入 A (Member B 工作)
    load_rows_to_shared(A, r1, r2, s_r1, s_r2, N);
    __syncthreads();

    // 2. 運算 A (Member C 工作)
    compute_and_rotate(s_r1, s_r2, N);
    __syncthreads();

    // 3. 寫回 A (Member B 工作)
    store_rows_to_global(A, r1, r2, s_r1, s_r2, N);
    __syncthreads();

    // 4. 處理 V 矩陣 (重複利用 B 和 C 的功能)
    // 載入 V -> 套用剛剛算出的旋轉 (c, s 還在 smem 裡可以用嗎? 
    // 注意: compute_and_rotate 會重算 c,s，所以 V 的旋轉通常需要一個
    // 簡化版的 apply_rotation_only 函式，或者由 Member C 寫在同一個函式裡判斷)
    
    // 簡單解法: 這裡再呼叫一次 load/store，但運算部分只做 "Apply Rotation"
    // 為了作業方便，V 的部分可以先留白，確認 A 收斂再做
}


// =========================================================
// [MEMBER A] HOST ARCHITECT 區塊
// 負責: Main, Round-Robin 排程, Padding, I/O
// =========================================================

std::vector<Pair> generate_step_pairs(int N, const std::vector<int>& ids) {
    std::vector<Pair> pairs;
    // TODO [Member A]: 實作 Round Robin 配對邏輯
    // ids[0]... 對應 ids[N-1]...
    // 記得檢查 -1 (Dummy ID)
    return pairs;
}

void rotate_ids(std::vector<int>& ids) {
    // TODO [Member A]: 實作 vector 旋轉 (std::rotate)
}

int main(int argc, char** argv) {
    // 1. 假裝讀取資料 (或是寫好 fread)
    int N_orig = 1080; // 假設 Input 大小
    
    // TODO [Member A]: Padding
    // 為了 float4/double2 優化，建議 Pad 到 2 的倍數，甚至 32 的倍數
    int N = N_orig;
    if (N % 2 != 0) N += 1;

    // 2. 記憶體配置 (Host & Device)
    size_t size = N * N * sizeof(double);
    double *h_A, *d_A, *d_V;
    // cudaMalloc...
    
    // 3. 初始化 Round Robin IDs
    std::vector<int> ids(N);
    std::iota(ids.begin(), ids.end(), 0);
    // if (odd) ids.back() = -1;

    // 4. 主迴圈
    int max_sweeps = 15;
    // 一個 Sweep 需要 2*N (或 N-1) 個 Steps，視 RR 實作而定
    int steps = N - 1; 

    Pair* d_pairs;
    cudaMalloc(&d_pairs, (N/2) * sizeof(Pair));

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        for (int step = 0; step < steps; ++step) {
            
            // [Member A]: 產生配對
            std::vector<Pair> h_pairs = generate_step_pairs(N, ids);
            
            // [Member A]: 傳給 GPU
            cudaMemcpy(d_pairs, h_pairs.data(), h_pairs.size() * sizeof(Pair), cudaMemcpyHostToDevice);

            // [Member A]: 啟動 Kernel
            // Grid Size = 配對數, Block Size = 256
            // Shared Mem Size = 2 * N * sizeof(double)
            parallel_jacobi_kernel<<<h_pairs.size(), BLOCK_SIZE, 2*N*sizeof(double)>>>(d_A, d_V, d_pairs, N);

            // [Member A]: 旋轉 ID
            rotate_ids(ids);
        }
    }

    // 5. 收尾與輸出
    // cudaMemcpy DeviceToHost...
    // 驗證結果...

    return 0;
}