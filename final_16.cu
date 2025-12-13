#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>

#include <cuda_runtime.h>   
#include <sched.h>
#include <png.h>

using namespace std;

constexpr double EPSILON = 1e-15;
constexpr int BLOCK_SIZE = 128;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

void read_png(const char* filename, int& width, int& height, 
              vector<double>& r_vec, vector<double>& g_vec, vector<double>& b_vec) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) { fclose(fp); cerr << "Error: create read struct failed" << endl; exit(1); }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { png_destroy_read_struct(&png_ptr, NULL, NULL); fclose(fp); cerr << "Error: create info struct failed" << endl; exit(1); }

    if (setjmp(png_jmpbuf(png_ptr))) {
        cerr << "Error during init_io" << endl;
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        exit(1);
    }

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
    
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    if (bit_depth == 16) 
        png_set_strip_16(png_ptr);

    if (color_type == PNG_COLOR_TYPE_PALETTE) 
        png_set_palette_to_rgb(png_ptr);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) 
        png_set_expand_gray_1_2_4_to_8(png_ptr);

    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) 
        png_set_tRNS_to_alpha(png_ptr);

    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    png_set_strip_alpha(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png_ptr, info_ptr));
    }

    png_read_image(png_ptr, row_pointers);

    r_vec.resize(width * height);
    g_vec.resize(width * height);
    b_vec.resize(width * height);

    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {

            png_bytep px = &(row[x * 3]); 
            
            r_vec[y * width + x] = static_cast<double>(px[0]); // R
            g_vec[y * width + x] = static_cast<double>(px[1]); // G
            b_vec[y * width + x] = static_cast<double>(px[2]); // B
        }
    }

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);

    cout << "Loaded image: " << filename << " (" << width << "x" << height << ")" << endl;
}

void write_png(const char* filename, int width, int height, 
               const vector<double>& r_vec, const vector<double>& g_vec, const vector<double>& b_vec) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            auto clamp = [](double v) -> int { int p = (int)v; return (p<0)?0:((p>255)?255:p); };
            png_bytep color = row + x * 3;
            color[0] = (png_byte)clamp(r_vec[idx]);
            color[1] = (png_byte)clamp(g_vec[idx]);
            color[2] = (png_byte)clamp(b_vec[idx]);
        }
        png_write_row(png_ptr, row);
    }
    free(row); png_write_end(png_ptr, NULL); png_destroy_write_struct(&png_ptr, &info_ptr); fclose(fp);
    cout << "Saved: " << filename << endl;
}

// warp shuffle (reducting 32 threads within a warp)
__inline__ __device__ double warpReduceSum(double v) { 
    unsigned mask = 0xffffffffu; 
    for (int offset = 16; offset > 0; offset >>= 1) { 
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

// Block-wide reduction built from warp-level shuffles.
// Returns the block sum in thread 0
__inline__ __device__ double blockReduceSum(double v) { 
    __shared__ double warp_sums[32]; 

    int lane = threadIdx.x & 31; 
    int wid  = threadIdx.x >> 5; 

    // 1) Reduce within warp 
    v = warpReduceSum(v); 

    // 2) Lane 0 of each warp writes its partial sum 
    if (lane == 0) warp_sums[wid] = v; 
    __syncthreads(); 

    // 3) First warp reduces the warp sums
    v = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? warp_sums[lane] : 0.0; 
    if (wid == 0) v = warpReduceSum(v); 

    return v; 
}

__global__ void svd_step_kernel_streaming(int M, int N, double* U, double* V, int* pairs) {
    //   column major
    //   U(row=k, col=c) => U[c * M + k]
    //   V(row=r, col=c) => V[c * N + r]   
    int pair_idx = blockIdx.x;
    int col_i = pairs[2 * pair_idx];
    int col_j = pairs[2 * pair_idx + 1];
    int tid = threadIdx.x;

    double sum_aa = 0.0;
    double sum_bb = 0.0;
    double sum_ab = 0.0;

    // for coalesced accumulate partial dot-products along rows k for the two columns (col_i, col_j) 
    for (int k = tid; k < M; k += blockDim.x) {
        double val_i = U[col_i * M + k]; 
        double val_j = U[col_j * M + k]; 
        sum_aa += val_i * val_i;
        sum_bb += val_j * val_j;
        sum_ab += val_i * val_j;
    }

    // Using warp shuffle only needed fewer sync  
    double alpha = blockReduceSum(sum_aa); 
    double beta  = blockReduceSum(sum_bb); 
    double gamma = blockReduceSum(sum_ab); 
    __shared__ double c, s;
    __shared__ bool perform_rotation;

    if (tid == 0) {
        perform_rotation = false;
        if (abs(gamma) > EPSILON * sqrt(alpha * beta)) {
            perform_rotation = true;
            double zeta = (beta - alpha) / (2.0 * gamma);
            double sign_z = (zeta >= 0) ? 1.0 : -1.0;
            double t = sign_z / (abs(zeta) + sqrt(1.0 + zeta * zeta));
            c = 1.0 / sqrt(1.0 + t * t);
            s = c * t;
        }
    }
    __syncthreads();

    // Accumulate in column major indexing
    if (perform_rotation) {
        for (int k = tid; k < M; k += blockDim.x) {
            double val_i = U[col_i * M + k]; 
            double val_j = U[col_j * M + k]; 
            U[col_i * M + k] = c * val_i - s * val_j; 
            U[col_j * M + k] = s * val_i + c * val_j;
        }

        for (int k = tid; k < N; k += blockDim.x) {
            double val_i = V[col_i * N + k]; 
            double val_j = V[col_j * N + k]; 
            V[col_i * N + k] = c * val_i - s * val_j; 
            V[col_j * N + k] = s * val_i + c * val_j; 
        }
    }
}

void one_sided_jacobi_svd_cuda(int M, int N, vector<double>& h_U, vector<double>& h_S, vector<double>& h_V) {
    // 1. Initialize V as Identity
    h_V.assign((size_t)N * N, 0.0);
    for (int i = 0; i < N; ++i) {
        h_V[i * N + i] = 1.0; 
    }

    // 2. Allocate Device Memory
    double *d_U, *d_V;
    int *d_pairs;

    size_t size_U = (size_t)M * N * sizeof(double);
    size_t size_V = (size_t)N * N * sizeof(double);
    size_t size_pairs = (size_t)N * sizeof(int); 

    CUDA_CHECK(cudaMalloc(&d_U, size_U));
    CUDA_CHECK(cudaMalloc(&d_V, size_V));
    CUDA_CHECK(cudaMalloc(&d_pairs, size_pairs));

    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), size_U, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), size_V, cudaMemcpyHostToDevice));

    // 3. Round Robin Setup (pair columns)
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    int num_pairs = N / 2;
    int max_sweeps = 15;

    size_t shared_mem_size = 0; // warp shuffle -> no extern dynamic shared memory

    // 4. Main Loop
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        for (int step = 0; step < N - 1; ++step) {
            std::vector<int> current_pairs(N);

            for (int i = 0; i < num_pairs; ++i) {
                current_pairs[2 * i]     = idx[i];
                current_pairs[2 * i + 1] = idx[N - 1 - i];
            }

            CUDA_CHECK(cudaMemcpy(d_pairs, current_pairs.data(), (size_t)(2 * num_pairs) * sizeof(int), cudaMemcpyHostToDevice));

            svd_step_kernel_streaming<<<num_pairs, BLOCK_SIZE, shared_mem_size>>>(M, N, d_U, d_V, d_pairs);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            std::rotate(idx.begin() + 1, idx.begin() + 2, idx.end());
        }
    }

    // 5. Retrieve Results 
    CUDA_CHECK(cudaMemcpy(h_U.data(), d_U, size_U, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_V.data(), d_V, size_V, cudaMemcpyDeviceToHost));

    // 6. column-wise Compute S and Normalize U
    h_S.resize(N);
    for (int i = 0; i < N; ++i) {
        double norm = 0.0;
        const size_t base = (size_t)i * M; 
        for (int k = 0; k < M; ++k) {
            double v = h_U[base + k]; 
            norm += v * v;
        }
        h_S[i] = std::sqrt(norm);

        if (h_S[i] > EPSILON) {
            double inv_s = 1.0 / h_S[i];
            for (int k = 0; k < M; ++k) {
                h_U[base + k] *= inv_s; 
            }
        }
    }

    // 7. Sort Singular Values (descending) and reorder columns of U and V
    std::vector<std::pair<double, int>> order(N);
    for (int i = 0; i < N; ++i) order[i] = {h_S[i], i};
    std::sort(order.rbegin(), order.rend());

    std::vector<double> sorted_S(N);
    std::vector<double> sorted_U((size_t)M * N);
    std::vector<double> sorted_V((size_t)N * N);

    for (int i = 0; i < N; ++i) {
        int old_index = order[i].second;
        sorted_S[i] = order[i].first;

        // copy U column old_index -> new column i
        const size_t srcU = (size_t)old_index * M; 
        const size_t dstU = (size_t)i * M;
        memcpy(&sorted_U[dstU], &h_U[srcU], (size_t)M * sizeof(double)); // using memcpy seems faster a bit

        // copy V column old_index -> new column i
        const size_t srcV = (size_t)old_index * N; 
        const size_t dstV = (size_t)i * N; 
        memcpy(&sorted_V[dstV], &h_V[srcV], (size_t)N * sizeof(double));
    }

    h_S = std::move(sorted_S);  //little optimize
    h_U = std::move(sorted_U);
    h_V = std::move(sorted_V);

    cudaFree(d_U); cudaFree(d_V); cudaFree(d_pairs);
}

//change to column major
void check(const char* name, int M, int width, int N,
           const vector<double>& A_rowmajor,
           const vector<double>& U_colmajor, 
           const vector<double>& S,
           const vector<double>& V_colmajor) {

    //   column-major indexing:
    //   U(r,k) = U_colmajor[k*M + r] 
    //   V(c,k) = V_colmajor[k*N + c] 
    double frobenius_sq_error = 0.0;

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < width; ++c) {
            double reconstructed_val = 0.0;
            for (int k = 0; k < N; ++k) {
                reconstructed_val += U_colmajor[(size_t)k * M + r] * S[k] * V_colmajor[(size_t)k * N + c]; 
            }
            double diff = A_rowmajor[(size_t)r * width + c] - reconstructed_val;
            frobenius_sq_error += diff * diff;
        }
    }

    double distance = sqrt(frobenius_sq_error);
    cout << name << " Channel Error (Frobenius): " << scientific << distance << endl;
}

void process_channel_cuda(int width, int height, int k,
                          const vector<double>& input_vec,
                          vector<double>& output_vec,
                          const char* channel_name) {

    // 1. Padding
    int stride = (width % 2 == 0) ? width : width + 1;

    // 2. Build U in column-major: U(row=r, col=c) = U[c*height + r] 
    vector<double> U_padded((size_t)height * stride, 0.0);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            U_padded[(size_t)c * height + r] = input_vec[(size_t)r * width + c]; 
        }
    }

    vector<double> S, V_padded;

    cout << "Processing " << channel_name << " (Size: " << width << "->" << stride << ")..." << endl;

    one_sided_jacobi_svd_cuda(height, stride, U_padded, S, V_padded);
    check(channel_name, height, width, stride, input_vec, U_padded, S, V_padded);

    // 3. Rank-k reconstruction back to row-major output (height x width)
    fill(output_vec.begin(), output_vec.end(), 0.0);

    int kk = std::min(k, width);
    for (int i = 0; i < kk; ++i) {
        double sigma = S[i];
        const size_t Ucol = (size_t)i * height;   // U column i base (length = height) 
        const size_t Vcol = (size_t)i * stride;   // V column i base (length = stride) 
        for (int r = 0; r < height; ++r) {
            double u = U_padded[Ucol + r];
            for (int c = 0; c < width; ++c) {
                output_vec[(size_t)r * width + c] += sigma * u * V_padded[Vcol + c];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: ./svd_cuda input.png k output.png" << endl;
        return 1;
    }
    string input_filename = argv[1];
    int k = atoi(argv[2]);
    string output_filename = argv[3];

    int width, height;
    vector<double> r_in, g_in, b_in;
    
    read_png(input_filename.c_str(), width, height, r_in, g_in, b_in);

    vector<double> r_out((size_t)width * height);
    vector<double> g_out((size_t)width * height);
    vector<double> b_out((size_t)width * height);

    k = std::min(width, k);

    process_channel_cuda(width, height, k, r_in, r_out, "Red");
    process_channel_cuda(width, height, k, g_in, g_out, "Green");
    process_channel_cuda(width, height, k, b_in, b_out, "Blue");

    write_png(output_filename.c_str(), width, height, r_out, g_out, b_out);

    cout << "Done." << endl;
    return 0;
}