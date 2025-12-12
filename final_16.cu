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

__device__ double blockReduce(double* sdata, int tid, int block_dim) {
    __syncthreads();
    for (unsigned int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    return sdata[0];
}

__global__ void svd_step_kernel_streaming(int M, int stride, double* U, double* V, int* pairs) {
    int pair_idx = blockIdx.x;
    int col_i = pairs[2 * pair_idx];
    int col_j = pairs[2 * pair_idx + 1];
    int tid = threadIdx.x;

    extern __shared__ double s_red[];

    double sum_aa = 0.0;
    double sum_bb = 0.0;
    double sum_ab = 0.0;

    for (int k = tid; k < M; k += blockDim.x) {
        double val_i = U[k * stride + col_i];
        double val_j = U[k * stride + col_j];
        sum_aa += val_i * val_i;
        sum_bb += val_j * val_j;
        sum_ab += val_i * val_j;
    }

    // 1. Reduce Alpha
    s_red[tid] = sum_aa;
    double alpha = blockReduce(s_red, tid, blockDim.x);
    __syncthreads(); 

    // 2. Reduce Beta
    s_red[tid] = sum_bb;
    double beta = blockReduce(s_red, tid, blockDim.x);
    __syncthreads();

    // 3. Reduce Gamma
    s_red[tid] = sum_ab;
    double gamma = blockReduce(s_red, tid, blockDim.x);
    __syncthreads();

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

    if (perform_rotation) {
        for (int k = tid; k < M; k += blockDim.x) {
            double val_i = U[k * stride + col_i];
            double val_j = U[k * stride + col_j];
            U[k * stride + col_i] = c * val_i - s * val_j;
            U[k * stride + col_j] = s * val_i + c * val_j;
        }

        for (int k = tid; k < stride; k += blockDim.x) {
            double val_i = V[k * stride + col_i];
            double val_j = V[k * stride + col_j];
            V[k * stride + col_i] = c * val_i - s * val_j;
            V[k * stride + col_j] = s * val_i + c * val_j;
        }
    }
}

void one_sided_jacobi_svd_cuda(int M, int N, vector<double>& h_U, vector<double>& h_S, vector<double>& h_V) {
    // 1. Initialize Identity Matrix
    h_V.assign(N * N, 0.0);
    for (int i = 0; i < N; ++i) h_V[i * N + i] = 1.0;

    // 2. Allocate Device Memory
    double *d_U, *d_V;
    int *d_pairs;

    size_t size_U = (size_t)M * N * sizeof(double);
    size_t size_V = (size_t)N * N * sizeof(double);
    size_t size_pairs = N * sizeof(int); 

    CUDA_CHECK(cudaMalloc(&d_U, size_U));
    CUDA_CHECK(cudaMalloc(&d_V, size_V));
    CUDA_CHECK(cudaMalloc(&d_pairs, size_pairs));

    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), size_U, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), size_V, cudaMemcpyHostToDevice));

    // 3. Round Robin Setup
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    int num_pairs = N / 2;
    int max_sweeps = 15;
    
    size_t shared_mem_size = BLOCK_SIZE * sizeof(double);

    // 4. Main Loop
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        for (int step = 0; step < N - 1; ++step) {
            std::vector<int> current_pairs(N);

            for (int i = 0; i < num_pairs; ++i) {
                current_pairs[2 * i] = idx[i];
                current_pairs[2 * i + 1] = idx[N - 1 - i];
            }

            CUDA_CHECK(cudaMemcpy(d_pairs, current_pairs.data(), 2 * num_pairs * sizeof(int), cudaMemcpyHostToDevice));
            
            svd_step_kernel_streaming<<<num_pairs, BLOCK_SIZE, shared_mem_size>>>(M, N, d_U, d_V, d_pairs);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            std::rotate(idx.begin() + 1, idx.begin() + 2, idx.end());
        }
    }

    // 5. Retrieve Results
    CUDA_CHECK(cudaMemcpy(h_U.data(), d_U, size_U, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_V.data(), d_V, size_V, cudaMemcpyDeviceToHost));

    // 6. Compute S and Normalize U
    h_S.resize(N);
    for (int i = 0; i < N; ++i) {
        double norm = 0.0;
        for (int k = 0; k < M; ++k) norm += h_U[k * N + i] * h_U[k * N + i];
        h_S[i] = std::sqrt(norm);
        
        if (h_S[i] > EPSILON) {
            double inv_s = 1.0 / h_S[i];
            for (int k = 0; k < M; ++k) h_U[k * N + i] *= inv_s;
        }
    }

    // 7. Sort Singular Values
    std::vector<std::pair<double, int>> order(N);
    for (int i = 0; i < N; ++i) order[i] = {h_S[i], i};
    std::sort(order.rbegin(), order.rend());

    std::vector<double> sorted_S(N);
    std::vector<double> sorted_U((size_t)M * N);
    std::vector<double> sorted_V((size_t)N * N); 

    for (int i = 0; i < N; ++i) {
        int old_index = order[i].second;
        sorted_S[i] = order[i].first;

        for (int r = 0; r < M; ++r) sorted_U[r * N + i] = h_U[r * N + old_index];
        for (int r = 0; r < N; ++r) sorted_V[r * N + i] = h_V[r * N + old_index];
    }

    h_S = sorted_S;
    h_U = sorted_U;
    h_V = sorted_V;

    cudaFree(d_U); cudaFree(d_V); cudaFree(d_pairs);
}

void check(const char* name, int M, int N, const vector<double>& A, 
           const vector<double>& U, const vector<double>& S, const vector<double>& V) {
    
    double frobenius_sq_error = 0.0;
    
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            double reconstructed_val = 0.0;
            for (int k = 0; k < N; ++k) {
                reconstructed_val += U[r * N + k] * S[k] * V[c * N + k];
            }
            double diff = A[r * N + c] - reconstructed_val;
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
    
    vector<double> U_padded((size_t)height * stride, 0.0);
    
    // Copy original data with Padding
    for(int r = 0; r < height; ++r) {
        memcpy(&U_padded[r * stride], &input_vec[r * width], width * sizeof(double));
    }

    vector<double> S, V_padded;
    
    cout << "Processing " << channel_name << " (Size: " << width << "->" << stride << ")..." << endl;
    
    one_sided_jacobi_svd_cuda(height, stride, U_padded, S, V_padded);
    check(channel_name, height, width, input_vec, U_padded, S, V_padded);

    // Reconstruct
    fill(output_vec.begin(), output_vec.end(), 0.0);
    
    for (int i = 0; i < k; ++i) {
        double sigma = S[i];
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                output_vec[r * width + c] += sigma * U_padded[r * stride + i] * V_padded[c * stride + i];
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
