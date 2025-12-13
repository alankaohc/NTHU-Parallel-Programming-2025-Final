#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP

#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

const double EPSILON = 1e-5;

// ---------------------------------------------------------
// [New] Read PNG and split into R, G, B vectors
// ---------------------------------------------------------
void read_png(const char* filename, int& width, int& height, 
              vector<double>& r_vec, vector<double>& g_vec, vector<double>& b_vec) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        cerr << "Error: png_create_read_struct failed" << endl;
        exit(1);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        cerr << "Error: png_create_info_struct failed" << endl;
        exit(1);
    }

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
    
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // 轉換各種格式為標準的 8-bit RGB
    if (bit_depth == 16) png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png_ptr);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER); // Add alpha channel filler to make it RGBA or similar alignment if needed, but easier to force RGB
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    // 這裡我們強制轉換並移除 Alpha，只保留 RGB
    png_set_strip_alpha(png_ptr);
    
    png_read_update_info(png_ptr, info_ptr);

    // 配置記憶體讀取 rows
    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png_ptr, info_ptr));
    }

    png_read_image(png_ptr, row_pointers);

    // 將資料轉存到 vector<double> R, G, B
    r_vec.resize(width * height);
    g_vec.resize(width * height);
    b_vec.resize(width * height);

    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 3]); // 3 bytes per pixel (RGB)
            r_vec[y * width + x] = static_cast<double>(px[0]);
            g_vec[y * width + x] = static_cast<double>(px[1]);
            b_vec[y * width + x] = static_cast<double>(px[2]);
        }
    }

    // 清理 libpng 記憶體
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);

    cout << "Loaded image: " << filename << " (" << width << "x" << height << ")" << endl;
}

// ---------------------------------------------------------
// [Modified] Write PNG taking 3 channels (R, G, B)
// ---------------------------------------------------------
void write_png(const char* filename, int width, int height, 
               const vector<double>& r_vec, const vector<double>& g_vec, const vector<double>& b_vec) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        cerr << "Error: Could not open file " << filename << " for writing." << endl;
        return;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return;
    }

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            
            // Helper lambda to clamp values
            auto clamp = [](double v) -> int {
                int p = static_cast<int>(v);
                return (p < 0) ? 0 : ((p > 255) ? 255 : p);
            };

            png_bytep color = row + x * 3;
            color[0] = (png_byte)clamp(r_vec[idx]);
            color[1] = (png_byte)clamp(g_vec[idx]);
            color[2] = (png_byte)clamp(b_vec[idx]);
        }
        png_write_row(png_ptr, row);
    }

    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    
    cout << "Saved image: " << filename << endl;
}

// ---------------------------------------------------------
// Original One-Sided Jacobi SVD Function
// ---------------------------------------------------------
void one_sided_jacobi_svd(int M, int N, const vector<double>& A, 
                          vector<double>& U, vector<double>& S, vector<double>& V) {
    
    U = A; 
    fill(V.begin(), V.end(), 0.0);
    for (int i = 0; i < N; ++i) V[i * N + i] = 1.0;

    int max_sweeps = 15;
    
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_error = 0.0;
        for (int i = 0; i < N - 1; ++i) {
            for (int j = i + 1; j < N; ++j) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int k = 0; k < M; ++k) {
                    double u_val = U[k * N + i]; 
                    double v_val = U[k * N + j]; 
                    alpha += u_val * u_val;
                    beta  += v_val * v_val;
                    gamma += u_val * v_val;
                }
                max_error = max(max_error, abs(gamma));

                if (abs(gamma) > EPSILON * sqrt(alpha * beta)) {
                    double zeta = (beta - alpha) / (2.0 * gamma);
                    double t = copysign(1.0 / (abs(zeta) + sqrt(1.0 + zeta * zeta)), zeta);
                    double c = 1.0 / sqrt(1.0 + t * t);
                    double s = c * t;

                    for (int k = 0; k < M; ++k) {
                        double t1 = U[k * N + i];
                        double t2 = U[k * N + j];
                        U[k * N + i] = c * t1 - s * t2;
                        U[k * N + j] = s * t1 + c * t2;
                    }
                    for (int k = 0; k < N; ++k) {
                        double t1 = V[k * N + i];
                        double t2 = V[k * N + j];
                        V[k * N + i] = c * t1 - s * t2;
                        V[k * N + j] = s * t1 + c * t2;
                    }
                }
            }
        }
        if (max_error < EPSILON) break;
    }

    S.resize(N);
    for (int i = 0; i < N; ++i) {
        double norm = 0.0;
        for (int k = 0; k < M; ++k) norm += U[k * N + i] * U[k * N + i];
        S[i] = sqrt(norm);
        
        if (S[i] > EPSILON) {
            double inv_s = 1.0 / S[i];
            for (int k = 0; k < M; ++k) U[k * N + i] *= inv_s;
        }
    }
}

// ---------------------------------------------------------
// Check Function
// ---------------------------------------------------------
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

// ---------------------------------------------------------
// Helper: Perform SVD and Reconstruction on one channel
// ---------------------------------------------------------
void process_channel(int w, int h, int k, 
                     const vector<double>& input_vec, 
                     vector<double>& output_vec, 
                     const char* channel_name) {
    
    vector<double> U(w * h), S(w), V(w * w);
    
    cout << "Processing " << channel_name << " channel..." << endl;
    one_sided_jacobi_svd(h, w, input_vec, U, S, V);

    // Check Accuracy (Optional)
    check(channel_name, h, w, input_vec, U, S, V);

    // Reconstruct
    fill(output_vec.begin(), output_vec.end(), 0.0);
    for (int i = 0; i < k; ++i) {
        double sigma = S[i];
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                output_vec[r * w + c] += sigma * U[r * w + i] * V[c * w + i];
            }
        }
    }
}


double diff_sec(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}


// ---------------------------------------------------------
// Main
// ---------------------------------------------------------
int main() {

    struct timespec t_start, t_read, t_calc, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);


    int width, height;
    vector<double> r_in, g_in, b_in;
    
    // 1. 讀取圖片
    const char* input_filename = "jerry512.png";
    read_png(input_filename, width, height, r_in, g_in, b_in);

    // 2. 準備輸出容器
    vector<double> r_out(width * height);
    vector<double> g_out(width * height);
    vector<double> b_out(width * height);

    // 3. 設定要保留的奇異值數量 (Top K)
    int k = 100;
    cout << "Performing SVD compression (k=" << k << ")..." << endl;

    // 4. 對每個 Channel 執行 SVD 與重建
    // 注意：One-Sided Jacobi 很慢，如果圖片很大會跑很久
    process_channel(width, height, k, r_in, r_out, "Red");
    process_channel(width, height, k, g_in, g_out, "Green");
    process_channel(width, height, k, b_in, b_out, "Blue");

    // 5. 將結果寫回 PNG
    write_png("output_reconstructed.png", width, height, r_out, g_out, b_out);

    cout << "All operations complete." << endl;

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    cout << "Total Time: " << diff_sec(t_start, t_end) << " seconds" << endl;

    return 0;
}