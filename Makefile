CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++20 \
           -O3 \
		   -DNDEBUG \
		   -march=native \
		   -mtune=native \
		   -msse4.1 \
		   -funroll-loops \
		   -fno-signed-zeros \
		   -fno-trapping-math \
		   -ftree-vectorize

NVCCFLAGS = -std=c++20 \
            -O3 \
			-DNDEBUG \
			-Xptxas=-v \
			-arch=sm_61

LDFLAGS = -lm -lpng

TARGET = seq svd_cuda

.PHONY: all clean

all: $(TARGET)

clean:
	rm -f $(TARGET)

seq: final_16.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

svd_cuda: final_16.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<
