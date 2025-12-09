# 設定 C++ 編譯器
CXX = g++

# C++ 編譯參數：C++11 標準、O3 最佳化
CXXFLAGS = -std=c++11 -O3

# 連結參數：數學函式庫 (-lm) 與 PNG 函式庫 (-lpng)
# 注意：一定要有 -lpng 才能使用 write_png 函式
LDFLAGS = -lm -lpng

# 執行檔名稱
TARGET = seq

.PHONY: all clean

all: $(TARGET)

clean:
	rm -f $(TARGET) *.png

# 編譯規則
# $@ 代表目標 (seq)
# $< 代表來源檔 (final_16.cc)
seq: final_16.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)