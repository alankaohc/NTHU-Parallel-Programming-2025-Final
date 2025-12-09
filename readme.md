
make seq     # 編譯   
srun ./seq   # 執行

大概流程:
會讀取 input png file 使用 one-sided joccobic method 做 SVD，
取前 k 個 eigenvector 重建矩陣，寫到 output_reconstruted.png 中。 
達到影像壓縮的效果。

Red Channel Error (Frobenius): 2.662173e-09 是指:
||A - UDV|| (A是input file 的矩陣，U,D,V是分解後的3個矩陣)。
應該都是接近0，代表分解成功。 