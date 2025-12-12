
make seq     # 編譯   
srun ./seq   # 執行

大概流程:
會讀取 input png file 使用 one-sided joccobic method 做 SVD，
取前 k 個 eigenvector 重建矩陣，寫到 output_reconstruted.png 中。 
達到影像壓縮的效果。

Red Channel Error (Frobenius): 2.662173e-09 是指:
||A - UDV|| (A是input file 的矩陣，U,D,V是分解後的3個矩陣)。
應該都是接近0，代表分解成功。 


##one-sided joccobic method 的流程解釋

這段程式碼實作了 單邊雅可比奇異值分解 (One-Sided Jacobi SVD) 演算法。它的目的是將一個 $M \times N$ 的矩陣 $A$ 分解為三個部分：$$A = U \Sigma V^T$$其中：$U$: 左奇異向量矩陣 (Columns are orthogonal and normalized)。$S$ (即 $\Sigma$): 奇異值 (Singular Values)，對角矩陣的對角元素。$V$: 右奇異向量矩陣 (Orthogonal matrix)。核心概念：為什麼叫「單邊」？傳統的 SVD (雙邊 Jacobi) 會同時從左邊和右邊旋轉矩陣來消去非對角元素。而 單邊 (One-Sided) 方法只對矩陣的 行 (Column) 進行操作。它的基本思想是透過一系列的旋轉（Givens Rotations），將矩陣 $A$ 的所有行 (Columns) 變得兩兩互相垂直 (Orthogonal)。當所有行都互相垂直後，這些行的長度就是奇異值，正規化後的行就是矩陣 $U$，而累積的旋轉矩陣就是 $V$。