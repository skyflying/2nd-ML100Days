

import numpy as np

def conv_single_step(a_slice_prev, W, b):
    """
    定義一層 Kernel (內核), 使用的參數說明如下
    Arguments:
        a_slice_prev -- 輸入資料的維度
        W -- 權重, 被使用在 a_slice_prev
        b -- 偏差參數 
    Returns:
        Z -- 滑動窗口（W，b）卷積在前一個 feature map 上的結果
    """

    # 定義一個元素介於 a_slice and W
    s = a_slice_prev * W
    # 加總所有的 "s" 並指定給Z.
    Z = np.sum(s)
    # Add bias b to Z. 這是 float() 函數,
    Z = float(Z + b)

    return Z
	
	

np.random.seed(1)
#定義一個 4x4x3 的 feature map
a_slice_prev = np.random.randn(5, 5, 3)
W = np.random.randn(5, 5, 3)
b = np.random.randn(1, 1, 1)

#取得計算後,卷績矩陣的值
Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)