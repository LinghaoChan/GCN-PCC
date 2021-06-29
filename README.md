# GCN-based-PCC
**实验存在的问题：**

由于硬件资源的不足，项目的所有试验均依托Google Colab进行计算。

+ 由于显存不足，本文采用的参数设置均为最低的设置(k = 20, number of points = 1024)；
+ 由于担心显存不足的问题，实现的时候没有使用公式3中的GCN矩阵形式，而是采用了GCN结点级别邻居聚合的方式：

$$
h_{i}^{l+1} = g(\sum_{j\in \mathcal{N}_{i}}\frac{1}{|\mathcal{N}_{i}| \cdot|  \mathcal{N}_{j}|}h_{j})= g(\sum_{j\in \mathcal{N}_{i}}\frac{1}{\sqrt{k}\sqrt{k}}h_{j})= g(\frac{1}{k}\sum_{j\in \mathcal{N}_{i}}h_{j})
$$

+ 公式2中的全1向量$\mathbf{1}$的运算使用的是python的广播机制实现的。

```python
def knn(X, k):
    inner_pro = -2*torch.matmul(X.transpose(2, 1), X)
    X_square = torch.sum(X**2, dim=1, keepdim=True)
    pairwise_distance = -X_square - inner_pro - X_square.transpose(2, 1)
    index = pairwise_distance.topk(k=k, dim=-1)[1]
    return index
```

+ 我认为实验的结果还没有调整到最优(模型还存在一定程度的欠拟合，有提升空间)。由于使用Colab的时长有限制不能短时间内多次调整训练，我只进行了2-3次调参，因此一直没能调整到最优。