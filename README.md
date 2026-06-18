# universe_cuda_best_practicie

> **状态：参考材料 / 优化阶梯素材（reference material）。**
> 这个仓库是 CUDA 优化的**参考材料**与学习素材，**不是**已完成的训练资产，
> 也不是求职作品集项目。
> 近期求职实操代码集中在 [`triton-kernel-pack`](../triton-kernel-pack)
> 的 **softmax → RMSNorm** 闭环；本仓只作为优化阶梯
> （访存 / 归约 / 向量化 / shared memory / warp shuffle / occupancy）的素材来源。
> 若未来要把它整理成正式课程或作品集，另立委托书，请勿当作已完成训练资产。

## Build（参考）

```
cd build
cmake ..
make
```
