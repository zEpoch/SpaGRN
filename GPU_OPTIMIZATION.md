# Geary GPU 加速与镜像安装

## 重新构建镜像

此版本已直接修改 `src/spagrn/global_bivariate_gearys_C_fast.py`，安装后无需再打补丁。
将修改后的仓库上传 GitHub 或放入镜像构建目录。在该仓库根目录执行：

```bash
bash scripts/install_gpu.sh
```

该脚本按 Python 3.8、PyTorch 2.4.1 CUDA 11.8 及 `constraints-gpu.txt` 创建环境、安装当前仓库、
执行数值回归测试并注册 `spagrn-gpu` 内核。它不克隆其他版本的仓库。
如继续用旧的逐行安装命令，也可以；确保安装的是此修改版本，并使用仓库内的约束文件。
安装脚本针对新镜像；已存在同名 conda 环境时不会自动删除或覆盖它。

```python
# 原 infer 参数可以继续使用
# torch_device='cuda:0', pair_batch_size=256
```

`infer`、`spg` 和加速函数的默认 `pair_batch_size` 统一为 256。
新版该参数表示每块靶基因数量；每个 TF 块最多 64 个。
显存不足可降为 128 或 64；不自动回退 OOM。不要仅根据旧实现的显存占用调大批次。

## 修改内容

旧实现对每个 TF 重复上传同样的靶基因并生成边数×批量大小的邻居差分数组。
新版设 B 为边端点差分矩阵，以 L=B.T@diag(w)@B 表示空间变换，计算
`TF.T @ L @ targets`；每块靶基因只变换一次，与所有 TF 块复用。

对于可能有向、重复、有自环的权重表，使用：

```
L = diag(W行和 + W列和) - W - W.T
```

不能简单用 `diag(W行和)-W` 替代，否则改变当前评分。
保留 float64、原方差分母下限、输出列和 TF 外层的结果顺序。
基因先中心化，避免常量基因消去误差，不改变边差分的数学定义。
输入稀疏表达矩阵仅在当前基因块转稠密；不构造细胞×细胞稠密矩阵。
`num_workers` 与旧 GPU 实现一样不控制 GPU 内层。

启动日志打印实际 TF 数、靶基因数、配对总数、细胞数、批量大小及设备。
进度条以靶基因块为单位，显示总数；不再以 TF 为单位。

## 数值验证

```bash
conda activate spagrn-gpu
python -m unittest discover -s test -p 'test_geary_gpu.py' -v
```

回归测试直接对照原函数的逐边加权差分公式，覆盖有向边、重复边、自环、
稀疏/稠密输入、不同批次、跨 TF 块、零/非零常量基因、空输入和输出顺序。
没有 CUDA 时运行 CPU 测试并明确跳过 CUDA 测试；在 GPU 服务器重复运行以验证 CUDA 路径。
不需要导入整个 SpaGRN 即可执行数值测试。

浮点求和顺序改变，结果不保证逐位相同。此改动保留仓库原 GPU 函数的公式，
不另行修改统计定义、基因筛选或下游 motif 富集。
实际加速倍数和峰值显存必须在真实 A100 数据上测量，不能由 CPU 测试推断。

## 运行及缓存

首次比较新旧速度请用独立输出目录，以免 `cache=True` 直接读取已有 `geary_adj.csv`。
此版本仍无阶段内部断点；中断会丢失当前未返回/保存的关联结果。
全部 TF×靶基因结果仍存放在主机内存，CSV 写入开销仍存在。
CPU 表达块提取、TF 块上传也仍有开销；本次优先移除最昂贵的边级重复展开。

本地修改不会自动同步 GitHub。镜像继续从 GitHub 克隆时，必须先将本版本提交推送，
或者直接将此目录作为镜像构建输入。

## 本次本机检查记录

- Python 3.8 语法检查通过；安装脚本 `bash -n` 通过。
- PyTorch 2.7.1 CPU 下，30 组随机图的 CSR 稀疏乘法与原逐边分子对照通过。
- 完整 DataFrame 回归测试需 numpy/pandas/scipy/PyTorch/tqdm；GPU 镜像安装脚本会执行它。
- 本机没有 CUDA；未验证 Python 3.8 + PyTorch 2.4.1 + A100 的完整运行或加速倍数。

## 新增：Step 5 和 Step 6 可选 GPU 后端

此功能在 `/Users/sakura/Downloads/SpaGRN-push` 中实现。原有 CPU 行为仍是默认值；
更新镜像后，在原来的 `infer(...)` 参数中追加：

```python
num_workers=4,                   # 你的容器 CPU 配额约 4 核
# 原来的 torch_device='cuda:0' 保留，不要重复传参
cistarget_backend='gpu',         # Step 5：GPU AUC 与 recovery curves
aucell_backend='gpu',            # Step 6.0 和 6.1：GPU 排名及评分
aucell_cell_batch_size=512,      # 每次处理 512 个细胞；显存不足可降低
aucell_seed=42,                  # 固定并列值打破顺序，便于比较
```

示例是追加/替换现有同名参数，不要在同一次调用中重复写关键字。
`cistarget_backend`、`aucell_backend` 分别支持 `'cpu'` 和 `'gpu'`。
两个 GPU 后端都是本项目新增实现，并非 pySCENIC 官方 CUDA 参数。

### Step 5 实现范围

- 用单进程调度 GPU，每个数据库加载一次所有合格模块所需的排名列，缓存到 GPU。
- GPU 计算 motif AUC、加权 recovery curves、曲线均值和总体标准差。
- CPU 继续执行原 pySCENIC 的 NES 向量运算、注释合并、leading-edge 靶基因筛选和结果整理。
- 只将保留的 motif recovery curves 复制回 CPU，而非全部曲线。
- 按原版 `>=20%` 缺失规则提前跳过不合格模块，不改变原有丢弃条件。
- 一次只缓存一个数据库，预计排名缓存超过当前空闲显存的 60% 时提前报错。
  此检查不保证剩余临时数组一定不会 OOM；没有自动降级为 CPU。
- 只支持 pySCENIC 0.12.1；约束文件固定 ctxcore 0.2.0。
- 通过私有函数副本注入数值内核，不修改已安装的 pySCENIC 文件，也不全局替换其函数。
- 不再使用原 Dask 多进程调度，所以 `num_workers` 不控制 GPU 核心计算。
- 数据库初次读取、CPU 注释和 leading-edge 仍可能是瓶颈；不能保证整步总耗时都按 GPU 内核比例加速。

### Step 6 实现范围

- 细胞分块，GPU stable argsort，严格保持固定随机种子下的 pandas 列打乱和并列排名规则。
- GPU 稀疏签名矩阵乘法计算加权 AUC，保留 ctxcore 的 cutoff 舍入及边界约定。
- 保留匹配不足 80% 返回零、noweights 和 normalize 的行为。
- Step 6.0 与 receptor AUCell 都支持 GPU，但当前两个调用仍分别排名；尚未合并复用排名。
- 均使用 float64。完整表达 DataFrame 和最后输出仍占用 CPU 内存。

### 验证与启用

```bash
python -m unittest discover -s test -p 'test_*gpu*.py' -v
```

本地以 PyTorch CPU 内核对照 pySCENIC 0.12.1 / ctxcore 0.2.0；
验证 AUCell 权重、normalize、固定种子、并列值、NaN/无穷值、零细胞表达、缺失签名，
以及 cisTarget 的 AUC、recovery、曲线均值/标准差、最终模块表和 leading-edge 靶基因结果。
有 CUDA 时相同测试自动增加 CUDA 路径。
本机无 CUDA，尚未实测 A100 兼容性、峰值显存及加速倍数；首次使用应先在 GPU 服务器运行测试，
再以少量真实模块与原版核对，最后全量运行。阈值附近的浮点差异仍可能影响模块筛选。

已有任务不会自动切换后端。更新镜像并重启后，可沿用完全相同数据/参数的缓存目录：
`cache=True` 读取已保存的 Geary 和 modules；如果 motifs.csv/auc_mtx.csv 已存在，也会直接读缓存，
不会因为后端选为 gpu 就重算。若要测量 GPU 速度请使用独立目录或明确管理各阶段缓存。
Step 5 仍不具备模块级断点保存，中断会丢失本阶段尚未返回的进度。
