# AutoDL 从零部署到跑通完整清单

> 覆盖两个项目：`gaussian-grouping`（CUDA 11.x / Python 3.8）与 `semanticfoam`（CUDA 12.x / Python 3.11）。
> 二者 CUDA 与 Python 版本不同，**必须用两个独立 conda 环境**。AutoDL 底层驱动向下兼容，一台实例即可同时容纳。

---

## 背景：为什么要换卡

原服务器 output 两个项目**全部因显存不足（CUDA OOM）失败**：

- gaussian-grouping：6 个场景崩 5 个，均在 5% 迭代内（2900~5600 / 100000）OOM，只存下 iteration_1000，无任何分割/编辑产出。
- semanticfoam：5 个场景全崩，`test/` 全空（无 PSNR/IoU/渲染图/视频），日志显示 **GPU 被多个进程共用，仅剩 16MB 空闲**。

根因：卡为 24GB 级别 + GPU 被抢占。解决：租 **48GB 独占卡**。

---

## 第 0 步：租实例（一次性）

| 项 | 建议 |
|---|---|
| GPU | A40 / L20 / A6000（**48GB**）×1 |
| 镜像 | `Miniconda3` + `CUDA 12.1` 基础镜像 |
| 系统盘 | 扩到 **100GB** |

**记下你显卡的算力号（编译要用）**：

| 显卡 | 算力号 `TORCH_CUDA_ARCH_LIST` |
|---|---|
| A40 / A6000 | `8.6` |
| L20 / 4090 | `8.9` |
| A100 | `8.0` |
| V100 | `7.0` |

---

## 第 1 步：通用准备

```bash
# AutoDL 学术加速（github/pytorch 提速，强烈建议）
source /etc/network_turbo

# 代码/数据放数据盘，避免占满系统盘、关机不丢
cd /root/autodl-tmp

# 上传并解压两个项目（不要传 output.zip / outputsem.zip，太大且无用）
# 解压后应得到：
#   /root/autodl-tmp/gaussian-grouping
#   /root/autodl-tmp/semanticfoam

# 确认 CUDA 路径（后面 semanticfoam 编译要用）
ls /usr/local | grep cuda        # 通常是 cuda-12.1
```

---

## 第 2 步：环境 A —— gaussian-grouping（CUDA 11.x / Py3.8）

```bash
conda create -n gaussian_grouping python=3.8 -y
conda activate gaussian_grouping

# torch 1.12.1 + cu116（对应原日志 CUDA 11.6）
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116

# Python 依赖
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

# 编译两个 CUDA 子模块（最易出错，先设算力号，A40 为例 = 8.6）
export TORCH_CUDA_ARCH_LIST="8.6"
cd /root/autodl-tmp/gaussian-grouping
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

**验证：**

```bash
python -c "import torch; print('cuda ok:', torch.cuda.is_available(), torch.version.cuda)"
python -c "import diff_gaussian_rasterization, simple_knn; print('submodules ok')"
```

两行不报错 → 环境 A 通了。

---

## 第 3 步：环境 B —— semanticfoam（CUDA 12.x / Py3.11）

```bash
conda deactivate
conda create -n semanticfoam python=3.11 -y
conda activate semanticfoam

# torch cu128（README 指定）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Python 依赖
cd /root/autodl-tmp/semanticfoam
pip install -r requirements.txt

# 编译 C++/CUDA 后端 RadFoamBindings（submodule 已随代码带齐，无需 git submodule init）
export CUDA_HOME=/usr/local/cuda-12.1     # 按上面 ls 的实际路径改
export TORCH_CUDA_ARCH_LIST="8.6"         # A40=8.6，改成你的卡
pip install -e .                          # 会 cmake 编译几分钟，正常
```

**验证：**

```bash
python -c "import torch, radfoam; print('semanticfoam ok:', torch.cuda.is_available(), torch.version.cuda)"
```

> `nvcc not found` → CUDA_HOME 路径错，重设后再 `pip install -e .`。

---

## 第 4 步：准备数据目录

### gaussian-grouping —— `data/`

```
gaussian-grouping/data/
├── bear/                                  # 每场景含 images/ + sparse/ + object_mask/
├── lerf_mask/{figurines,ramen,teatime}/
└── mipnerf_360/{counter,kitchen}/
```

### semanticfoam —— `data/`（README 要求）

```
semanticfoam/data/<scene>/
├── images/                      # RGB 图
├── sparse/0/                    # COLMAP: cameras.bin images.bin points3D.bin
├── object_mask/                 # DEVA 分割图
└── segmentation_labels/masks/   # 每物体二值 mask（算 IoU 用，可选）
```

只有原始图时，用自带脚本生成 COLMAP：

```bash
# gaussian-grouping
python convert.py -s data/<scene>
# semanticfoam
python prepare_colmap_data.py --data_dir data/<scene>
```

---

## 第 5 步：训练（已加防 OOM 参数）

### A. gaussian-grouping

```bash
conda activate gaussian_grouping
cd /root/autodl-tmp/gaussian-grouping
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 防碎片，替代原来的 NO_CACHING

# 普通场景（bear / mipnerf），iterations 改回 30000（别再用 100000）
python train.py -s data/bear -r 1 -m output/bear \
  --config_file config/gaussian_dataset/train.json \
  --iterations 30000

# LERF 场景要加 --train_split
python train.py -s data/lerf_mask/figurines -r 1 -m output/lerf_mask_figurines \
  --config_file config/gaussian_dataset/train.json \
  --iterations 30000 --train_split

# 训练后渲染分割结果（项目卖点产出）
python render.py -m output/bear --num_classes 256
```

> 48GB 仍紧张的超大场景：train.py 加 `--data_device cpu`（图像放内存，省显存）。

### B. semanticfoam

每个场景一份 config。基于模板 `configs/mipnerf360.yaml` / `configs/lerf.yaml` 复制后改 `scene:`、`data_path:`、`experiment_name:` 三行：

```bash
conda activate semanticfoam
cd /root/autodl-tmp/semanticfoam
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 例：造 counter 的 config
cp configs/mipnerf360.yaml configs/mipnerf360_counter.yaml
# 编辑 configs/mipnerf360_counter.yaml：
#   scene: "counter"
#   data_path: "data/mipnerf_360"     # 指向包含 counter/ 的父目录
#   experiment_name: "counter"

# 训练（iterations 已是 20000）
python train.py -c configs/mipnerf360_counter.yaml
```

> 原来 5 个场景全崩根因是 GPU 被共用（仅剩 16MB）。AutoDL 单实例独占 GPU，问题天然消失。
> 若 figurines 仍报 `illegal memory access (delaunay)`，把该 config 里 `radiance_batch_size` / `segmentation_batch_size` 从 `900000` 降到 `400000`。

---

## 第 6 步：评测 / 出图出视频（semanticfoam）

```bash
# 完整评测：渲染测试视图 + PSNR/SSIM/LPIPS + 分割 IoU + 提取物体 + 360 视频
python test.py -c output/counter/config.yaml \
  --eval_segmentation \
  --trajectory_type 360
```

产出写入：`output/counter/test/`（RGB、误差图、分割图）、`metrics_summary/`（指标）、`objects/`、`videos/`。这些即可放入报告/答辩。

---

## 常见坑速查

| 现象 | 原因 | 解决 |
|---|---|---|
| 子模块 `pip install` 报 nvcc / arch 错误 | 没设算力号 | `export TORCH_CUDA_ARCH_LIST="8.6"`（用你的卡号） |
| semanticfoam `pip install -e .` 找不到 CUDA | CUDA_HOME 没设对 | `ls /usr/local\|grep cuda` 确认后重设 |
| 训练中途 OOM | batch/分辨率过大 | 加 `expandable_segments`；降 batch_size；gaussian 加 `--data_device cpu` |
| gaussian 训练极慢 / OOM | 迭代设成 100000 | 改回 `--iterations 30000` |
| github/pip 下载慢 | 没开加速 | `source /etc/network_turbo` |
| 关机后环境/数据没了 | 装在系统盘 | 代码/数据放 `/root/autodl-tmp`（数据盘持久） |

---

## 成本预估

- 两项目各约 6 场景，单场景 20~40 分钟（30k/20k 迭代），全部训练 + 评测约 **8~12 小时**。
- A40 按 ~3 元/时算，跑通一整轮 ≈ **25~40 元**。留调试余量，**充 100 元**足够。
