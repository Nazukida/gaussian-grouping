# Gaussian Grouping 输出结果解读指南

## 一、运行概况

本次实验由 `run_all` 自动化脚本于 **2026-07-09 17:58** 在 AutoDL 云服务器（A100 40GB）上启动，**2026-07-10 03:24** 结束，总耗时约 **9.5 小时**，共处理 5 个场景，全部成功完成训练和渲染。

| # | 场景 | 数据集类型 | 分辨率 | 训练模式 | 物体移除 | 物体修复 | 状态 |
|---|------|-----------|--------|---------|---------|---------|------|
| 1 | bear | 自定义视频 | 1x | 全量训练 | ✓ (id=34) | ✓ | 完成 |
| 2 | counter | MipNeRF360 | 2x | eval split | ✓ (id=49) | — | 完成 |
| 3 | kitchen | MipNeRF360 | 2x | eval split | ✓ (id=49) | ✗ OOM | 移除完成，修复OOM |
| 4 | figurines | LERF-Mask | 1x | train_split+eval | — | — | 完成 |
| 5 | ramen | LERF-Mask | 1x | train_split+eval | — | — | 完成 |
| 6 | teatime | LERF-Mask | 1x | train_split+eval | — | — | 完成 |

> kitchen 场景的物体修复(inpaint)因 CUDA OOM 失败（需分配 3.12 GiB，仅剩 2.77 GiB），其余步骤均正常。

---

## 二、输出目录总览

```
output/
├── run_all_20260709_175854.log          ← 完整运行日志
├── bear.zip                             ← bear 场景打包
├── bear/bear/                           ← bear 场景完整输出
├── mipnerf360/
│   ├── counter/counter_20260709_1758/   ← counter 场景
│   ├── counter.zip
│   └── kitchen.zip                      ← kitchen 仅 zip（解压后同 counter 结构）
├── lerf_mask/lerf_mask/
│   ├── figurines_20260709_1758/         ← figurines 场景
│   ├── ramen_20260709_1758/             ← ramen 场景
│   └── teatime_20260709_1758/           ← teatime 场景
└── lerf_mask.zip
```

---

## 三、单场景输出结构详解

以 bear 场景为完整示例（包含训练 + 渲染 + 移除 + 修复全流程）：

```
bear/bear/
├── cfg_args                              ← 训练配置快照
├── cameras.json                          ← 相机内外参 (96帧, 985×729)
├── input.ply                             ← COLMAP 初始点云
│
├── point_cloud/                          ← 训练产出的3D高斯点云
│   ├── iteration_1000/
│   │   ├── point_cloud.ply              ← 早期检查点
│   │   └── classifier.pth              ← 分类器权重
│   ├── iteration_7000/                  ← 中期检查点
│   └── iteration_30000/                 ← 最终模型 ★
│
├── point_cloud_object_removal/           ← 物体移除后的点云
│   └── iteration_30000/point_cloud.ply
│
├── point_cloud_object_inpaint/           ← 物体修复后的点云
│   └── iteration_9999/point_cloud.ply
│
├── train/                                ← 训练视角渲染结果
│   ├── ours_30000/                      ← 基础模型渲染
│   ├── ours_object_removal/iteration_30000/  ← 移除后渲染
│   └── ours_object_inpaint/iteration_9999/   ← 修复后渲染
│
└── test/                                 ← 测试视角渲染结果
    ├── ours_object_removal/iteration_30000/
    └── ours_object_inpaint/iteration_9999/
```

---

## 四、各子目录内容含义

每个渲染结果目录 (`ours_XXXX/`) 内包含 6 个子文件夹：

| 文件夹 | 内容 | 由谁生成 | 科研意义 |
|--------|------|---------|---------|
| `renders/` | 模型渲染的 RGB 图像 | `render.py` | 评估重建质量 (PSNR/SSIM/LPIPS) |
| `gt/` | 对应视角的真值图像 | `render.py` (从数据集读取) | 作为比较基准 |
| `objects_feature16/` | 16维物体特征的 PCA 可视化 | `render.py` | 展示分割特征的空间分布 |
| `gt_objects_color/` | 真值分割 mask 的彩色可视化 | `render.py` | SAM/DEVA 伪标签可视化 |
| `objects_pred/` | 模型预测的分割 mask 彩色可视化 | `render.py` | 评估分割精度 |
| `concat/` | 以上 5 列横向拼接 + result.mp4 视频 | `render.py` | **快速总览，一图看全** |

### concat 图从左到右的顺序：
```
[ renders | gt | objects_feature16 | gt_objects_color | objects_pred ]
   渲染     真值      特征PCA           真值分割          预测分割
```

---

## 五、各程序与配置的对应关系

### 5.1 训练阶段

| 程序 | 配置 | 产出 |
|------|------|------|
| `train.py` | `config/gaussian_dataset/train.json` + 命令行参数 | `point_cloud/iteration_*/` + `classifier.pth` |

关键配置 (`config/gaussian_dataset/train.json`):
- `num_classes: 256` — 最大物体类别数
- `densify_until_iter` — 高斯致密化截止步数
- `reg3d_interval/k/lambda_val` — 3D正则化超参

命令行关键参数:
- `-s data/<dataset>` — 数据源路径
- `-r <scale>` — 图像分辨率倍率 (1=原始, 2=半分辨率)
- `--eval` — 启用 train/test 分割
- `--train_split` — LERF-Mask 场景使用的分割方式

### 5.2 分割渲染阶段

| 程序 | 配置 | 产出 |
|------|------|------|
| `render.py` | `--num_classes 256` | `train/ours_30000/` 和 `test/ours_30000/` 下全部子目录 |

### 5.3 物体移除阶段

| 程序 | 配置 | 产出 |
|------|------|------|
| `edit_object_removal.py` | `config/object_removal/<scene>.json` | `point_cloud_object_removal/` + `ours_object_removal/` 渲染 |

配置示例 (`config/object_removal/bear.json`):
```json
{
    "num_classes": 256,
    "removal_thresh": 0.3,    // 凸包内判定阈值
    "select_obj_id": [34]     // 要移除的物体 ID
}
```
- 原理：根据 `select_obj_id` 找到对应的 3D Gaussians，构建凸包，将凸包内的高斯移除

### 5.4 物体修复阶段

| 程序 | 配置 | 产出 |
|------|------|------|
| `edit_object_inpaint.py` | `config/object_inpaint/<scene>.json` | `point_cloud_object_inpaint/` + `ours_object_inpaint/` 渲染 |

配置示例:
```json
{
    "num_classes": 256,
    "removal_thresh": 0.3,
    "select_obj_id": [34],
    "images": "images_inpaint_unseen",    // 2D修复后的图像目录
    "object_path": "inpaint_object_mask_255",  // 修复区域mask
    "lambda_dlpips": 0.5,                 // LPIPS感知损失权重
    "finetune_iteration": 10000           // 微调步数
}
```
- 原理：移除物体后，用 LaMa 等 2D inpainting 方法生成伪标签，再微调剩余高斯使 3D 场景一致

---

## 六、训练指标汇总 (PSNR @ 30k iterations)

| 场景 | Test PSNR | Train PSNR | 备注 |
|------|-----------|------------|------|
| counter | 28.50 | 30.90 | 室内台面场景，半分辨率 |
| kitchen | 30.48 | 33.44 | 室内厨房场景，半分辨率 |
| figurines | 26.64 | 26.77 | 桌面小物件，全分辨率 |
| ramen | 32.51 | 28.34 | 拉面场景，全分辨率 |
| teatime | 29.19 | 30.40 | 茶具场景，全分辨率 |

> bear 场景未开 eval，无独立 test PSNR（全部数据参与训练）。

---

## 七、推荐的科研审阅顺序

### 第1步：快速浏览全局效果
1. 打开 `bear/bear/train/ours_30000/concat/result.mp4` — 一个视频看完 RGB渲染、特征可视化、分割预测
2. 对比 `ours_object_removal` 和 `ours_object_inpaint` 的 concat 视频 — 看编辑效果

### 第2步：评估重建质量
1. 查看 log 中各场景的 **PSNR 指标**（见上表）
2. 对比 `renders/` vs `gt/` 中的同名图像 — 看渲染 vs 真值差异
3. 如需定量指标，运行 `metrics.py`（项目未在本次 run 中执行）

### 第3步：评估分割质量
1. 查看 `objects_pred/` — 预测分割是否连贯、边界清晰
2. 对比 `objects_pred/` vs `gt_objects_color/` — 预测 vs 伪标签
3. 查看 `objects_feature16/` — 特征空间是否有清晰的物体边界

### 第4步：评估编辑效果（bear + counter 场景）
1. **物体移除**: 对比 `ours_30000/renders/` vs `ours_object_removal/.../renders/`
   - 关注：移除区域是否有残影、边界是否自然
2. **物体修复** (仅 bear): 对比移除 vs 修复
   - 关注：修复区域是否填充合理、多视角一致性

### 第5步：LERF-Mask 开放词汇分割（待补充）
- figurines/ramen/teatime 三个场景已训练完成
- 但 `segment_anything` 未安装，故 `render_lerf_mask.py` 评估未执行
- 需后续运行该脚本 + `script/eval_lerf_mask.py` 得到 mIoU 指标

### 第6步：3D点云可视化
- 用 CloudCompare 或 MeshLab 打开 `.ply` 文件
- 对比 `point_cloud/iteration_30000/` (原始) vs `point_cloud_object_removal/` (移除后) vs `point_cloud_object_inpaint/` (修复后)

---

## 八、已知问题与后续工作

| 问题 | 影响 | 建议 |
|------|------|------|
| kitchen inpaint OOM | 该场景缺少修复结果 | 降低分辨率(-r 4)或减少 finetune batch |
| segment_anything 未安装 | LERF-Mask 评估未跑 | `pip install segment_anything` 后重跑 |
| bear 无 test split | 无法报告 Novel View PSNR | 如需论文数据，重训加 `--eval` |
| kitchen 仅有 zip | 需解压查看 | `unzip output/mipnerf360/kitchen.zip` |

---

## 九、文件大小参考

- 单个 `point_cloud.ply` (30k iter): ~200-500 MB
- 单帧渲染图 (985×729): ~500 KB
- concat 视频 (result.mp4): ~5-20 MB
- 整个 output 目录: 数十 GB 级别

---

## 十、复现命令速查

```bash
# 训练 (以 bear 为例)
python train.py -s data/bear -r 1 -m output/bear \
    --config_file config/gaussian_dataset/train.json

# 分割渲染
python render.py -m output/bear --num_classes 256 --images images

# 物体移除
python edit_object_removal.py -m output/bear \
    --config_file config/object_removal/bear.json

# 物体修复
python edit_object_inpaint.py -m output/bear \
    --config_file config/object_inpaint/bear.json

# LERF-Mask 评估 (需 segment_anything)
python render_lerf_mask.py -m output/lerf_mask/figurines_20260709_1758 \
    --num_classes 256 --images images
python script/eval_lerf_mask.py
```
