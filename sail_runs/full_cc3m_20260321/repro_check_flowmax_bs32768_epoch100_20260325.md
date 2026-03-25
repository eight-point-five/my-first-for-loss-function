# SAIL 新损失函数（FlowMax）训练复现核查报告（bs32768, epoch_100）

## 1. 核查目标
- 对象：`sail_flowmax_cfg_on_cc3m_clipb32_bs32768_20260325` 训练产物。
- 目标：在与 baseline（`sail_official_cfg_on_cc3m_clipb32_bs32768_20260323`）相同数据与训练参数下，对比新损失函数 FlowMax 的效果差异。

## 2. 本次评测使用的模型与产物
- 训练 checkpoint：`/home/aiscuser/SAIL/logs/sail_flowmax_cfg_on_cc3m_clipb32_bs32768_20260325/checkpoints/epoch_100.pt`
- 评测结果 JSON：
  - `SAIL/evaluation/eval_result/COCO/sail_flowmax_cfg_on_cc3m_clipb32_bs32768_20260325/alignment_probing.json`
  - `SAIL/evaluation/eval_result/imagenetv1/sail_flowmax_cfg_on_cc3m_clipb32_bs32768_20260325/alignment_probing.json`
  - `SAIL/evaluation/eval_result/winoground/sail_flowmax_cfg_on_cc3m_clipb32_bs32768_20260325/alignment_probing.json`
- 评测日志（本次）
  - COCO：`/home/aiscuser/sail_runs/full_cc3m_20260321/eval_flowmax_bs32768_epoch100_coco.log`
  - ImageNetv1：`/home/aiscuser/sail_runs/full_cc3m_20260321/eval_flowmax_bs32768_epoch100_imagenetv1.log`
  - Winoground：`/home/aiscuser/sail_runs/full_cc3m_20260321/eval_flowmax_bs32768_epoch100_winoground.log`

## 3. 本次实测结果（FlowMax, epoch_100）
- COCO
  - T2I R@1 = 27.608
  - I2T R@1 = 38.800
- ImageNetv1
  - Top-1 = 49.918
- Winoground
  - Text = 31.25
  - Image = 10.25
  - Group = 7.50

## 4. 与 baseline（SigLip, 同配置）对比
参考 baseline：`sail_official_cfg_on_cc3m_clipb32_bs32768_20260323` 的 epoch_100 结果。

### 4.1 COCO / ImageNet
- T2I R@1：27.608 vs 36.436（Δ -8.828）
- I2T R@1：38.800 vs 50.880（Δ -12.080）
- IN-1K Top-1：49.918 vs 57.202（Δ -7.284）

### 4.2 Winoground
- Text：31.25 vs 27.25（Δ +4.00）
- Image：10.25 vs 11.25（Δ -1.00）
- Group：7.50 vs 8.50（Δ -1.00）

## 5. 复现结论
- 结论1（对“是否在当前配置下优于 baseline”）：**整体未优于 baseline**；检索与 ImageNet 指标均下降。
- 结论2（对“新损失函数带来的行为差异”）：FlowMax 在 Winoground Text 上有提升，但在主检索与分类指标上出现明显回落，表现为更偏文本判别而弱化跨模态检索与分类泛化。
- 结论3（工程链路稳定性）：训练与评测链路均可跑通；为兼容当前环境（PyTorch 2.6 + 同名 backbone 缓存）已做最小兼容修复，结果可复现。

## 6. 关键工程修复（本次实验涉及）
- `SAIL/model/sail_model.py`
  - `load_vlhead_weights` 显式使用 `torch.load(..., weights_only=False)`（兼容 PyTorch 2.6 默认行为变化）。
- `SAIL/evaluation/coco_zs_retrieval.py`
- `SAIL/evaluation/imagenet_zs_classificaiton.py`
- `SAIL/evaluation/winoground.py`
  - backbone 缓存路径从 `{model_name}/...` 改为 `vision__{model_name}/...` 与 `text__{model_name}/...`，避免视觉/文本同名模型缓存互相覆盖导致维度错误。

## 7. 可直接汇报给学长的一句话
- “在与 baseline 完全同配置（cc3m, bs32768, epoch100）下替换为 FlowMax 后，实验可稳定复现；但相较 SigLip，COCO 检索与 ImageNet 指标均下降，只有 Winoground 文本项有提升，当前不建议直接替换为默认损失。”
