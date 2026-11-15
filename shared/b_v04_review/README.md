# B 冻结 0.4 独立复核记录

总体状态：`completed_with_feature_numerical_finding`。数据、模型输入、15头重算和训练参考预测核对通过；CPU/GPU特征严格数值一致性未通过，保留为待A确认项。不是正式测试结果。

完整解释与可整合方法章节见 [B_v04_review.md](../../report/B_v04_review.md)。

- `summary.json`：数量、误差、21项检查和范围摘要。
- `review.json`：逐模型复核、特征比较、本机环境、源码及配置摘要。
- `real_feature_comparisons.json`：598张真实图片每个视图的最大绝对差。
- `initial_failures.json`：最初严格特征阈值检查失败及诊断记录；未改阈值以宣称通过。
- `protocol_checks.txt`：21项现有检查的原始输出。
- `cpu_requirements.txt`：本机CPU复核环境版本，不替代A的正式GPU依赖。

## 复核命令与输入

以主分支26e107f为基线创建B模块分支，使用独立CPU环境。下载Release的training_bundle.zip，按release_assets.json中的SHA-256校验，在全新目录 `runs/b_review_v04/input/` 解包。

复核还需要校验过的六个官方CIFAR-10训练批次/元数据文件（data_batch_1到5及batches.meta），放在 `runs/_cache/cifar10/cifar-10-batches-py/`；不需要test_batch。官方ResNet-18 V1权重文件放在 `runs/_cache/torch/checkpoints/resnet18-f37072fd.pth`。脚本会独立校验训练文件MD5及完整权重SHA-256，之后再反序列化。请从官方来源取得这些输入。

```powershell
python code/reviews/review_b_v04.py --input runs/b_review_v04/input --output runs/b_review_v04/results
python -m unittest discover -s code -p "test_*.py" -v
```

脚本只读取A的包，独立重算划分、全部真实特征和15个分类头；输出进入独立复核目录。它不调用正式CLI、不重新生成候选图、不加载最终测试标签，不改冻结配置或任何封存产物。目录下新增的复核工具不进入A的顶层冻结源码列表。

数值匹配阈值保持：分类头权重1e-6、分类头logits 1e-5、特征1e-5。CPU特征最大差约2.01e-4，故严格特征匹配标为false；不要只看脚本正常结束就写成“全部通过”。
