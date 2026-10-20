# 04：同一道题采四条真实轨迹

[train.py](../../evisearch/train.py) 是三个 RL 方法共同的入口。先看 `specification()`：GRPO、IGPO、VCR 的初始策略与冻结 reference 都指向同一个 SFT adapter；题序、预算、学习率与更新数相同。

## 从执行到奖励

TRL 在线采样，每次工具调用都进入同一个 `EvidenceEnv`。每道题采四条完整轨迹，先根据格式、工具错误、引用与截断检查终局，再计算训练口径的答案 token-set F1 × 证据 F1。硬错误为 −2。

这一点可以避免两个捷径：工具调用次数本身不加分；没有合法证据的正确答案不会获得正常的正奖励。但“合法且引用完整”仍不保证答案推理正确。

## 组内优势

普通 GRPO 将一组奖励中心化并按组标准差缩放。高于同组平均的轨迹被鼓励，低于平均的被抑制。可以直接运行当前实现的算术：

```bash
python - <<'PY'
from evisearch.training.advantages import normalize_group
print(normalize_group([0.0, 0.0, 0.5, 1.0]))
assert normalize_group([0.0, 0.0, 0.0, 0.0]) == [0.0] * 4
PY
```

全组奖励相等时，普通 GRPO 没有任务相对优势；若 KL 项仍在，权重仍可能变化。因此“adapter 改变了”“gradient 不为零”不能证明策略从任务中学到了东西。

当前预算为 1,024 组 × 4 条轨迹，每组复用两次更新，共 2,048 次优化。恢复 RL 必须位于完整的两次更新边界；只恢复一个 adapter 文件不足以恢复 optimizer、采样位置与 reference。

## 信号加在哪里

在 [training/trainer.py](../../evisearch/training/trainer.py) 中：

- `RolloutTrainer` 保留 TRL 原生采样与 loss，负责终局评分、轨迹记录以及 IGPO 的前缀评分。
- `AgentTrainer` 加上实际更新 token 的 KL 审计。
- [training/vcr.py](../../evisearch/training/vcr.py) 再在当前 assistant 轮次上添加声明奖励。

这是按职责划分的同一套训练链，不是三套历史版本。工具 observation 不可训练；实际 token mask 为 `completion_mask * tool_mask`，被截断并屏蔽的轨迹也不能重新获得可训练优势。

## IGPO 与终局奖励有何不同

IGPO 比较观察前后金标答案对数似然的变化，经归一化形成轮次信号。它回答的是“这些观察是否让金标答案更可能”，并不直接回答“策略是否能使用证据正确推理”。本项目保留它作为同条件的过程奖励对照，不把它预设为无效。

检查恢复契约：

```bash
python check_resume.py
```

检查使用小型 CPU 模型，复现当前 Transformers/PEFT 的 root adapter 加载问题，并确认恢复策略不会覆盖或解冻 reference。它不是 2B 模型训练实验。

下一课：[可验证声明奖励](05-vcr.md)。
