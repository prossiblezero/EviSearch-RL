# 03：让监督落在模型的动作上

SFT 的目标，是让模型学会合法操作、引用和中间判断。仅仅把“问题 + 正确答案”放进训练集，不会自动教会它如何搜索、什么时候继续读。

## 教师示范必须是真的

[data/teacher.py](../../evisearch/data/teacher.py) 的 `episode()` 每次都实际执行教师调用。`replay()` 再执行一次完整轨迹，核对每条工具返回是否一致。一个看起来很流畅、但 observation 由模型编写的故事不能通过回放。

历史生产示范使用了答案和证据条件化的教师构造。教师不是从零自主发现所有证据；最终示范恢复成不含这些标签的 actor 提示。这种特权监督可以降低示范采集难度，但会产生偏差：轨迹过于顺利，很少展示读错文档后如何恢复。必须把这一点写清楚。

## 补救示范与相关性句子

[data/recovery.py](../../evisearch/data/recovery.py) 对部分示范插入一次可见干扰文档的阅读，然后通过环境重新生成 observation。注入的干扰阅读动作 loss 为零，不训练模型主动犯这个错；之后的正常行为仍可受监督。

[data/rationale.py](../../evisearch/data/rationale.py) 在阅读后的工具决策前加入简短判断，例如：

```text
"Journal Z" answers step 1 (Who owns Journal Z?): Example University.
"Unrelated Institute" does not answer step 1 (Who owns Journal Z?); try another document.
```

这些句子来自训练题的分解与已经读到的材料。评测时不提供分解；模型必须自己生成判断。最终答案轮仍是一个 JSON，不在其中加入推理段落。

已有冻结输入齐全时，可以在独立生成目录重建增强数据：

```bash
python -m evisearch.data.recovery
python -m evisearch.data.rationale
```

生成产物位于仓库外 `data/release-prepared`，不会覆盖用于现有结果的冻结 SFT。当前代码从已冻结的教师混合数据开始增强；完整原始语料下载与历史候选池重建不包含在这个发布仓库中。具体输入见[复现说明](../reproduction.md)。

## 一个 token 到底算不算 loss

| 内容 | 进入上下文 | 进入 SFT loss |
|---|---|---|
| system/user 提示 | 是 | 否 |
| 工具 observation | 是 | 否 |
| 正常 assistant 工具调用 | 是 | 是 |
| assistant 相关性判断 | 是 | 是 |
| 人工注入的干扰阅读动作 | 是 | 否 |
| 最终答案 JSON | 是 | 是 |

编码时，监督位置的 label 等于对应 token ID，其他位置为 `-100`。已有的特殊注入 mask 必须保留，不能用一个泛化的“所有 assistant 全算 loss”覆盖它。

`python check_data.py` 会用真实分词器检查：工具返回里的原文没有进入监督文本；工具名与最终答案确实进入监督文本。[sft.py](../../evisearch/sft.py) 使用预建 `input_ids/labels`，不重新把 observation 变成标签。

## 样本比不等于 token 比

数据共 6,000 条，但不同类别和不同轨迹长度不同。加入判断句后，条数不变，assistant 监督 token 数会增加。混合配方必须同时报告类别样本数与监督 token 数，不能说“样本数相同，所以 SFT 训练信号完全相同”。

自检：为什么“工具文本不算 loss”不等于“工具文本对学习没用”？因为它仍是模型生成下一步动作的条件，只是不要求模型去预测环境已经提供的内容。

下一课：[在线 RL](04-grpo.md)。
