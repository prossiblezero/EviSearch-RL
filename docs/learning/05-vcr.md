# 05：模型说自己读懂了，能不能核对

若模型写出“这篇文档回答第 2 跳，答案是 X”，就给了训练器一个可检查的中间判断。这是 [training/claims.py](../../evisearch/training/claims.py) 与 [training/vcr.py](../../evisearch/training/vcr.py) 的职责。

## 核验规则

正面声明正确，需要标题属于题目语料、跳号有效、文档对应该跳标注支持文档，以及声明答案与子答案的词频 F1 ≥ 0.5。

负面声明正确，需要该文档不是任何一跳的支持文档。注意这比“不是当前这一跳的支持文档”更严格。不存在的标题、越界跳号算错。

每轮把正确声明记 +1、错误声明记 −1，然后求和并截到 [−1, 1]。无声明为零。若一正一错正好抵消，也为零；所以实现中的“非零声明轮次占比”不严格等于“解析到任何声明的轮次占比”。

## 直接跑验证器

```bash
python - <<'PY'
from evisearch.data.teacher import contract_task
from evisearch.training.claims import verify, turn_rewards
task = contract_task()
task['decomposition'] = [dict(answer='Example University', support_id='d1')]
good = '"Journal Z" answers step 1: Example University.'
bad = '"Unrelated Institute" answers step 1: Example University.'
assert verify(task, good) == [('answers', True)]
assert verify(task, bad) == [('answers', False)]
assert turn_rewards(task, [dict(role='assistant', content=good)]) == [1.0]
assert turn_rewards(task, [dict(role='assistant', content=bad)]) == [-1.0]
print('PASS: same answer, different source, opposite claim rewards')
PY
```

这里两个声明的答案文字相同，但来源不同。VCR 不只是检查生成文本里是否出现了正确答案。

## 从轮次奖励到训练优势

对于 assistant 第 t 轮：

```text
A_token = A_GRPO + 0.5 × c_t
```

该轮的所有 assistant token 使用这一增量，工具 observation 的优势为零。`assistant_spans()` 从 TRL 实际 tool mask 找到轮次边界；如果消息轮数与 mask 对不上，代码直接失败，避免静默把奖励加到错误位置。

VCR 不是额外奖励模型，也没有调用线上大模型裁判。代价是依赖训练数据的分解、支持文档和子答案标签。

## 它还没有验证什么

固定正则无法验证任意自由文本推理；没有匹配的句子不会自动得到错误惩罚。支持文档标签也不能验证所有可能的有效论证。VCR 信号可以与负终局奖励并存，不应把它宣传成“失败轨迹绝无正过程优势”的通用保证。

当前结果更支持“停止与证据选择行为变化”，没有证明相对同起点 GRPO 的 EM 稳定改善。为什么选择研究这个信号，见[演化故事](../evolution.md)；如何判断结论，见下一课。

下一课：[评测与归因](06-evaluation.md)。
