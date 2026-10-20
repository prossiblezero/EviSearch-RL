# 02：看到标题，不等于拥有证据

这节只读 [environment.py](../../evisearch/environment.py) 和 [retrieval.py](../../evisearch/retrieval.py)。先理解环境，再理解任何 RL 算法。

## 三个集合

| 状态 | 哪个动作增加它 | 能做什么 |
|---|---|---|
| `discovered_docs` | `search(query)` | 允许阅读该文档 |
| `read_passages` | `read(doc_id, page)` | 允许保存对应原文片段 |
| `selected_evidence` | `add_evidence(passage_id)` | 允许在最终答案引用 |

BM25 只决定候选排序，按原始文档 ID 打破同分，不使用支持标签。搜索结果是标题与最多 32 词的预览；阅读返回最多 400 词的一页、原文字符偏移和 `passage_id`。保存返回 `evidence_id`，最终引用需要精确匹配。

分开这三步，才可以区分“模型搜到过”“模型读过”“模型选择作为证据”。否则，一个只碰巧出现在搜索结果中的文档就可能被当成模型核验过的证据。

## 合法性是状态约束

`read()` 拒绝未发现的文档；`add_evidence()` 拒绝未读的片段；`_parse_final()` 拒绝伪造或重复引用。重复保存同一片段不会生成多份证据，但仍占一次调用预算，避免用重复动作获取奖励。

工具预算为 search/read/add_evidence = 3/8/8。每次工具返回都附带 `remaining_calls`。批量独立动作可以一起提交，有依赖的动作必须等前一步 observation：尚未知道 `passage_id` 时就提交保存，不是有效的并行。

## 契约是怎么跑的

安装所需环境和分词器后：

```bash
python check_data.py
```

检查会真实执行搜索、阅读、保存，故意尝试未读引用、伪造结果、重复引用、超预算、跨 episode 状态污染。它还调用当前安装的 TRL `_tool_call_loop`，验证单步和批量工具调用，而不是只检查工具字符串能否生成。

## “证据完整”还不等于“证明答案”

MuSiQue 的证据评分基于标注支持文档集合。覆盖正确 ID 是可执行的监督信号，但不是对任意引用句子的蕴含验证。MedHop 扩展更弱：药物与蛋白质的实体共现路径只是代理，不能解释成临床关系证明。

自检：若答案正确但引用里加入无关文档，证据 precision 会降低；若引用一个没有保存的正确文档，合法性检查先失败。两者不是同一类错误。

下一课：[数据与 SFT](03-data-and-sft.md)。
