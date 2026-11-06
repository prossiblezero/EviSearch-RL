# 数据集说明与具体样例

样例都直接取自外部归档 `/data3/dzh/MedSearchRL` 下的冻结数据（`data/`）和真实预测（`runs/`），这些产物不在 Git 仓库里，路径中的 `v3`、`v5` 是产物目录名，不是代码版本（见 [artifacts.py](../evisearch/artifacts.py)）；工具返回内容较长的地方做了截断（以"…"标出），其余原文照录。

## 1. 任务来源与划分

主任务是 **MuSiQue** 的通用多跳问答（2–3 跳），**MedHop** 只作辅助领域与迁移评测。

| 划分 | 文件 | 规模 | 2 跳 / 3 跳（3hop1）/ 3 跳（3hop2） | 用途 |
|---|---|---:|---|---|
| SFT | `data/v3/sft.jsonl`、`data/v5/sft_err.jsonl` | 6,000 条对话 | — | 监督微调（第 4 节） |
| RL | `data/v3/rl.jsonl` | 1,024 题 | 781 / 217 / 26 | GRPO 等在线 RL（第 5 节） |
| dev | `data/v3/dev_tasks.jsonl` | 128 题 | — | 早期开发 |
| dev2 | `data/v3/dev2_tasks.jsonl` | 512 题 | 394 / 116 / 2 | 方法筛选 |
| test | `data/v3/test_tasks.jsonl` | 512 题 | 394 / 96 / 22 | 最终评测 |
| 医学辅助 | `data/v3/medical_tasks.jsonl` | 64 题 | — | MedHop 迁移评测 |

各划分在实验前冻结。逐项核对过：RL、dev2、test 两两之间，以及它们与 SFT 父题之间，题目 ID、问题文本和子问题 ID 的重叠都为 0。3hop1 是链式三跳（A→B→C→答案），3hop2 是两支汇合的三跳。

## 2. 一道题长什么样

以 test 的第一题为例（`musique_2hop__367318_7292`）：

```text
问题：What artist was featured on Smooth Jazz Stations, along with Kenny G and the performer of The Dance?
答案：["George Benson"]

金标分解（只在训练和离线审计中使用，模型看不到）：
  第 1 跳：The Dance >> performer                                         → Dave Koz
  第 2 跳：Along with Kenny G and #1, what artist was featured on smooth jazz stations? → George Benson

支持段落（2 篇）：
  [The Dance (Dave Koz album)] The Dance is the fifth studio album by saxophone player Dave Koz. It was
      released by Capitol Records on September 28, 1999. ...
  [Adult contemporary music] In its early years of existence, the smooth jazz format was considered to be
      a form of AC ... For many years, artists like George Benson, Kenny G and Dave Koz had crossover hits
      that were played on both smooth jazz and soft AC stations.

干扰段落（18 篇，标题举例）：Jazz Contemporary, 2 Horns / 2 Rhythm, Arturo Tappin, Soul Train,
      Tijuana Jazz, The Millennium Dance Complex, Barefoot on the Beach, The Jazz Skyline, ...
```

每道题的字段：`question`、`answers`（答案及别名）、`documents`（候选段落，512 道 test 中 508 道为 20 篇、其余 18–19 篇；上例每篇 21–126 个词）、`support_ids`（支持段落，2 跳题 2 篇、3 跳题 3 篇）、`decomposition`（逐跳子问题、子答案和对应的支持段落）。

- **题目**：平均 15.6 个词。干扰段落是 MuSiQue 刻意挑选的，与问题共享实体或主题（上例中都是爵士乐相关），只靠关键词匹配容易读错。
- **子问题**有两种写法：自然语言（`who played thelma in the movie thelma and louise`），或知识库关系式（`Ross Porter >> educated at`）。`#1` 指代上一跳的答案。
- **答案**是短实体，平均 2.75 个词：人名、地名、机构、数字（`11,900`）、度量（`about 575 feet`）等。512 道 test 中 342 道只有一个答案，其余带 2–6 个别名，例如 `["Arthur Peel", "1st Viscount Peel", "Arthur Peel, 1st Viscount Peel"]`、`["al-Razi", "Razi"]`。

更多 test 样例：

| 类型 | 问题 | 答案 | 分解 |
|---|---|---|---|
| 2 跳 | What character in A League of their Own was played by the actress who played Thelma in the movie Thelma and Louise? | Dottie Hinson | Thelma 的扮演者 → Geena Davis；#1 在 A League of Their Own 中的角色 → Dottie Hinson |
| 2 跳 | Who became the president of Frederic Chopin's original country in 1990? | Lech Wałęsa | Chopin 的国籍 → Poland；#1 在 1990 年的总统 → Lech Wałęsa |
| 2 跳 | How many people work in the school where Ross Porter was educated? | 11,900 | Ross Porter >> educated at → University of Oklahoma；#1 的员工数 → 11,900 |
| 3 跳 | What is the farthest home run ever hit in the league where the team that Jim Wilson plays for? | about 575 feet | Jim Wilson 的球队 → Indians；Cleveland #1 所在联盟 → Major League Baseball；#2 最远本垒打 → about 575 feet |

**标注噪声**：MuSiQue 由单跳问题自动拼接而成，存在噪声。例如 "Who is the father of the man who conceived of the 'new' police?" 的第二跳标注是 `#1 >> child`，金标答案 Arthur Peel 是 Robert Peel 的儿子而不是父亲。项目没有修改任何标注。

### 评分（[evaluation/rollout.py](../evisearch/evaluation/rollout.py)、[training/trainer.py](../evisearch/training/trainer.py) 的 `outcome_reward`）

- **EM**：答案小写、去标点、去冠词后与任一别名完全相同。
- **F1**：与最佳别名的词集合 F1。
- **证据 F1 / 证据完整率**：模型最终引用的段落与 `support_ids` 的 F1；"证据完整"指所有支持段落都被引用。
- **RL 终局奖励**：`答案词集 F1 × 证据 F1`；截断、格式错误、引用了未保存的证据、非法工具调用统一记 −2。只答不引得 0 分，工具调用本身不给奖励。

## 3. 模型实际看到什么、怎么做

模型只看到系统提示和问题；候选段落藏在环境里，必须通过工具获取。三个工具（[environment.py](../evisearch/environment.py)）：

| 工具 | 作用 | 返回 |
|---|---|---|
| `search(query)` | 在本题 20 篇段落上做 BM25 检索 | 候选的 `doc_id`、标题和简短预览 |
| `read(doc_id, page)` | 阅读已发现文档的原文 | 原文、字符偏移、`passage_id` |
| `add_evidence(passage_id)` | 保存一段已读原文供引用 | `evidence_id`；重复保存是幂等的 |

每题预算：3 次检索、8 次阅读、8 次保存。最终回复必须恰好是一个 JSON：`{"answer": "...", "evidence_ids": [...]}`，且只能引用已保存的证据。系统提示的关键要求：从问题中明确出现的实体开始、沿中间实体逐跳推进、不要凭记忆跳过中间事实、引用整条证据链而不只是最后一跳、证据不足时回答 `insufficient evidence`。

同一道题上两个模型的真实轨迹（`runs/v5/9b/eval/<模型>/test/predictions.jsonl`）：

**乘性 VCR 9B（EM 1，证据完整）**

```text
[assistant] search("The Dance Kenny G")
[tool]      d4 The Millennium Dance Complex …, d15 The Dance (Dave Koz album) …, …
[assistant] read(d15)
[tool]      The Dance is the fifth studio album by saxophone player Dave Koz. …   passage_id = d15:0:330
[assistant] "The Dance (Dave Koz album)" answers step 1 (performer of The Dance?): Dave Koz.
            add_evidence(d15:0:330)   search("Smooth Jazz Stations Dave Koz Kenny G")
[tool]      evidence_id = e:d15:0:330 …
[tool]      d9 Adult contemporary music: "… artists like George Benson, Kenny G and Dave Koz had crossover hits …"
[assistant] read(d9)
[tool]      In its early years of existence, the smooth jazz format …   passage_id = d9:0:334
[assistant] "Adult contemporary music" answers step 2 (who else was on Smooth Jazz Stations with Kenny G and Dave Koz?): George Benson.
            add_evidence(d9:0:334)
[assistant] {"answer":"George Benson","evidence_ids":["e:d15:0:330","e:d9:0:334"]}
```

**未微调 Qwen3.5-9B（EM 0）**：检索和阅读路径几乎相同，也读到了两篇支持段落，但只保存了第二篇，最后回答 `"Dave Koz"`——把第一跳的桥接实体当成了最终答案。这是多跳问答最典型的错误：中间实体也出现在最终段落里，模型没有分清问的是哪一跳。

轨迹里引号开头的那句就是相关性推理句（ERR），由 SFT 数据教会（第 4 节）。VCR（[training/claims.py](../evisearch/training/claims.py)）在训练时把这些句子与金标分解逐条核对、作为过程奖励；评测时只用于离线计算声明准确率。

## 4. SFT 训练数据

6,000 条对话，按"完整对话数"和"assistant 监督 token 数"分别计数（两者比例不同，不能混用）。下表为当前主线使用的 `data/v5/sft_err.jsonl`：

| 类别 | 对话数 | assistant token | 来源 | 构造方式 |
|---|---:|---:|---|---|
| agent_general | 1,320 | 472,974 | MuSiQue | 教师在真实工具环境中生成的多跳检索轨迹 + 相关性推理句；其中 631 条为补救增强示范 |
| agent_medical | 480 | 199,707 | MedHop | 教师在真实工具环境中生成的医学检索轨迹 |
| tool | 600 | 91,916 | MuSiQue 单跳子问题 | 程序化构造的单跳"检索→阅读→保存→作答"示范 |
| general | 1,800 | 357,313 | No Robots | 人工撰写的通用多轮对话 |
| long | 600 | 89,642 | MuSiQue 400 条、No Robots 200 条 | 给定文本的阅读理解问答（MuSiQue 部分直接给出段落作答，不调用工具） |
| code | 900 | 215,755 | Magicoder-OSS-Instruct 600 条、No Robots 300 条 | 代码解释与编写 |
| safety | 300 | 22,087 | PKU-SafeRLHF-10K | 来源中人工标注为安全的回答 |

合计监督 token 约 145 万，最长序列 7,895 tokens，不截断最终答案。非 Agent 数据的作用是保持通用能力，防止只会做这一个任务。

**损失掩码**：只监督 assistant 自己生成的 token，系统提示、用户问题和工具返回的标签全部为 −100。例如单工具样例 `musique_2hop__401835_332063_tool0` 共 1,694 个 token，只有 153 个参与损失。

**Agent 示范的来源**：DeepSeek 教师在真实工具环境中生成 4,495 条候选轨迹，经格式、引用合法性、证据完整度筛选和确定性回放校验，保留 1,800 条（通用 1,320 + 医学 480）。教师生成时可以参考训练题的金标标注（这些标注从学生提示中移除），属于"特权示范"。在教师示范的基础上，又按子问题分解给每次阅读后补了一句相关性推理（ERR，[data/rationale.py](../evisearch/data/rationale.py)），并为部分题目合成了"先读到干扰文档、判断不相关、再换一篇"的补救示范（RAD，[data/recovery.py](../evisearch/data/recovery.py)）。

### 4.1 通用 Agent 示范（教师轨迹 + 相关性推理句）

`musique_2hop__689512_90382-err`，共 2,564 个 token，其中 331 个为 assistant token：

```text
[user]      Who was the pilot who dropped the atomic bomb on the city where Akinoshu Kenji was born?
[assistant] search("Akinoshū Kenji sumo wrestler born")   search("pilot dropped atomic bomb Hiroshima")
[tool]      d5 Akinoshū Kenji: "… a former sumo wrestler from Hiroshima, Japan …", d12 Fame (duo) …, …
[tool]      d15 Paul Tibbets: "… the pilot who flew the Enola Gay … on the Japanese city of Hiroshima.", d18 Enola Gay …
[assistant] read(d5)   read(d15)
[tool]      Akinoshū Kenji (born 17 January 1969 as Kenji Imada) is a former sumo wrestler from Hiroshima … passage_id = d5:0:292
[tool]      Paul Warfield Tibbets Jr. (23 February 1915 -- 1 November 2007) was a brigadier general … passage_id = d15:0:307
[assistant] "Akinoshū Kenji" answers step 1 (place of birth of Akinoshu Kenji?): Hiroshima.
            "Paul Tibbets" answers step 2 (who was the pilot who dropped the atomic bomb on Hiroshima?): Paul Warfield Tibbets Jr..
            add_evidence(d5:0:292)   add_evidence(d15:0:307)
[tool]      evidence_id = e:d5:0:292 …   remaining_calls: search 1, read 6, add_evidence 7
[tool]      evidence_id = e:d15:0:307 …
[assistant] {"answer":"Paul Warfield Tibbets Jr.","evidence_ids":["e:d5:0:292","e:d15:0:307"]}
```

注意第一轮的第二条检索 `pilot dropped atomic bomb Hiroshima`：在读到"出生于 Hiroshima"之前，教师就已经检索了第二跳。这就是特权示范的副作用——教师知道金标，示范里会提前用上中间答案。[演化故事第 7 章](evolution.md#stage-5b)的诊断发现，学生因此学到了"读到什么就存什么"，读到干扰文档后几乎不会补救。

### 4.2 补救增强示范（RAD + 相关性推理句）

`musique_2hop__20214_841802-rad-err`：在原示范中插入"读到不相关文档 → 写否定判断 → 换一篇"的步骤：

```text
[user]      Who is the sibling of the gospel singer Freddie Mercury cited as an inspiration?
[assistant] search("Freddie Mercury gospel singer inspiration Aretha Franklin")
[tool]      d18 If You Want Me: "… Gospel singer Carolyn Franklin (sister of Aretha Franklin) …", d9 Queen (band) …, …
[assistant] read(d17)
[tool]      This is a discography of works by Freddie Mercury as a solo artist. …   passage_id = d17:0:137
[assistant] "Freddie Mercury discography" does not answer step 1 (What gospel singer did Freddie Mercury cite as an inspiration?); try another document.
            read(d9)   read(d18)
[tool]      Queen drew artistic influence from British rock acts of the 1960s … passage_id = d9:0:697
[tool]      If You Want Me is the final solo studio album recorded by … Carolyn Franklin (sister of Aretha Franklin) … passage_id = d18:0:150
[assistant] "Queen (band)" answers step 1 (What gospel singer did Freddie Mercury cite as an inspiration?): Aretha Franklin.
            "If You Want Me" answers step 2 (sibling of Aretha Franklin?): Carolyn Franklin.
            add_evidence(d9:0:697)   add_evidence(d18:0:150)
[assistant] {"answer":"Carolyn Franklin","evidence_ids":["e:d9:0:697","e:d18:0:150"]}
```

同样能看到特权信息：第一条检索里已经包含了第一跳的答案 "Aretha Franklin"。

### 4.3 单工具示范（程序化构造）

把多跳题的一个子问题单独拿出来，程序化生成"检索→阅读→保存→作答"，用来教工具协议本身：

```text
[user]      Vernon Township >> located in the administrative territorial entity
[assistant] search("Vernon Township >> located in the administrative territorial entity")
[tool]      d19 Bogotá …, …, d6 Vernon Township, Clinton County, Ohio …
[assistant] read(d6)
[tool]      Vernon Township is one of the thirteen townships of Clinton County, Ohio, United States. … passage_id = d6:0:201
[assistant] add_evidence(d6:0:201)
[assistant] {"answer": "Clinton County", "evidence_ids": ["e:d6:0:201"]}
```

### 4.4 医学 Agent 示范（MedHop）

```text
[user]  In this MedHop literature task, which candidate drug interacts with DB08879? Candidates: DB00036, DB00065,
        DB02709, DB04881, DB04941, DB05187, DB05424, DB06243, DB06802. Return one DrugBank ID and cite the
        retrieved evidence chain. Drug/protein IDs in abstracts are normalized identifiers.
…（多轮检索、阅读文献摘要、保存证据）…
[final] {"answer":"DB00065","evidence_ids":["e:d9:0:1554","e:d3:0:1783"]}
```

医学评测题平均 33.6 篇文献摘要（17–64 篇）。MedHop 没有人工标注的证据链，项目用"药物—蛋白实体共现"作为证据代理（`evidence_kind = entity_cooccurrence_proxy`），**它不代表经过验证的医学蕴含**。

### 4.5 非 Agent 数据

| 类别 | 用户 | 助手（节选） |
|---|---|---|
| general（多轮） | I had a great day today! / That's true. We have to know how to get through the hard times. / Do you think tomorrow will be a good day too? | I'm happy to read that! As Helen Keller once said, … |
| long | What happened on January 5, 2018, according to the attached text?（后附一篇人物传记） | On January 5, 2018, Patricia Gualinga was the victim of a home invasion. |
| code | I'm trying to understand PyTorch … Can you explain this Python code for me? `class Network(nn.Module): …` | Absolutely! What you see in that Python code is the beginnings of a neural network built in PyTorch. … |
| safety | 有害请求（如盗取他人社交账号） | 来源中人工标注为"安全"的回答；项目没有对这些标签重新审核 |

## 5. RL 训练数据

`data/v3/rl.jsonl`，1,024 道与 SFT 去重的 MuSiQue 题。每条只有两部分：

```json
{
  "prompt": [
    {"role": "system", "content": "Answer the question using the local research documents. …（与评测相同的系统提示）"},
    {"role": "user",   "content": "How long is a governor's term in the state WNSP broadcasts to?"}
  ],
  "task": "{\"id\": \"musique_2hop__129680_84128\", \"question\": …, \"answers\": …, \"documents\": …,
            \"support_ids\": …, \"decomposition\": …}"
}
```

- `prompt` 是策略唯一能看到的内容。
- `task` 交给环境：`documents` 构成本题的检索库，`answers` 与 `support_ids` 用于计算终局奖励，`decomposition` 只供 VCR 核对声明。
- 训练时每题采样 4 条完整轨迹组成一组（GRPO 组内标准化），1,024 组共 4,096 条轨迹、2,048 次更新；工具由 TRL 原生执行，工具返回的 token 不计入损失。所有对比组的题序、轨迹数和更新数完全相同。

## 6. 面试速答

- **题目是什么**：MuSiQue 多跳问答，每题约 20 篇候选段落（2–3 篇支持、其余是同主题干扰），需要先找到中间实体再找最终答案。
- **答案是什么**：短实体或数字，平均不到 3 个词，约三分之一带别名；按 EM 和词集 F1 评分，并要求引用支持段落。
- **训练数据是什么**：SFT 6,000 条对话，其中 1,800 条是教师在真实工具环境中生成、经回放校验的 Agent 轨迹，600 条单工具示范，其余 3,600 条通用、长文本、代码、安全数据用于保持通用能力；RL 用 1,024 道独立题目，只给问题，奖励由环境根据金标答案和支持段落计算。
- **一个值得主动提的坑**：教师示范是在看到金标的条件下生成的，会提前使用中间答案；学生因此不会补救。后来用相关性推理句（ERR）和补救示范（RAD）修正，再用 VCR 在 RL 阶段直接奖励正确的相关性判断。
