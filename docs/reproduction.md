# 运行当前实现

仓库提供一套当前的证据 Agent 与三个 RL 对照。合成教学例子不依赖模型；真实训练与报告需要仓库外的冻结产物。这份说明明确区分两种起点，不把本地已有数据伪装成一次 `git clone` 就能得到。

## 1. 无模型的入口

```bash
python -m evisearch.demo
```

只需要 Python 标准库。例子是固定动作的合成轨迹，不是模型推理成绩。

## 2. 环境与存储

本机使用 `/data3/dzh/MedSearchRL/venv/bin/python`。Python 与依赖版本见 [requirements.txt](../requirements.txt) 和 [environment.freeze.txt](../environment.freeze.txt)。不要安装进其他项目的环境。本次整理没有修改已有环境。

新机器需自行建立隔离环境；torch 使用与 CUDA 对应的 cu128 wheel 源，其余按冻结版本安装。兼容的 NVIDIA GPU、Linux 和本地 Qwen3.5-2B 权重是 GPU 运行前提。本机资源根在 [resources.py](../resources.py) 的 `ROOT`，模型位置由它派生；换机器需要对应提供产物，不能仅改路径就假定数据齐全。

所有缓存、临时文件、日志、权重和 checkpoint 都留在该根目录下。`admit()` 在分配前要求可用显存 ≥ 进程上限 + 6 GiB；训练与评测继续检查 6 GiB 余量。GPU 实验串行执行。

## 3. 冻结输入是什么

所有位置集中在 [artifacts.py](../evisearch/artifacts.py)。其中带数字的**外部存储目录是既有实验身份**，不是需要学习的旧源码，也不被打包到 Git；保留它们是为了不改动冻结数据、hash 和 checkpoint。

| 用途 | 当前本机输入 | 缺少时的影响 |
|---|---|---|
| 模型与 tokenizer | `models/Qwen3.5-2B/` | 只能跑合成 demo；不能跑原生分词器检查或训练 |
| RL 与评测 | `data/v3/rl.jsonl`、各 `*_tasks.jsonl`、`manifest.json` | 不能真实训练或评测 |
| dev2 与协议检查 | `data/v4/manifest.json`、`checks.json`、`dev2_manifest.json`、`protocol.json` | 冻结协议校验失败 |
| 当前 SFT | `data/v5/sft_err.jsonl` 及 `.manifest.json` | 不能训练当前 SFT 起点 |
| 重新构建增强数据 | `data/v3/sft.jsonl`、`data/v2/sft_pool.jsonl` | 不能重新构建 recovery/rationale 示范 |
| 额外教师采集 | 上述种子池、`data/v2/{rl,dev,test}_tasks.jsonl` 及 `data/{dev,test}_tasks.jsonl` | 无法执行原有排重约束 |
| 重算公开结果 | `artifacts.RECORDED` 指定的完整 prediction/config/summary | `report.py` 失败，已提交的 JSON 仍可阅读但不等于本地重新核验 |

这些数据与权重未随源码发布，也未承诺已经提供公开下载地址。当前发行版支持从已有冻结产物继续复现，不包含最早的原始语料下载、种子池筛选和混合语料从零构建流水线。获取/发布数据时需保留来源、许可、冻结划分和 hash；不能临时重抽 test 来凑齐输入。

## 4. 先运行 CPU 检查

以下示例中的 `python` 在本机均指专用解释器。

```bash
python check_data.py
python check_resume.py
python check_release.py
python report.py
```

`check_data.py` 执行原生 TRL 工具循环与 SFT/RL mask 检查；`check_resume.py` 用 CPU 小模型检查 policy/reference 恢复；`check_release.py` 检查发布入口、VCR 信号以及比较与恢复的错误路径。`report.py` 重算既有真实预测，并不训练模型。

这些检查不替代当前代码的完整 GPU smoke 或重训；不要把它们称为“完整实验已复现”。

## 5. 数据构造

复用冻结 SFT 时直接进入下一节。要重新构建增强示范：

```bash
python -m evisearch.data.recovery
python -m evisearch.data.rationale
```

它们读取冻结教师混合数据，输出 `data/release-prepared/sft_rad.jsonl` 和 `sft_err.jsonl` 及 manifest。输出是独立工作产物；不会自动替换冻结数据。接入新训练前先检查回放、类别样本数、assistant token 数和数据 hash。

需要补充教师轨迹时，[data/teacher.py](../evisearch/data/teacher.py) 提供真实工具采集入口。先准备 `MEDSEARCH_TEACHER_API_KEY`；密钥只放在环境中。默认官方 DeepSeek API；访问教师 API 会产生费用，下面是人工选择运行的示例，本次整理未执行：

```bash
python -m evisearch.data.teacher --general 4 --medical 0 --attempts 1 \
  --supervised --no-thinking --out /data3/dzh/MedSearchRL/runs/release/teacher-pilot
```

`--supervised` 使用答案/证据条件化构造示范；不要把它称为教师零提示自主解题。该入口输出候选与验收轨迹，不会自动把新轨迹混入当前冻结 SFT。

## 6. SFT 与同起点 RL

真实运行前确认冻结输入齐全、GPU 空间允许。下面的命令按顺序执行，不加后台 `&`：

```bash
python -m evisearch.sft --gpu 0 --cap-gib 8 --probe
python -m evisearch.sft --gpu 0 --cap-gib 8

python -m evisearch.train grpo --gpu 0 --cap-gib 8 --probe
python -m evisearch.train grpo --gpu 0 --cap-gib 8
python -m evisearch.train igpo --gpu 0 --cap-gib 8
python -m evisearch.train vcr --gpu 0 --cap-gib 8
```

SFT 默认读取冻结的相关性推理数据，写 `runs/release/sft`。三个 RL 入口都从它的 `final/` 开始，并冻结相同 reference。probe 的目录与正式运行分开，probe 结果不是正式实验。

如果使用新生成的增强数据，需要显式指定：

```bash
python -m evisearch.sft --data /data3/dzh/MedSearchRL/data/release-prepared/sft_err.jsonl --gpu 0 --cap-gib 8
```

这与默认命令是**替代选择**，不要在已存在的同名 SFT 运行上混用数据。已有 run manifest 会拒绝不一致的恢复。SFT 示例仅训练主线 2B 模型；更大模型不是此发布版复现主线的必要步骤。

## 7. 恢复与新旧运行隔离

当前产物统一写到 `runs/release`。旧研究 checkpoint 留在原目录；代码位置和 source hash 已变化，不能将它们当作新入口的同一运行续上。

恢复时使用**当前运行目录内实际存在且完整**的 checkpoint 路径：

```text
python -m evisearch.sft --resume /data3/dzh/MedSearchRL/runs/release/sft/checkpoint-N --gpu 0
python -m evisearch.train grpo --resume /data3/dzh/MedSearchRL/runs/release/grpo-42/checkpoint-N --gpu 0
```

上面 `N` 是路径说明，不是可以原样执行的 shell 命令。RL 的 N 必须为偶数且处于完整两次更新边界。恢复需要 optimizer/scheduler/RNG/trainer state，只有 adapter 权重不够。SFT 校验数据、模型、源码和 checkpoint 归属，并区分本次暂停与旧暂停标记。

SFT 当前仍保留一个审计限制：恢复日志可能含重复 step，`summary.seconds` 只表示最后训练段；不要拿它当作跨段完整训练成本。RL 则记录各训练段并在恢复前备份、截断 checkpoint 之后的日志。

## 8. 评测新运行

每组都使用同一命令、相同模式和分母。先在 dev2 观察，再在已声明最终 checkpoint 上执行 test；不要根据 test 挑 checkpoint：

```bash
for model in base sft grpo-42 igpo-42 vcr-42; do
  for split in dev2 test; do
    python -m evisearch.evaluate --model "$model" --split "$split" --modes agent --gpu 0
  done
done
python -m evisearch.evaluation.compare vcr-42 grpo-42 igpo-42
```

Base 与 SFT 的静态材料对照应在第一次评测时共同声明 `--modes agent static`。已存在的 config 不允许变更模式；不要直接覆盖它来补跑不同协议。

公开表格来自整理前的已完成实验，`report.py` 用明确的 RECORDED 映射读取它们；上面的命令写入新目录。新的结果只有在实际训练、评测完成后才可与旧摘要并列，不能复用旧 summary 冒充完成。

## 9. 发布内容与验证边界

发布仓库只包括当前源码、固定配方、教程、演化故事和当前证据摘要。不带模型、语料、个人简历草稿、历史方案、旧结果报告或旧 Git 提交。

这次整理的验证结果与日期见 [validation.md](validation.md)。训练配方、工具与评分主体保留现有实现；仍需完整的新 GPU 运行，才能确认整理后的端到端训练复现。
