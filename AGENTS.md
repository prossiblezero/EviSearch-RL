# Project invariants

- This repository exposes one current evidence-agent implementation. Do not recreate version-number source directories or add old result reports.
- Keep code and small reports here. Models, environments, corpora, caches, logs, checkpoints and charts belong under `/data3/dzh/MedSearchRL`.
- Use `/data3/dzh/MedSearchRL/venv/bin/python`. Do not alter another project's environment or GPU processes.
- Use the admission gate and memory cap in `resources.py`; preserve at least 6 GiB free GPU memory. Run GPU experiments sequentially.
- Frozen external artifact identities live in `evisearch/artifacts.py`. New runs go to `runs/release`; never resume old research checkpoints with reorganized source hashes.
- Freeze splits before evaluation. Do not change task/corpus, decoder, token/search budgets or test membership between arms. Keep example counts and assistant-token mixture counts distinct.
- Tool observations and prompt text must have zero training-loss masks. Exercise native TRL tool execution; generated tool tags alone are insufficient.
- All current RL arms start from the exact same rationale SFT adapter with a frozen copy as reference. Preserve task order and rollout/update budgets. A KL-only adapter change is not proof of task-policy learning.
- MedHop entity co-occurrence is a proxy, not verified medical entailment. Keep that qualification when discussing it.
- CPU checks: `python check_data.py`, `python check_resume.py`, `python check_release.py`. `python report.py` recomputes recorded evidence; a complete new experiment additionally requires real training and evaluation artifacts.
- Never claim a positive result without held-out evidence. Preserve negative outcomes, malformed calls, zero-variance groups, failed gates and declared overrides. Label the repeatedly inspected test split honestly.
- SFT summaries must include the unadapted baseline and distinguish output-format/tool-control improvements from retrieval and answer quality. Compare identical retrieval modes before attributing differences to SFT.
- Documentation teaches the current implementation; `docs/evolution.md` is the single narrative of earlier attempts. Do not make readers learn prior code versions.
