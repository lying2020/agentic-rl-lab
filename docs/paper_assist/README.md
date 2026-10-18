# paper_assist 索引

默认目录：**`docs/paper_assist/<slug>/`**（不要写到仓库根 `paper_assist/`）。

## 写作规范：`paper-group-briefing` skill

本目录的精讲稿都按仓库内的 Cursor skill 写成，文件在 [`.cursor/skills/paper-group-briefing/`](../../.cursor/skills/paper-group-briefing/)：

| 文件 | 作用 |
|---|---|
| `SKILL.md` | 入口：用户原始六问提示词（逐字保留）、章节 0–7 对照表、Preview 硬约束、落盘位置、工作流 |
| `briefing-outline.md` | 六问逐条展开，规定每节必须写全的子项 |
| `math-and-mermaid.md` | 公式只用 `$` / `$$`，Mermaid 节点不放公式，保证 Cursor Preview 能渲染 |
| `examples.md` | 公式与方法图的正误样例 |
| `scripts/check_briefing_md.py` | 格式检查脚本，写完必须跑通 |

配套规则 [`.cursor/rules/paper-briefing-preview.mdc`](../../.cursor/rules/paper-briefing-preview.mdc)：编辑 `docs/**/*.md` 时自动提示遵循该 skill。

用法：在 Cursor 里对 agent 说“讲解 / 精读 / 组会汇报 `docs/xxx.pdf`”，会自动触发该 skill。写完后在仓库根检查：

```text
python3 .cursor/skills/paper-group-briefing/scripts/check_briefing_md.py docs/paper_assist/<slug>/BRIEFING.md
```

换电脑：skill 和 rule 随仓库同步，`git clone` 后用 Cursor 打开仓库即可直接使用，无需拷贝到 `~/.cursor/skills/`。`.gitignore` 只放行 `.cursor/skills/` 与 `.cursor/rules/`，新增 skill 正常 `git add` 即可。

## 对照总稿

- [`组会汇报_三篇OPD变体_TGOPD_GCOPD_RLCSD对照精读.md`](../组会汇报_三篇OPD变体_TGOPD_GCOPD_RLCSD对照精读.md)

| 论文 | 组会精读（Preview：公式 + Mermaid） |
|---|---|
| TGOPD | [`tgopd/BRIEFING.md`](tgopd/BRIEFING.md) |
| GC-OPD | [`gc-opd/BRIEFING.md`](gc-opd/BRIEFING.md) |
| RLCSD | [`rlcsd/BRIEFING.md`](rlcsd/BRIEFING.md) |
| FlowBalance | [`flowbalance/BRIEFING.md`](flowbalance/BRIEFING.md) |
| SPOT | [`spot/BRIEFING.md`](spot/BRIEFING.md) |
| SAGE-OPD | [`sage-opd/BRIEFING.md`](sage-opd/BRIEFING.md) |
| ALIGNOPSD | [`alignopsd/BRIEFING.md`](alignopsd/BRIEFING.md) · [`CODE_PAPER_MAP.md`](alignopsd/CODE_PAPER_MAP.md) · 代码 `algorithm/Align-OPSD/` |
| RetireOPD | [`retireopd/BRIEFING.md`](retireopd/BRIEFING.md) · [`CODE_PAPER_MAP.md`](retireopd/CODE_PAPER_MAP.md) · 代码 `algorithm/SDAR/` |
