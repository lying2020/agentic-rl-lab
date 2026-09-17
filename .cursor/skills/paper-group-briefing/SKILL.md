---
name: paper-group-briefing
description: >-
  组会论文讲解：按六问写可预览 Markdown（KaTeX 美元公式 + Mermaid 方法流程图）。
  Use when the user asks to 讲解/精读/组会汇报/分析论文, wants BRIEFING.md or
  组会汇报_*.md, or needs paper methods drawn as mermaid that renders in
  Cursor preview. Do not leave formulas as raw LaTeX environments in .md.
---

# 组会论文讲解（Preview-first Markdown）

角色：人工智能专业本科生的博士科研助理。默认中文。专有名词可保留英文。
证据：论文 LaTeX > PDF > 仓库。不确定标 `[假设]` / `[推断]` / `[估计]`。

先读 [math-and-mermaid.md](math-and-mermaid.md)（公式与图能否在 Cursor Preview 里变成排版/图例，完全看这一份）。
六问展开见 [briefing-outline.md](briefing-outline.md)。正误样例见 [examples.md](examples.md)。

## 用户提示词（逐字保留，不可改写、不可删减）

我是人工智能专业的本科生, 你是我的博士科研助理,你有很高的水平,现在需要在组会上对论文进行分析总结汇报给我,我需要你汇报以下内容: 1 本paper的任务是什么?是在解决一个什么问题?motivation是什么?现有的方法有哪些局限性?用一个例子解释? 2 针对这个motivation,这个论文是怎么想到用什么什么算法工具来解决这个问题的呢? （针对这里，请你分层次来讲解，比如整个模型或者算法是三个模块，他们之间的关系和交互时怎么样的人，互相会实现什么功能，然后再说每个模块下面各自的实现的算法和小的模块以及各个的层级），整个算法流程像一个tree一样展开呈现在我的面前 3 本文的算法流程是什么样子的?尤其是算法的细节是什么呢?请用一个例子搭配着公式或者是算法流程帮我细致的过一遍 4 baseline和数据集以及训练成本是什么?5 做了哪些实验？这些实验分别说明了什么内容？实验表格中的每一项评价标准是什么？怎样支撑你的观点的呢？6 如果我们要复现论文中的方法，在数据集，模型，硬件，代码和算法等方面，需要做哪些准备？

上面六问是内容合同。下面是执行合同：必须让 Markdown Preview 里公式变成排版、方法变成 Mermaid 图，而不是源码。

## 写进文档的章节（与六问一一对应）

| # | 提示词原问 | 文档标题 |
|---|---|---|
| 0 | （补充） | 一句话贡献 |
| 1 | 任务 / 问题 / motivation / 局限 / 例子 | 任务定义与动机剖析 |
| 2 | 为何用这些算法；模块关系；tree 展开 | 方法论树状拆解 **+ 强制 Mermaid** |
| 3 | 流程细节；例子配公式走一遍 | 算法流程与公式细节推演 |
| 4 | baseline、数据集、训练成本 | 实验设定与资源开销 |
| 5 | 实验、每列指标、如何支撑观点 | 实验结论的因果支撑 |
| 6 | 复现准备 | 复现前置准备清单 |
| 7 | （补充） | Paper 与代码缺口 / 组会追问 |

第 2 问的「tree」用两种同时给：**ASCII 文本树**（方便检索）+ **` ```mermaid ` 流程图**（Preview 出图）。只给 `text` 树、不给 Mermaid，视为未完成。

## 硬约束（Preview）

1. 组会交付物是 **Markdown**。公式写在 `.md` 里，用 `$...$` / `$$...$$`，让 Cursor Preview / KaTeX 渲染。禁止用 `\begin{equation}`、`\[...\]`、未包美元的 `\frac` 充当正文公式。
2. 方法必须有至少一张 **Mermaid**（`flowchart` 模块关系；推荐再加 `sequenceDiagram` 训练/推理一步）。图要在 Preview 里出图，不要 PlantUML，不要把 `$` 公式塞进节点。
3. 不要把「公式清晰」推给 `BRIEFING.tex` 而在 `.md` 里留源码。用户要看的是 Preview 后的 md。除非用户点名要 TeX PDF，否则不强制 `.tex`。
4. 落盘：用户指定目录优先；否则 `paper_assist/<slug>/BRIEFING.md`。对照多篇可另写 `docs/组会汇报_*.md`。
5. 写完对每个新 md 跑：`python3 ~/.cursor/skills/paper-group-briefing/scripts/check_briefing_md.py <file.md>`，不过就改到过。

## 工作流

```
Task Progress:
- [ ] 读 PDF / TeX / 仓（缺仓则标无官方代码）
- [ ] 按 briefing-outline.md 写 0–7 节
- [ ] 第 2 节：ASCII 树 + Mermaid（节点用纯文本，公式写在图外 $$ 块）
- [ ] 第 3 节：关键式用 $$；手算小例子
- [ ] python scripts/check_briefing_md.py
- [ ] 对话里先给一句话贡献和文件路径，细节指向 md
```

## 和 paper-read-reproduce 的分工

- 本 skill：组会讲解、Preview 公式与图。
- `paper-read-reproduce`：要跑通推理/训练、`conda_env_setup`、`CODE_PAPER_MAP` 时再用。两者可叠，但组会 md 仍遵守本 skill 的 Preview 规则。
