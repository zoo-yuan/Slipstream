# Local-Markdown Tracker — Wayfinding Operations（本仓约定）

> 本仓未接入外部 issue tracker（想用 GitHub Issues 可运行 `/setup-matt-pocock-skills`），按 wayfinder 默认走 local-markdown tracker。此文件定义本仓的 wayfinding 操作口径，后续会话照此执行。

## 实体与位置

- **地图（map）**：`.wayfinder/map.md`，等价于 label `wayfinder:map` 的 issue。地图是**索引**不是存储：决策细节只住在票里。
- **票（ticket）**：`.wayfinder/tickets/<NN>-<slug>.md`，每张票是地图的子 issue，文件即身份。frontmatter：

```yaml
---
id: NN
title: 票名（叙述中一律用名引用，不裸用编号）
labels: [wayfinder:grilling]   # research | prototype | grilling | task
status: open                    # open | closed
assignee: null                  # 认领人 = claim；open 且 assignee:null = 未认领
blocked-by: []                  # 阻塞票的 id 列表；全部 closed 才解锁
---
## Question
<这张票要解决的决策/调研问题，一票一个 100K token 会话的量>
```

- **裁决记录**：答案不写在正文里，解决时在票文件末尾追加 `## Resolution` 小节，`status` 改 `closed`，并在地图 Decisions so far 加一行指针。
- **调研产物**：`.wayfinder/research/<slug>.md`，由子代理写入，票的 Resolution 链接它。（wayfinder 原文要求 throwaway 分支；本仓 git 为 unborn master，降级为 research/ 目录，属已批准偏差。）

## 查询

```bash
# frontier（可认领的票）：open + 未认领 + blocked-by 全部 closed
python3 - <<'EOF'
import re, pathlib
d = pathlib.Path('.wayfinder/tickets')
st = {p.name.split('-')[0]: re.search(r'status: (\w+)', p.read_text()).group(1)
      for p in d.glob('*.md')}
for p in sorted(d.glob('*.md')):
    t = p.read_text()
    if re.search(r'status: closed', t) or re.search(r'assignee: \S+', t):
        continue
    blockers = re.findall(r'\d+', re.search(r'blocked-by: \[(.*?)\]', t, re.S).group(1))
    if all(st.get(b) == 'closed' for b in blockers):
        print(p.name)
EOF
```

## 并发

多会话可能同时编辑 tracker：认领先改 `assignee` 再干活；只写自己的文件。

## 技能映射（wayfinder 引用的子技能在本机不存在，等价实现）

- grilling / domain-modeling → `ask_user_question` 结构化提问（可参考 `~/.pi/agent/skills/grill-me`、`grill-with-docs` 的追问法）；**HITL 票不得由代理代答用户立场**。
- research → AFK 子代理（subagent 工具），读完票后独立调研，产物写 `.wayfinder/research/`。
- prototype → 子代理或本会话做廉价原型，产物链接进票。
