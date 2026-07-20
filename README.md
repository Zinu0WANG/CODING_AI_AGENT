# Observable Coding Agent Harness

一个面向 AI Agent 工程实践与面试演示的 Coding Agent。它不只展示模型答案，还展示 Agent 如何理解仓库、选择工具、请求审批、修改代码、运行质量门禁，以及如何回放完整执行轨迹。

> 安全说明：命令策略和审批属于应用层安全边界，不是 Docker 或操作系统级沙箱。危险命令默认拒绝，但不要在包含高价值凭据的主机上运行不可信任务。

## 核心能力

- Anthropic Tool Use 驱动的 ReAct 循环，兼容 Anthropic-compatible Provider。
- 轻量 RepoMap：文件树、语言、关键配置、Python 顶层符号、Git 状态和增量缓存。
- 三级工具策略：只读自动允许、写操作按配置审批、删除/安装/联网等危险动作默认拒绝。
- 每次任务生成独立 Run，记录模型响应、工具调用、审批、上下文选择、验证和最终状态。
- 修改后自动执行 lint/test；失败结果反馈给 Agent，最多自动修复两轮。
- Agent 修改前后统一 Diff，不自动 commit、push 或覆盖 Git 历史。
- 临时子 Agent、具名长期队友、原子任务领取、并发安全 JSONL 信箱和后台命令。
- Rich 终端界面以及只读历史回放。

## 安装与启动

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env，填写 ANTHROPIC_API_KEY 与 MODEL_ID
python agent.py
```

也可以配置兼容 Anthropic Messages API 的服务：

```dotenv
ANTHROPIC_API_KEY=...
ANTHROPIC_BASE_URL=https://provider.example/anthropic
MODEL_ID=provider-model-id
```

阿里云百炼千问示例：

```dotenv
DASHSCOPE_API_KEY=你的新密钥
MODEL_ID=qwen3.7-plus
ANTHROPIC_BASE_URL=https://dashscope.aliyuncs.com/apps/anthropic
```

如果使用百炼子工作空间专属域名，将 Base URL 改为
`https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/apps/anthropic`。Base URL 不要追加 `/v1`。

## 项目配置

在工作区根目录使用 `.agent.yml`：

```yaml
test_commands:
  - python -m pytest -q
lint_commands: []
ignore_patterns:
  - .venv/**
  - build/**
approval_policy: ask_on_write  # ask_on_write | allow_write | read_only
max_steps: 40
max_fix_attempts: 2
command_timeout: 120
context_keep_tool_batches: 3
artifact_threshold_tokens: 1000
artifact_read_default_chars: 8000
artifact_search_max_hits: 5
context_window_tokens: 128000
context_compaction_trigger_ratio: 0.70
context_compaction_target_tokens: 25000
context_summary_max_tokens: 12000
context_summary_retry_count: 1
context_message_trim_trigger: 60
context_message_keep_head: 3
context_message_keep_tail: 47
team_auto_receive: true
team_message_batch_size: 20
team_message_token_limit: 4000
team_delivery_timeout_seconds: 60
team_session_recent_messages: 12
team_session_summary_tokens: 2000
team_require_write_scope: true
model_max_output_tokens: 3000
no_progress_replan_after: 3
```

为减少模型往返，Agent 会优先使用 `read_files` 批量读取候选文件，并使用 `batch_edit`
在全部替换预检通过后批量修改。未发生变化的文件会返回缓存引用；连续 3 轮完全重复工具调用时，
Runtime 会要求模型停止重复并重新规划。

配置文件中的 lint/test 命令被视为仓库所有者提供的可信命令。模型临时生成的 Shell 命令仍经过策略判断。

## CLI

| 命令 | 作用 |
|---|---|
| `/runs` | 查看历史 Run、状态、工具次数、Token 和耗时 |
| `/plan <需求>` | 只读分析仓库并保存一份可执行计划 |
| `/plans` | 查看持久化计划及状态 |
| `/show-plan <id>` | 查看计划全文 |
| `/implement <id>` | 校验仓库未变化后执行指定计划 |
| `/inspect <run_id>` | 查看结构化执行时间线 |
| `/replay <run_id>` | 只读回放，不调用模型、不执行工具 |
| `/diff` | 查看最近一次任务产生的修改 |
| `/test` | 手动运行配置的质量门禁 |
| `/abort` | 阻止当前 Runtime 发起新的工具调用并保留轨迹 |
| `/exit` | 退出 |

计划模式使用应用层只读工具边界。典型流程：

```text
/plan 为用户模块增加输入校验和测试
/plans
/show-plan a1b2c3d4
/implement a1b2c3d4
```

计划保存在 `.plans/<plan_id>.json`。如果生成计划后 Git HEAD、文件列表或文件内容发生变化，
`/implement` 会将计划标记为 `stale` 并拒绝执行，避免旧计划修改已经变化的代码。

审批提示支持：`y` 仅允许本次、`a` 允许本 Run 后续普通写操作、回车或 `n` 拒绝。危险动作始终默认拒绝。

## 面试演示脚本

准备一个带测试的小型仓库，然后输入：

```text
实现一个带输入校验的 Todo 创建功能，并补充测试。先分析仓库结构，再修改代码，最后运行质量门禁。
```

建议依次展示：

1. RepoMap 如何帮助模型定位文件。
2. 第一次写文件时出现的审批卡片。
3. Agent 修改和自动测试；可故意准备一个失败测试展示修复循环。
4. 最终 Diff、验证结果和 Run ID。
5. 使用 `/inspect <run_id>` 展示工具与 Token 指标。
6. 使用 `/replay <run_id>` 证明轨迹可以安全复盘。

## 架构

```text
CLI
 └─ AgentRuntime
     ├─ RepoMap / context selection
     ├─ Model client
     ├─ ToolRegistry ─ ToolPolicy ─ approval
     ├─ quality gates / diff
     ├─ EventStore (.runs/<run_id>/events.jsonl)
     └─ TeammateManager ─ MessageBus / TaskManager
```

- `runtime` 只负责模型循环、修复循环和生命周期。
- `tools` 负责工具执行、变更快照和质量门禁。
- `policy` 是模型输出与物理执行之间的应用层信任边界。
- `events` 提供 append-only 轨迹与派生指标。
- `context` 负责低成本仓库理解。
- `context_management` 将三批之前的大型工具结果外置为当前 Run 的 artifact，并提供关键词搜索和分页取回。
- `team` 与 `state` 负责协作和并发一致性。

## 上下文与 Artifacts

当前 Run 最近三次工具调用批次保留完整结果。更早批次中超过约 1000 Token 的结果会写入：

```text
.runs/<run_id>/artifacts/
├── index.jsonl
└── <artifact_id>.txt
```

原 `tool_result` 仍保留 `tool_use_id`，但正文会替换成短引用。模型可调用：

- `artifact_search(query)`：在当前 Run 的外置结果中按关键词查找。
- `artifact_read(artifact_id, offset, limit)`：按字符分页取回，单页最多 12000 字符。

Artifact 只在当前 Run 内检索，不会自动进入下一次任务；内容为本地明文，并随 `.runs/` 一起被 Git 忽略。

## 测试

```powershell
python -m pytest -q
python -m compileall -q agent.py coding_agent
```

测试覆盖配置、事件损坏恢复、路径穿越、命令风险、RepoMap、并发信箱、原子任务认领、Fake Model 端到端运行、危险命令拒绝、审批拒绝、质量门禁重试、只读回放，以及大型工具结果的外置、搜索和分页取回。

## 运行数据

以下目录是本地运行状态，默认被 Git 忽略：

- `.runs/`：轨迹、RepoMap 缓存。
- `.tasks/`：持久化任务。
- `.team/`：团队配置与信箱。
- `.team/team.db`：可靠消息队列；消息经过 `pending → delivered → acknowledged`，不再读取即删除。
- `.team/sessions/`：持久队友的精简会话检查点。
- `/team` 查看队友、任务与写入范围；`/messages` 查看消息；`/retry-message ID` 重新投递未确认消息。
- 主 Agent 每轮模型调用前自动接收队友更新，模型调用成功后才 ACK；写入型子 Agent 受 `write_scope` 限制。
- `.transcripts/`：旧版本兼容数据，可手动清理。
