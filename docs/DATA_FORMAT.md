# Experience 数据格式

每次 rollout（真机执行）从第一帧自动策略观测开始连续保存。人工接管数据不会与触发接管的
策略失败过程分开，以便后续完整还原“失败—接管—纠正—结果”的因果链。

```text
episode_YYYYmmdd_HHMMSS_<id>.partial/
  metadata.json
  DO_NOT_ADD_TO_SFT
  events.jsonl
  frames.jsonl
  images/top/00000000.jpg
  images/wrist/00000000.jpg
```

正常结束的 episode 会从 `.partial` 原子重命名为 `.complete`，并新增 `result.json`。进程
崩溃时会保留仍可读取的 `.partial` 目录。使用 `lingbot-recap audit` 标记并列出可恢复的
采集会话。

每一帧记录以下内容：

- 策略观测和两路相机图像路径；
- LingBot 建议执行的动作；
- 实际发送给 follower 的动作；
- 动作来源：`lingbot_policy` 或 `human_intervention`；
- 当前控制模式和各类时间戳。

每个事件记录 action chunk、检测器报警、接管原因、leader 对齐、已验证的 leader 力矩状态、
控制权交还、任务结果及运行时异常。这是一份 RL experience 日志，默认明确排除在现有 SFT
数据清单之外。

## Multi-policy teacher 标签

离线运行 `lingbot-mopd relabel` 后会新增两个旁路文件，不会修改原始日志：

```text
teacher_labels.jsonl
teacher_labels.meta.json
```

每条标签包含原始 `frame_index`、精确任务文本、teacher 名称/checkpoint/server、推理耗时、
第一步 `teacher_action`、完整 16 步 `teacher_chunk`，以及原 student 动作。元数据记录标签
步长、允许的动作来源、样本数量和 `MULTI_POLICY_ON_POLICY_DISTILLATION_ONLY` 使用边界。
它还记录经过服务端核验的 teacher checkpoint、训练契约 ID，以及 normalization stats、
robot config、相机映射和动作空间的 SHA-256 契约。

如果标注中途停止，系统会保留 `teacher_labels.partial.jsonl`；重新运行时会跳过已经完成的
帧，从断点继续。

使用 `--max-frames` 时只生成 `teacher_labels.preview.jsonl` 和对应 preview 元数据；preview
永远不会被导出器当成正式训练标签，也不会阻塞后续全量标注。

LeRobot 导出器会在派生数据集中写入 `DISTILLATION_DATASET_ONLY` 和
`distillation_provenance.json`。它会把不连续的标签区间拆成独立 episode，确保未来动作
查询不会跨越人工接管造成的时间缺口。
