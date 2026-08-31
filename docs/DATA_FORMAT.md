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
