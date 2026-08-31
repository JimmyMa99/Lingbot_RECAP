# Multi-policy on-policy 蒸馏

本分支实现把多个 LingBot 专项策略蒸馏到一个通用策略所需的数据、身份校验、导出和迭代
调度基础设施。它复用 LingBot 的连续动作 flow-matching 头，不修改真机控制协议。

## 方法边界

当前版本采用 student 实际访问的状态分布，但监督目标是 teacher 连续动作：

1. student 在真机执行 rollout，RECAP 保存每个动作前的状态和双相机观测；
2. 根据精确任务文本把状态路由给对应的冻结 teacher；
3. teacher 对每个 student 状态输出 16 步 action chunk；
4. 保存完整 chunk，并以第一步动作作为当前帧的蒸馏标签；
5. 将连续标签导出为 LeRobot v3 数据集；
6. 使用 LingBot `L1_fm` 训练器，将蒸馏数据与清洗后的 replay 数据混合。

这是一种 on-policy DAgger/RLDG 风格的连续动作 MOPD 近似。LingBot 尚未提供稳定的连续动作
概率或 flow-density 评分接口，因此当前实现不是 token-level reverse-KL MOPD。

## 强制安全检查

### Teacher 权重身份

每个 teacher 的 `/healthz` 必须返回以下字段之一，并且值必须与注册表的绝对 checkpoint
路径一致：

- `checkpoint`
- `checkpoint_path`
- `model_path`
- `adapter_path`
- `policy_checkpoint`

只有 `model_loaded=true` 不够。若服务没有公开权重身份或路径不一致，所有标注命令都会
立即失败，防止把错误模型的动作写成目标 teacher 标签。

若现有 LingBot `server_http.py` 尚未返回该字段，可在 LingBot 代码目录应用仓库提供的补丁：

```bash
patch -p1 < /path/to/Lingbot_RECAP/patches/lingbot_server_checkpoint_health.patch
```

### 训练契约

注册表 v2 顶层必须包含一个共享 `training_contract`：

- normalization stats 文件及 SHA-256；
- robot config 文件及 SHA-256；
- `top/wrist` 到 LingBot 相机字段的精确映射；
- 动作空间名称；
- 六个关节的固定顺序。

计算哈希：

```bash
lingbot-mopd fingerprint \
  /absolute/path/to/norm_stats.json \
  /absolute/path/to/so_arm101.yaml
```

复制并填写注册表：

```bash
cp configs/multi_policy_teachers.example.json \
  configs/multi_policy_teachers.local.json

lingbot-mopd validate-teachers \
  --teacher-registry configs/multi_policy_teachers.local.json
```

注册表加载时会重新计算两个文件的 SHA-256。contract ID 和完整内容会写入标签元数据及
`distillation_provenance.json`；不同 contract 的 episode 禁止导出到同一个数据集。

## 离线 Teacher 标注

全量标注：

```bash
lingbot-mopd relabel \
  --teacher-registry configs/multi_policy_teachers.local.json \
  --experience-root /home/mzm/lerobot_data/recap_experience
```

标注通过逐行 flush 与 `fsync` 保存到 `teacher_labels.partial.jsonl`，进程重启后跳过已经
完成的 frame。成功结束后原子生成：

```text
teacher_labels.jsonl
teacher_labels.meta.json
```

默认只标注 `action_source=lingbot_policy`，人工接管帧不会混入蒸馏动作。

### Preview 冒烟测试

```bash
lingbot-mopd relabel \
  --teacher-registry configs/multi_policy_teachers.local.json \
  --episode /path/to/episode.complete \
  --max-frames 8
```

`--max-frames` 永远生成独立的：

```text
teacher_labels.preview.jsonl
teacher_labels.preview.meta.json
```

它不会生成或占用正式标签文件，因此之后直接运行全量标注即可，不需要删除 preview，也不
需要 `--overwrite`。

## 导出为 LeRobot

```bash
lingbot-mopd export-distill \
  --experience-root /home/mzm/lerobot_data/recap_experience \
  --output-root /home/mzm/lerobot_data/mopd_teacher_labeled_v1 \
  --repo-id mzm/lingbot_mopd_teacher_labeled_v1
```

导出器会拒绝：

- preview 标签；
- 未验证 teacher checkpoint 身份的标签；
- contract 不一致的数据；
- NaN/Inf 或错误维度的 state/action；
- teacher 身份在同一连续 segment 内发生变化；
- 没有达到最短连续帧数的数据；
- 覆盖已存在的输出目录。

人工介入造成的时间缺口会切成不同 episode，未来 action chunk 不会跨越控制权切换。

## 可恢复迭代调度器

`run-iteration` 串联“teacher 检查 → 全量标注 → 导出 → replay 清单 → 训练命令 →
后处理命令”，并将每个阶段写入 `iteration_state.json`：

```bash
lingbot-mopd run-iteration \
  --iteration 1 \
  --teacher-registry configs/multi_policy_teachers.local.json \
  --experience-root /home/mzm/lerobot_data/recap_experience \
  --run-root /export4/mzm/output/lingbot_mopd \
  --repo-id mzm/lingbot_mopd \
  --replay-manifest /path/to/clean_replay.txt \
  --replay-repeat 2 \
  --train-command '/path/to/train_mopd_iteration.sh' \
  --post-train-command '/path/to/merge_and_validate.sh'
```

训练命令会收到：

```text
MOPD_ITERATION
MOPD_ITERATION_DIR
MOPD_TRAIN_MANIFEST
MOPD_DATASET_ROOT
MOPD_TEACHER_REGISTRY
```

外部命令不通过 shell 执行。中断后使用相同参数加 `--resume`；已经成功的训练或后处理阶段
不会重复运行。锁文件防止同一 iteration 被两个进程同时执行。

调度器不会自动进行真机 rollout，也不会绕过人工安全确认。student rollout 完成并保存后，
它只负责离线数据与训练阶段。

## 第一次正式实验

1. 三个 teacher 都从服务端公开并通过 checkpoint 身份校验。
2. 三个任务各采集一批 student rollout。
3. teacher 保持冻结，student 从三者共同祖先 checkpoint 初始化。
4. 蒸馏样本与 replay 从 1:1 或 1:2 起步。
5. 使用独立 held-out rollout 与真机任务成功率比较：
   - 混合 SFT；
   - 参数平均；
   - continuous-action MOPD 加 replay。
6. eval 不得复用训练的同一批 8 帧。

在实现 teacher/student 连续动作评分接口和 flow 分布 divergence 前，实验报告必须继续标注
为“continuous-action MOPD approximation”，不能宣称是严格 reverse-KL MOPD。
