# Multi-policy on-policy 蒸馏

本仓库实现了把多个 LingBot 专项策略蒸馏到一个通用 LingBot 策略所需的数据与路由层。
第一版有意复用 LingBot 现有的连续动作 flow matching（流匹配）动作头，以便直接接入当前
训练代码。

## 为什么不能直接照搬 LLM MOPD

LLM MOPD 会把 student 生成的每个 token 前缀路由给对应领域的 teacher，并优化 teacher 与
student 的 token 概率分布差异。LingBot 输出的是连续的 16 步 action chunk，动作头采用
flow matching，并不提供自回归 action token 的对数概率。因此，直接套用 NeMo-RL 或 verl
这类 LLM MOPD 训练器，优化到的会是语言模型接口，而不是 SO-101 的连续动作分布。

当前可落地的第一阶段保留相同的 on-policy（由当前 student 实际访问）状态分布，但改用连续
动作监督目标：

1. student 在真机执行 rollout，RECAP 在每次 student 动作前保存完整观测；
2. rollout 结束后，根据精确任务文本把每个已保存状态路由给一个冻结的专项 teacher；
3. teacher 在每个 student 已访问状态上输出动作，并以第一步动作作为蒸馏标签；
4. 将连续的已标注状态导出为 LeRobot v3 数据集；
5. 使用常规 LingBot `L1_fm` 训练器，将蒸馏数据与 replay 示范数据混合训练。

它是对 MOPD 的一种 on-policy DAgger/RLDG 风格连续动作近似。这样既能减少只在 teacher
rollout 上训练造成的状态分布偏差，也不需要在真机控制循环中同步等待 teacher，从而降低
控制停顿风险。等动作头具备稳定的评分接口后，可以把“teacher 第一步动作”监督升级为
flow vector 或 probability-flow 分布差异。

## 安全边界与数据边界

- 重标注完全离线运行，不会向电机发送命令。
- 任务文本只做空白规范化后的精确匹配；任务未知或存在重复路由时立即报错。
- 默认只标注 `action_source` 为 `lingbot_policy` 的帧。除非显式传入
  `--include-human-states`，否则排除人工接管状态。
- 原始 RECAP experience 始终保留 `DO_NOT_ADD_TO_SFT`。派生数据集单独标记为
  `DISTILLATION_DATASET_ONLY`，并包含 `distillation_provenance.json`。
- 导出器拒绝覆盖已经存在的数据集目录。

## Teacher 注册表

复制 `configs/multi_policy_teachers.example.json` 为一份只在本机使用的配置，并为每个专项
策略填写 server 和 checkpoint。多个任务别名可以指向同一个 teacher，但同一个任务不能
同时指向两个 teacher。

```bash
lingbot-recap validate-teachers \
  --teacher-registry configs/multi_policy_teachers.local.json
```

所有 teacher 不需要同时常驻显存。`relabel` 只会初始化当前 episode 任务对应的 teacher。
显卡有限时，可以先启动一个任务的 teacher 并完成其标注，停止服务后再启动下一个 teacher
继续处理。

## 离线 Teacher 标注

按源数据 30 Hz 频率标注所有完整 student rollout：

```bash
lingbot-recap relabel \
  --teacher-registry configs/multi_policy_teachers.local.json \
  --experience-root /home/mzm/lerobot_data/recap_experience
```

快速冒烟测试时，只标注 8 个 student 帧：

```bash
lingbot-recap relabel \
  --teacher-registry configs/multi_policy_teachers.local.json \
  --episode /path/to/episode.complete \
  --max-frames 8
```

标注过程通过 `teacher_labels.partial.jsonl` 支持断点恢复。完整结束后会原子生成：

```text
teacher_labels.jsonl
teacher_labels.meta.json
```

每一行会保留完整 teacher chunk 供后续分析；第一版训练数据使用其中的 `teacher_action`
（即 chunk 的第一步动作）作为监督标签。

## 导出为 LeRobot 数据集

```bash
lingbot-recap export-distill \
  --experience-root /home/mzm/lerobot_data/recap_experience \
  --output-root /home/mzm/lerobot_data/mopd_teacher_labeled_v1 \
  --repo-id mzm/lingbot_mopd_teacher_labeled_v1
```

过滤人工接管状态后，帧序列中可能出现时间缺口。导出器会把这些缺口拆成独立 episode，避免
LingBot 在构造未来 action chunk 时跨越控制权切换边界。

## LingBot 训练建议

把导出的数据集作为普通 `multi` 训练清单的一项，并与原始、清洗后的示范数据混合。保守的
起始比例是每 1 份蒸馏样本搭配 1～2 份 replay 样本。robot config、动作/状态 normalization
以及相机映射必须与各个 teacher 保持一致。

第一次实验建议：

1. student 从所有专项策略的共同祖先 checkpoint 初始化；
2. 所有专项 teacher 保持冻结；
3. 在三个真机任务上，比较以下方法的成功率：
   - 仅使用混合 SFT；
   - 参数平均；
   - multi-policy on-policy 蒸馏加 replay。

在实现并验证 teacher/student 连续动作评分接口和 flow 分布差异之前，不应把当前结果称为
严格的 reverse-KL MOPD。当前版本是可直接接入 LingBot 连续动作训练的工程化近似方案。
