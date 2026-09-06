# RECAP 阶段实验台账（2026-09-06）

本文把真实录制视频、RECAP episode、训练权重和验证结果串在一起。以下目录均位于 H20 工作区
`/mnt/workspace/users/zb/lingbot_ws`。阶段目录禁止复用、覆盖或自动清理。

## 公共冻结组件

- LingBot teacher：`checkpoints/newdata_yellow_beside_milk_teacher_cumulative_epoch90_merged_20260902`
- 归一化统计：`norm_stats/all_data_train776_selectorfix_20260829.json`
- 视觉缓存：`experiments/lingbot_recap_visual_cache_all776_16000_3h20_20260906`
- 视觉 bottleneck：`experiments/lingbot_recap_visual_bottleneck_512d_3h20_20260906`
- 原始 RECAP episode（操作机）：`/home/mzm/lerobot_data/recap_straw_into_cup`
- 原始 RECAP episode（H20 镜像）：`data/recap_straw_into_cup`

## 阶段权重

### Bootstrap / v1

- 权重：`experiments/lingbot_recap_straw_cup_residual_20260906`
- 日志：`logs/recap_bootstrap_residual_20260906.log`
- 用途：第一轮 actor/critic bootstrap 与安全门槛验证。

### v2

- 权重：`experiments/lingbot_recap_straw_cup_intervention_residual_v2_20260906`
- 日志：`logs/recap_intervention_residual_v2_20260906.log`
- 数据规模：最初 26 条 sealed episode；固定 held-out 为 `142957`、`143028`、`143103`。
- 策略：手臂与夹爪统一 residual limit 0.20，部署统一 scale 0.25。
- 真机结论：能靠近目标，但夹爪有效修正范围过小。

### v3-gripper

- 权重：`experiments/lingbot_recap_straw_cup_intervention_residual_v3_gripper_20260906`
- 日志：`logs/recap_intervention_residual_v3_gripper_20260906.log`
- 策略：手臂 residual limit 0.20、部署 scale 0.25；夹爪 residual limit 0.60、部署 scale 1.0。
- held-out 最佳：grasp epoch 125，place epoch 168。
- 真机结论：夹爪实际从约 26 闭到 1.95，但一次 rollout 在错误位置闭合，暴露目标定位覆盖不足。
- 对应 rollout：操作机 `recap_straw_into_cup_online_v3/episode_20260906_212016_6046028a.partial`。

### v4（新增人工纠正）

- 权重：`experiments/lingbot_recap_straw_cup_intervention_residual_v4_20260906`
- 日志：`logs/recap_intervention_residual_v4_20260906.log`
- 数据：61 条 sealed episode，其中 59 条 success、2 条 failure。
- 监督训练：只使用 success；53 条训练，6 条 held-out（旧场景 3 条、新场景 3 条）。
- failure 保留给后续带奖励的 RL，不作为监督动作目标。
- 新增 35 条中的失败 episode：`episode_20260906_213817_6c75aee7.complete`。

### v5（接管前纠正锚点）

- 权重：`experiments/lingbot_recap_straw_cup_intervention_residual_v5_20260906`
- 日志：`logs/recap_intervention_residual_v5_20260906.log`
- 数据：99 条 sealed episode，其中 95 条 success、4 条 failure。
- 监督训练：85 条 success 训练，10 条 success held-out；4 条 failure 继续隔离保存。
- 策略修正：每条 episode 最后 4 个接管前决策点学习紧随其后的人工纠正 chunk；
  更早的 zero-residual 上下文只占采样总质量 10%，人工帧与纠正锚点占 90%。
- 学习率保持 `3e-4`，避免把监督标签问题误判成优化步长不足。
- 自动部署门禁：grasp/place 必须全部 finite，且 held-out action MAE 均优于冻结 teacher。

## 保留规则

1. 每一阶段必须使用新的实验目录；不得覆盖上一阶段的 `best.pt`、`latest.pt` 或报告。
2. 保留训练日志、完整配置、held-out episode ID、训练报告和真机 rollout ID。
3. 人工视频按录制时间与 episode 时间戳对应，不复制进 checkpoint 目录。
4. failure episode 不删除，但必须与监督训练集隔离。
5. 用户未明确授权前，不删除任何阶段权重。
