# LingBot RECAP 在线强化学习设计

## 结论

4090 上 OpenPI 项目中的在线学习逻辑可以迁移到 LingBot，且不需要 JAX。迁移后冻结
LingBot-VLA，只使用 PyTorch 训练很小的 residual actor 与 twin critic。actor 不重新生成
完整动作，而是在 LingBot 原始 action chunk 周围做有界修正；初始化时修正严格为零，因此
未训练的 RL 层与原 LingBot 行为一致。

这套实现属于 RECAP 风格的人工介入在线强化学习基础设施，但不是对某篇 RECAP 论文逐行复刻：

- 保存“自动失败 → 人工接管 → 人工纠正 → 成功/失败”的完整轨迹；
- 使用人工结果构造稀疏 terminal reward；
- 失败和人工纠正数据都进入 replay，不混入普通 SFT 数据；
- 冻结 VLA，用轻量 actor/critic 在线更新；
- checkpoint、replay 和 optimizer 都可恢复；
- 真机 online 前必须先 warmup、bootstrap，再由操作员显式 activate。

## 从 OpenPI 迁移与不迁移的部分

| 部分 | 处理方式 |
|---|---|
| 人工接管、封存原始轨迹、成功/失败标记 | 直接复用 `Lingbot_RECAP` 现有采集层 |
| residual chunk actor、twin-Q、target network、UTD、replay | 已迁为纯 PyTorch `online_rl.py` |
| 幂等 episode commit、checkpoint 恢复、warmup/bootstrap/activate 门槛 | 已迁为 `rl_learner.py` |
| OpenPI/JAX policy、OpenPI token、OpenPI 数据变换 | 不迁移 |
| LingBot action chunk | 作为 frozen reference action |
| LingBot 视觉表征 | 下一阶段由 LingBot PyTorch 推理旁路导出 |

## LingBot 表征接口

不能把 OpenPI 的 token encoder 原封不动套到 LingBot。LingBot 的
`QwenvlWithExpertV2Model.forward()` 已经产生最终层 prefix hidden states；推理时应只取
`visual_pos_masks` 对应的视觉 token，避免读取未来动作或 action expert 输出。建议：

1. 在冻结 checkpoint 上缓存训练/验证图像的最终层视觉 token；
2. 训练一个带严格验证和 zero-token ablation 的 512D bottleneck；
3. 推理服务同时返回 `reference_actions` 和冻结的 512D `visual_feature`；
4. `visual_feature + 6D 当前关节 + 6D 速度` 组成 critic/actor 状态；
5. actor 只修改当前要执行的 16-step chunk，执行后重新观察和规划。

若 bottleneck 的 held-out 重构和 zero-token 检查不通过，不得进入真机 online。不要使用
训练时的未来视频/未来深度标签作为在线状态，因为真机执行时拿不到未来观测。

## 推荐的第一轮流程

第一轮只做一个任务，并拆成 `grasp`、`place` 两个 phase，各自维护独立 replay 和 learner。

1. held-shadow：只计算 LingBot reference 与 RL 输出，不把 RL 动作发送给机械臂；确认未训练
   actor 与 reference 逐元素一致。
2. warmup：建议至少 20 条完整介入轨迹、至少 8 条最终成功，并保证每个 phase 的 replay
   transition 数达到门槛。
3. bootstrap：对已封存 replay 做离线更新，检查所有 loss、Q 值和梯度为 finite。
4. activate：必须显式执行，不能因 warmup 达标自动启用。
5. online：一次 rollout 完整结束、机械臂安全下电、数据封存后再更新；不能在单条轨迹中途
   换 actor。

首轮参数沿用经过实测的保守值：chunk 16、residual limit 0.05、reference dropout 0.5、
policy delay 2、UTD 5。探索噪声只能在完成 held-shadow 和人工急停演练后开启。

## 当前代码边界

当前分支已经完成可独立测试的 PyTorch learner 核心，但尚未把 RL 动作接入真机。仍需完成：

- LingBot 推理侧视觉 token 导出及 512D bottleneck 训练；
- 将 `.complete` experience 编码成 chunk replay；
- phase 标注和 terminal window 过滤；
- held-shadow 全链路和真实延迟测试；
- 管理命令、后台上传队列以及显式 bootstrap/activate 命令。

在这些门槛完成前，现有 `lingbot-recap collect` 仍只采数据，不会在线更新或改变机器人动作。
