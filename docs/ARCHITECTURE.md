# 架构与模块职责

## 运行时模块

| 模块 | 负责内容 | 禁止事项 |
|---|---|---|
| `detectors` | 保存 action chunk 级状态历史并计算异常分数 | 向任意机械臂发送命令 |
| `inputs` | 接收终端键盘和外接按钮事件 | 决定控制权归属 |
| `handoff` | 管理状态切换并保证“单一动作写入者”规则 | 推理策略动作 |
| `hardware` | 管理串口总线、主臂对齐和力矩状态读回 | 标注任务结果 |
| `journal` | 持久化保存 experience 记录 | 将数据加入 SFT 清单 |
| `policy` | 请求 LingBot HTTP 推理服务 | 直接写电机命令 |
| `runtime` | 按控制频率调度各模块 | 绕过 `handoff` 的控制权判断 |
| `multi_policy` | 将精确任务文本路由到对应专项 teacher | 猜测任务或进行模糊匹配 |
| `distillation` | 离线为已保存状态生成 teacher 动作标签 | 进入真机控制循环 |
| `lerobot_export` | 生成带 teacher 动作的派生训练数据集 | 修改原始 RECAP experience |
| `orchestrator` | 可恢复地串联校验、标注、导出和外部训练命令 | 直接控制机械臂或绕过 teacher 身份校验 |

## 状态切换

```text
AUTO（自动策略控制）
  | 检测器报警：保持 follower，等待操作员
  | 手动按 Space：明确请求接管
  v
TAKEOVER_PENDING（等待接管） --R--> AUTO
  | 按键 1 / Space
  v
ALIGNING_LEADER（主臂对齐中）
  | 对齐误差连续满足阈值
  v
LEADER_ALIGNED（主臂已对齐） --R--> AUTO
  | 按键 2 / Space
  | 六个电机的 Torque_Enable 均读回 0
  v
HUMAN（人工控制） --Space--> AUTO

任何对齐、状态读回或控制异常 -> FAULT（故障） -> 保留 partial 数据并停止。
```

接管过程中，follower 始终保持位置控制。主臂对齐和主臂卸力是两个相互独立、均需操作员
授权的步骤。只有六个 leader 电机的力矩状态全部确认卸载后，系统才允许进入人工控制。

## 训练边界

采集器保存所有策略动作、人工介入和任务结果。后续完整 RECAP 训练流水线将独立完成：

1. 校验已经完整结束的 episode；
2. 根据任务结果和子任务事件生成稀疏 return；
3. 训练 value model（价值模型）；
4. 计算每个时间步的 advantage（优势值）；
5. 构造带 advantage 条件的 LingBot 训练样本。

本仓库的采集运行时不会修改已有 LingBot SFT 数据清单。

Multi-policy（多策略）蒸馏是一条独立的离线流水线：把 student 访问过并保存下来的状态按
任务路由给冻结的 teacher；将 teacher 标签写入可断点恢复的旁路文件；按照连续的策略控制
区间切分数据；最后导出派生的 LeRobot 数据集。随后使用常规 LingBot 训练器，将该数据集与
明确指定的 replay（回放）数据混合训练。
