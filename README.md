# Lingbot_RECAP

用于 LingBot-VLA + SO-101 的安全人工介入（human intervention）采集系统。

它从 rollout 开始就连续保存自动策略轨迹。当策略来回振荡、长时间没有进展，或操作员按下空格键时，系统暂停 follower；经操作员确认后，leader 缓慢对齐 follower。只有六个 leader 电机全部读回 `Torque_Enable=0`，系统才会播报“可以人工接管”。

## 这版做什么

- A-B-A-B 关节振荡检测；
- 无进展检测；
- 空格键接管/交还，USB 脚踏映射为空格即可直接使用；
- 单一控制者状态机，禁止 LingBot 和人工同时向 follower 写动作；
- leader 自动慢速对齐和逐电机卸力读回确认；
- 从自动 rollout 开始保存图像、状态、策略动作、执行动作、控制来源和事件；
- 崩溃后保留 `.partial` 数据，可审计和恢复；
- 数据带有 `DO_NOT_ADD_TO_SFT` 标记，不会自动加入现有 SFT 数据。

这不是完整的 RECAP 训练实现。完整 RECAP 还需要 outcome reward、value model、时序 advantage 和 advantage-conditioned policy。本仓库第一阶段解决安全控制与不浪费 rollout 数据的问题。

## 模块

```text
detectors.py  只检测振荡/无进展
inputs.py     空格键及未来按钮事件
handoff.py   控制权仲裁状态机
hardware.py  LeRobot 0.4.x / Feetech SO-101 适配
journal.py   追加式 experience 数据记录
policy.py    LingBot HTTP 推理客户端
runtime.py   采集循环
```

## 操作键

- `Space`：自动→人工接管；人工→交还自动
- `S`：标记成功并保存
- `F`：标记失败并保存
- `R`：检测器暂停后恢复自动
- `Q`/`Esc`：中止，但保留已采集数据

## 安装与离线测试

```bash
python -m pip install -e '.[dev,robot]'
pytest
```

操作机已有 LeRobot 0.4.2 时，可以只安装当前项目而不升级 LeRobot：

```bash
/home/mzm/miniconda3/envs/lerobot/bin/pip install -e . --no-deps
```

## 采集命令

确保 LingBot 服务健康、机械臂周围无人且没有其他程序占用串口：

```bash
lingbot-recap collect \
  --server http://116.196.82.74:8007 \
  --policy-checkpoint yellow_epoch_08_step_65528 \
  --task '夹起黄色鸭子' \
  --experience-root /home/mzm/lerobot_data/recap_experience
```

第一次真机接管前必须阅读 [docs/SAFETY.md](docs/SAFETY.md)。数据格式见 [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)。
