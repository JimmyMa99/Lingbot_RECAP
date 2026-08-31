# Lingbot_RECAP

Lingbot_RECAP 是面向 LingBot-VLA 与 SO-101 的安全人工介入数据采集系统。它在策略 rollout
过程中检测来回振荡和无进展；操作员也可以主动请求接管。系统会先保持 follower，再让
leader 缓慢对齐 follower，确认对齐后才允许卸掉 leader 力矩并开始人工遥操作。

项目会完整保留“策略失败 → 接管 → 人工纠正 → 任务结果”的经历，供后续 RECAP / 强化学习
使用。数据默认带 `DO_NOT_ADD_TO_SFT` 标记，不会被误加进现有 SFT 数据清单。

> 这是研究原型，不是安全认证控制器。软件按键不能替代物理急停或电源开关。第一次真机
> 测试前必须阅读 [安全约束](docs/SAFETY.md)。

## 功能

- 自动检测 A-B-A-B 关节振荡和长时间无进展；
- 两键 USB 键盘接管，不依赖终端窗口焦点；
- 按键 1：暂停策略、保持 follower、缓慢对齐 leader；
- 按键 2：对齐完成后释放 leader 力矩并进入人工遥操作；
- 单一控制者状态机，禁止策略和人工同时向 follower 下发动作；
- 逐电机读回 `Torque_Enable`，只有六个 leader 电机全部为 0 才授予人工控制；
- 连续保存相机观测、关节状态、策略动作、实际动作、控制来源、事件和结果；
- 崩溃后保留 `.partial` 目录，可审计、恢复，不浪费 rollout；
- 终端键盘仍可作为两键设备的备用控制入口。

当前仓库实现的是安全采集层。完整 RECAP 训练仍需单独实现 outcome reward、value model、
时序 advantage 和 advantage-conditioned policy。

多策略 on-policy 蒸馏属于另一条实验路线，代码与使用文档位于 `mopd` 分支，不放在
RECAP 的 `main` 分支中。

## 硬件配置

本项目当前测试配置如下。

| 角色 | 型号与执行器 | 接口 | 说明 |
|---|---|---|---|
| follower（从臂/执行臂） | SO-ARM101 / SO-101 follower，6× Feetech STS3215 | `/dev/ttyACM1` | 接收策略或人工动作，接管期间始终保持使能 |
| leader（主臂/示教臂） | SO-ARM101 / SO-101 leader，6× Feetech STS3215 | `/dev/ttyACM0` | 接管时先主动对齐，再卸力供人操作 |
| top camera | OpenCV/V4L2，640×480 | `/dev/video2` | 顶视策略观测 |
| wrist camera | OpenCV/V4L2，640×480 | `/dev/video1` | 腕部策略观测 |
| 接管按钮 | 任意能表现为 USB HID keyboard 的两键有线键盘 | `/dev/input/by-id/*-event-kbd` | 两个按键必须产生不同 Linux key code |

这里的 leader 才是需要在接管前卸力的“主臂”。不要把按键 2 配成 follower 断电；follower
卸力后可能下坠，而且无法继续跟随人工示教。

相机用于保存策略实际看到的训练观测，不等同于外部手机拍摄的演示视频。

## 软件环境

### 已测试的操作机环境

- Linux（当前操作机为 Ubuntu）；
- Python 3.10；
- LeRobot 0.4.2；
- OpenCV 4.8 或更新版本；
- LingBot 推理服务通过 HTTP 提供 `/healthz` 和 `/infer`；
- 用户需要 `dialout` 组读取机械臂串口，需要 `input` 组读取 USB 按键。

推荐先配置设备权限，随后注销并重新登录：

```bash
sudo usermod -aG dialout,input "$USER"
```

### 全新 venv

```bash
git clone git@github.com:JimmyMa99/Lingbot_RECAP.git
cd Lingbot_RECAP

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[robot,dev]'
pytest
```

`.[robot]` 固定安装 LeRobot 0.4.2，并安装 OpenCV。建议在和 LingBot 训练环境分开的操作机
环境中运行本项目。

### 复用已有 LeRobot 环境

当前操作机已有环境时，避免让 pip 改动 LeRobot 及其依赖：

```bash
cd /home/mzm/code/Lingbot_RECAP
/home/mzm/miniconda3/envs/lerobot/bin/pip install -e . --no-deps
/home/mzm/miniconda3/envs/lerobot/bin/python -m compileall -q lingbot_recap tools
/home/mzm/miniconda3/envs/lerobot/bin/pytest
```

仓库也提供相同用途的安装脚本：

```bash
bash scripts/install_operation_machine.sh
```

## 机械臂与相机准备

1. 使用 LeRobot 完成 leader 和 follower 的标定，并确认以下标定文件存在：

   ```text
   ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/None.json
   ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/None.json
   ```

2. 固定设备名可能会在重插后变化。每次启动前检查：

   ```bash
   ls -l /dev/ttyACM* /dev/video* /dev/input/by-id/*-event-kbd
   fuser /dev/ttyACM0 /dev/ttyACM1
   ```

3. 确保没有 `lerobot_record`、旧推理脚本或其他进程同时占用两个串口。
4. 确保 follower 周围没有人或障碍物，leader 对齐时操作员必须松手。

代码使用 LeRobot 的校准后关节坐标，而不是直接复制原始编码器数值。前五个关节归一化到
`[-100, 100]`，夹爪归一化到 `[0, 100]`。

## 识别两键 USB 键盘

识别脚本完全使用 Python 标准库和 Linux evdev ABI，不需要安装 `evdev` 包。识别时先关闭
机械臂电源，因为这个步骤只需要读取键盘。

列出候选设备：

```bash
python tools/identify_two_key_keyboard.py --list
```

优先选择稳定的 `/dev/input/by-id/...-event-kbd` 路径，然后依提示分别按下两个按键：

```bash
python tools/identify_two_key_keyboard.py \
  --device /dev/input/by-id/usb-YOUR_TWO_KEY_PAD-event-kbd \
  --output configs/two_button_keyboard.local.json
```

脚本生成类似下面的本机配置：

```json
{
  "schema_version": 1,
  "device": "/dev/input/by-id/usb-YOUR_TWO_KEY_PAD-event-kbd",
  "buttons": {
    "align": {"code": 2, "name": "KEY_1"},
    "release": {"code": 3, "name": "KEY_2"}
  }
}
```

实际文件还会保存设备名称、USB vendor/product 和解析后的 event 路径。它与具体操作机绑定，
已由 `.gitignore` 排除，不应提交到公开仓库。如果出现 `Permission denied`，确认用户已加入
`input` 组并重新登录。

## 录制人工参与数据

### 1. 启动前检查

先确认 LingBot 服务已经加载目标权重：

```bash
curl -fsS http://LINGBOT_SERVER:8007/healthz
```

检查串口、相机、磁盘空间和物理急停，再启动采集：

```bash
lingbot-recap collect \
  --server http://LINGBOT_SERVER:8007 \
  --policy-checkpoint yellow_epoch_08_step_65528 \
  --task '夹起黄色鸭子' \
  --experience-root /home/mzm/lerobot_data/recap_experience \
  --follower-port /dev/ttyACM1 \
  --leader-port /dev/ttyACM0 \
  --top-camera /dev/video2 \
  --wrist-camera /dev/video1 \
  --button-config configs/two_button_keyboard.local.json
```

终端必须保持前台运行，因为 `S/F/R/Q/Space` 是备用控制键；两键设备本身直接读 evdev，
不依赖哪个窗口当前获得焦点。

### 2. 接管流程

```text
AUTO
  | 策略异常自动暂停
  v
TAKEOVER_PENDING
  | 按键 1：操作员授权 leader 对齐
  | （AUTO 中主动按按键 1 会直接完成上述两步）
  v
ALIGNING_LEADER
  | 对齐误差连续满足阈值
  v
LEADER_ALIGNED
  | 按键 2：操作员授权 leader 卸力
  | 六个 Torque_Enable 全部读回 0
  v
HUMAN
```

- 按键 1 后，策略 action chunk 会停止继续执行，follower 保持当时姿态；leader 默认用约
  4 秒缓慢移动到 follower 的校准后关节位置。
- 听到“主臂已对齐”后才能按按键 2。
- 按键 2 只在 `LEADER_ALIGNED` 状态有效；卸力读回失败会进入 `FAULT`，不会授予人工控制。
- 进入 `HUMAN` 后，操作员移动 leader，follower 实时跟随，所有人工动作都会记录。
- 人工完成任务后按 `S` 保存成功，按 `F` 保存失败。

终端备用键：

| 键 | 动作 |
|---|---|
| `Space` | 第一次请求/执行对齐，第二次确认卸力；人工模式下交还自动策略 |
| `S` | 标记成功并结束保存 |
| `F` | 标记失败并结束保存 |
| `R` | 检测器暂停或已对齐时取消接管并恢复自动策略 |
| `Q` / `Esc` | 中止运行，但保留已记录数据 |

## 数据与审计

每次 rollout 从第一帧开始连续保存：

```text
episode_YYYYmmdd_HHMMSS_<id>.partial/
  metadata.json
  DO_NOT_ADD_TO_SFT
  events.jsonl
  frames.jsonl
  images/top/00000000.jpg
  images/wrist/00000000.jpg
```

正常结束后目录会原子改名为 `.complete` 并写入 `result.json`。异常退出会保留 `.partial`；
运行以下命令列出完整数据和可恢复数据：

```bash
lingbot-recap audit \
  --experience-root /home/mzm/lerobot_data/recap_experience
```

每帧包含：策略观测、策略建议动作、实际执行动作、动作来源、控制状态、时间戳与两路图像。
每个接管事件包含原因、对齐开始/完成、leader 力矩读回、人工控制授予、交还与最终结果。
完整字段见 [数据格式](docs/DATA_FORMAT.md)。

这些数据首先是 RECAP/RL experience，不能直接当成行为克隆 SFT episode。若要做 SFT，必须由
独立清洗工具抽取连续、成功且动作语义一致的人工片段，不能把失败策略前缀或 leader 自动对齐
过程混进监督动作。

## 代码结构

```text
lingbot_recap/
  detectors.py   振荡与无进展检测
  inputs.py      终端键盘、两键 evdev 与多输入合并
  handoff.py     控制权仲裁及两阶段接管状态机
  hardware.py    LeRobot 0.4.x / Feetech SO-101 适配
  cameras.py     top/wrist 观测采集
  policy.py      LingBot HTTP 推理客户端
  journal.py     追加式、可恢复 experience 记录
  runtime.py     采集调度循环
tools/
  identify_two_key_keyboard.py
```

状态机与模块所有权见 [架构说明](docs/ARCHITECTURE.md)。

## 开发与无电机测试

```bash
python -m pip install -e '.[dev]'
pytest
python -m compileall -q lingbot_recap tools
```

CI/单元测试不得连接真实串口。真机测试按“无物体、低速、工作区清空、第二人在电源开关旁”
逐步进行；确认按钮识别和状态日志正确后，才能加入物体 rollout。
