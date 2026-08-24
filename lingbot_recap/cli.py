from __future__ import annotations

import argparse
from pathlib import Path

from .cameras import CameraConfig, OpenCVCameraRig
from .hardware import SO101BusArm
from .inputs import KeyboardEventSource
from .journal import ExperienceJournal
from .policy import LingBotHTTPPolicy
from .runtime import CollectorConfig, ExperienceCollector


DEFAULT_CAL_ROOT = Path.home() / ".cache/huggingface/lerobot/calibration"


def collect(args) -> None:
    policy = LingBotHTTPPolicy(args.server, use_length=args.execute_length)
    health = policy.health()
    if not health.get("model_loaded"):
        raise SystemExit(f"LingBot service is not ready: {health}")
    follower = SO101BusArm(args.follower_port, args.follower_calibration)
    leader = SO101BusArm(args.leader_port, args.leader_calibration)
    cameras = OpenCVCameraRig(
        {
            "top": CameraConfig(args.top_camera),
            "wrist": CameraConfig(args.wrist_camera),
        }
    )
    config = CollectorConfig(
        task=args.task,
        policy_checkpoint=args.policy_checkpoint,
        experience_root=Path(args.experience_root),
        fps=args.fps,
        execute_length=args.execute_length,
        detectors_enabled=not args.disable_detectors,
    )
    with KeyboardEventSource() as keyboard:
        path = ExperienceCollector(
            config, follower, leader, cameras, policy, keyboard
        ).run()
    print(f"experience saved: {path}")


def audit(args) -> None:
    root = Path(args.experience_root)
    incomplete = ExperienceJournal.recover_incomplete(root)
    complete = sorted(root.glob("episode_*.complete"))
    print(f"complete={len(complete)} incomplete={len(incomplete)}")
    for path in incomplete:
        print(f"RECOVERABLE {path}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="lingbot-recap")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.set_defaults(func=collect)
    run.add_argument("--server", default="http://116.196.82.74:8007")
    run.add_argument("--task", required=True)
    run.add_argument("--policy-checkpoint", required=True)
    run.add_argument("--experience-root", default="/home/mzm/lerobot_data/recap_experience")
    run.add_argument("--follower-port", default="/dev/ttyACM1")
    run.add_argument("--leader-port", default="/dev/ttyACM0")
    run.add_argument("--top-camera", default="/dev/video2")
    run.add_argument("--wrist-camera", default="/dev/video1")
    run.add_argument(
        "--follower-calibration",
        default=str(DEFAULT_CAL_ROOT / "robots/so101_follower/None.json"),
    )
    run.add_argument(
        "--leader-calibration",
        default=str(DEFAULT_CAL_ROOT / "teleoperators/so101_leader/None.json"),
    )
    run.add_argument("--fps", type=float, default=30.0)
    run.add_argument("--execute-length", type=int, default=16)
    run.add_argument("--disable-detectors", action="store_true")

    check = commands.add_parser("audit")
    check.set_defaults(func=audit)
    check.add_argument("--experience-root", default="/home/mzm/lerobot_data/recap_experience")
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
