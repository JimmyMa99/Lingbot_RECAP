from __future__ import annotations

import argparse
from pathlib import Path

from .cameras import CameraConfig, OpenCVCameraRig
from .distillation import ExperienceRelabeler, RelabelConfig, find_experiences
from .hardware import SO101BusArm
from .inputs import CompositeEventSource, KeyboardEventSource, LinuxTwoButtonEventSource
from .journal import ExperienceJournal
from .lerobot_export import export_lerobot_distillation_dataset
from .multi_policy import MultiPolicyRouter
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
    sources = [KeyboardEventSource()]
    if args.button_config:
        sources.append(LinuxTwoButtonEventSource(args.button_config))
    with CompositeEventSource(*sources) as events:
        path = ExperienceCollector(
            config, follower, leader, cameras, policy, events
        ).run()
    print(f"experience saved: {path}")


def audit(args) -> None:
    root = Path(args.experience_root)
    incomplete = ExperienceJournal.recover_incomplete(root)
    complete = sorted(root.glob("episode_*.complete"))
    print(f"complete={len(complete)} incomplete={len(incomplete)}")
    for path in incomplete:
        print(f"RECOVERABLE {path}")


def validate_teachers(args) -> None:
    router = MultiPolicyRouter.from_file(args.teacher_registry)
    for key, health in router.validate_health().items():
        print(f"READY {key}: {health}")


def relabel(args) -> None:
    router = MultiPolicyRouter.from_file(args.teacher_registry)
    allowed_sources = ["lingbot_policy"]
    if args.include_human_states:
        allowed_sources.append("human_intervention")
    relabeler = ExperienceRelabeler(
        router,
        RelabelConfig(
            stride=args.stride,
            max_frames=args.max_frames,
            allowed_action_sources=tuple(allowed_sources),
        ),
    )
    if args.episode:
        episodes = [Path(args.episode)]
    else:
        episodes = find_experiences(args.experience_root, include_partial=args.include_partial)
    if not episodes:
        raise SystemExit("no RECAP experiences found")
    for episode in episodes:
        summary = relabeler.label_episode(episode, overwrite=args.overwrite)
        print(
            f"LABELED {summary.episode} teacher={summary.teacher_key} "
            f"frames={summary.labeled_frames} skipped={summary.skipped_frames}"
        )


def export_distill(args) -> None:
    episodes = [
        path
        for path in find_experiences(args.experience_root, include_partial=False)
        if (path / "teacher_labels.jsonl").exists()
    ]
    if not episodes:
        raise SystemExit("no completed, teacher-labeled experiences found")
    summary = export_lerobot_distillation_dataset(
        episodes,
        args.output_root,
        args.repo_id,
        use_videos=not args.no_videos,
        min_segment_frames=args.min_segment_frames,
    )
    print(
        f"EXPORTED {summary.output_root} experiences={summary.source_experiences} "
        f"episodes={summary.output_episodes} frames={summary.output_frames}"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="lingbot-recap")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.set_defaults(func=collect)
    run.add_argument("--server", default="http://127.0.0.1:8007")
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
    run.add_argument(
        "--button-config",
        help="JSON generated by tools/identify_two_key_keyboard.py",
    )

    check = commands.add_parser("audit")
    check.set_defaults(func=audit)
    check.add_argument("--experience-root", default="/home/mzm/lerobot_data/recap_experience")

    teachers = commands.add_parser("validate-teachers")
    teachers.set_defaults(func=validate_teachers)
    teachers.add_argument("--teacher-registry", required=True)

    labels = commands.add_parser("relabel")
    labels.set_defaults(func=relabel)
    labels.add_argument("--teacher-registry", required=True)
    labels.add_argument("--experience-root", default="/home/mzm/lerobot_data/recap_experience")
    labels.add_argument("--episode", help="label one .complete/.partial experience directory")
    labels.add_argument("--stride", type=int, default=1)
    labels.add_argument("--max-frames", type=int)
    labels.add_argument("--include-human-states", action="store_true")
    labels.add_argument("--include-partial", action="store_true")
    labels.add_argument("--overwrite", action="store_true")

    export = commands.add_parser("export-distill")
    export.set_defaults(func=export_distill)
    export.add_argument("--experience-root", default="/home/mzm/lerobot_data/recap_experience")
    export.add_argument("--output-root", required=True)
    export.add_argument("--repo-id", default="mzm/lingbot_recap_distillation")
    export.add_argument("--min-segment-frames", type=int, default=2)
    export.add_argument("--no-videos", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
