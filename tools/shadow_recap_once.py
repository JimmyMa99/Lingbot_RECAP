#!/usr/bin/env python3
"""Read one live observation and query RECAP without commanding the robot."""

from __future__ import annotations

import argparse
import base64
import json

import requests

from lingbot_recap.cameras import CameraConfig, OpenCVCameraRig
from lingbot_recap.hardware import MOTOR_NAMES, SO101BusArm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8011")
    parser.add_argument("--task", default="把吸管放进杯子里")
    parser.add_argument("--follower-port", default="/dev/ttyACM1")
    parser.add_argument("--follower-calibration", required=True)
    parser.add_argument("--top-camera", default="/dev/video2")
    parser.add_argument("--wrist-camera", default="/dev/video1")
    args = parser.parse_args()
    arm = SO101BusArm(args.follower_port, args.follower_calibration)
    cameras = OpenCVCameraRig({"top": CameraConfig(args.top_camera), "wrist": CameraConfig(args.wrist_camera)})
    arm.connect()
    cameras.connect()
    try:
        state = arm.read_positions()
        images = cameras.capture_jpegs()
        response = requests.post(
            f"{args.server.rstrip('/')}/infer",
            json={
                "image": {name: base64.b64encode(value).decode("ascii") for name, value in images.items()},
                "state": [state[name] for name in MOTOR_NAMES], "task": args.task,
                "robo_name": "so_arm101", "use_length": 16,
            }, timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        print(json.dumps({"state": state, "recap": payload["recap"], "action_shape": [len(payload["action"]["action"]), 6]}, ensure_ascii=False, indent=2))
    finally:
        cameras.disconnect()
        arm.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
