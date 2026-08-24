#!/usr/bin/env python3
"""Identify a two-button USB keyboard and write a Lingbot_RECAP mapping.

This tool uses Linux's built-in evdev ABI and has no third-party dependencies.
It records key-down events only; key-up and auto-repeat events are ignored.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import select
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


INPUT_EVENT = struct.Struct("@llHHi")
EV_KEY = 0x01
KEY_UP = 0
KEY_DOWN = 1


@dataclass(frozen=True)
class Device:
    path: str
    event_path: str
    name: str
    vendor: str | None = None
    product: str | None = None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def device_info(path: str) -> Device:
    event_path = os.path.realpath(path)
    event_name = Path(event_path).name
    sys_device = Path("/sys/class/input") / event_name / "device"
    return Device(
        path=path,
        event_path=event_path,
        name=_read_text(sys_device / "name") or event_name,
        vendor=_read_text(sys_device / "id/vendor"),
        product=_read_text(sys_device / "id/product"),
    )


def keyboard_devices() -> list[Device]:
    stable = sorted(glob.glob("/dev/input/by-id/*-event-kbd"))
    paths = stable or sorted(glob.glob("/dev/input/event*"))
    devices: list[Device] = []
    seen: set[str] = set()
    for path in paths:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        devices.append(device_info(path))
    return devices


def key_names() -> dict[int, str]:
    result: dict[int, str] = {}
    pattern = re.compile(r"^#define\s+(KEY_[A-Z0-9_]+)\s+(0x[0-9a-fA-F]+|[0-9]+)\b")
    for header in (
        Path("/usr/include/linux/input-event-codes.h"),
        Path("/usr/include/linux/input.h"),
    ):
        try:
            lines = header.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            match = pattern.match(line)
            if match:
                result[int(match.group(2), 0)] = match.group(1)
        if result:
            break
    return result


def iter_events(fd: int):
    buffer = bytearray()
    while True:
        readable, _, _ = select.select([fd], [], [], None)
        if not readable:
            continue
        chunk = os.read(fd, INPUT_EVENT.size * 32)
        if not chunk:
            raise RuntimeError("input device disconnected")
        buffer.extend(chunk)
        while len(buffer) >= INPUT_EVENT.size:
            raw = buffer[: INPUT_EVENT.size]
            del buffer[: INPUT_EVENT.size]
            yield INPUT_EVENT.unpack(raw)


def capture_button(events, prompt: str, names: dict[int, str]) -> dict[str, int | str]:
    print(prompt, flush=True)
    pressed: int | None = None
    for _sec, _usec, event_type, code, value in events:
        if event_type != EV_KEY:
            continue
        if value == KEY_DOWN and pressed is None:
            pressed = code
            print(f"  detected code={code} name={names.get(code, f'KEY_CODE_{code}')}", flush=True)
        elif value == KEY_UP and code == pressed:
            return {"code": code, "name": names.get(code, f"KEY_CODE_{code}")}
    raise RuntimeError("input device ended while capturing a button")


def choose_device(devices: list[Device]) -> Device:
    if not devices:
        raise SystemExit("No keyboard event device found under /dev/input")
    if len(devices) == 1:
        return devices[0]
    print("Keyboard candidates:")
    for index, device in enumerate(devices, 1):
        print(f"  {index}. {device.path}  [{device.name}]")
    while True:
        answer = input("Select the two-button keyboard number: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(devices):
            return devices[int(answer) - 1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Identify a Linux two-button USB keyboard for Lingbot_RECAP"
    )
    result.add_argument("--list", action="store_true", help="list keyboard event devices and exit")
    result.add_argument("--device", help="prefer a stable /dev/input/by-id/*-event-kbd path")
    result.add_argument(
        "--output",
        default="configs/two_button_keyboard.local.json",
        help="mapping JSON to create",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if sys.platform != "linux":
        raise SystemExit("This tool must run on the Linux operation machine")
    devices = keyboard_devices()
    if args.list:
        for device in devices:
            print(
                f"{device.path}\tname={device.name!r}\tvendor={device.vendor}\tproduct={device.product}"
            )
        return
    device = device_info(args.device) if args.device else choose_device(devices)
    print(f"Using {device.path} ({device.name})")
    print("Keep the robot powered off while identifying buttons.")
    try:
        fd = os.open(device.path, os.O_RDONLY)
    except PermissionError as exc:
        raise SystemExit(
            f"Cannot read {device.path}. Add the user to the input group, log out/in, then retry.\n"
            "  sudo usermod -aG input \"$USER\""
        ) from exc
    try:
        events = iter_events(fd)
        names = key_names()
        align = capture_button(
            events,
            "Press and release button 1 (pause policy and align leader to follower).",
            names,
        )
        release = capture_button(
            events,
            "Press and release button 2 (release leader torque and grant human control).",
            names,
        )
    finally:
        os.close(fd)
    if align["code"] == release["code"]:
        raise SystemExit("Both buttons emitted the same key code; configure the keypad and retry")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "device": device.path,
        "device_info": asdict(device),
        "buttons": {"align": align, "release": release},
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved mapping: {output}")
    print("This file is machine-specific; keep it out of Git.")


if __name__ == "__main__":
    main()
