from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    device: str
    width: int = 640
    height: int = 480
    jpeg_quality: int = 90


class OpenCVCameraRig:
    def __init__(self, configs: dict[str, CameraConfig]):
        import cv2

        self.cv2 = cv2
        self.configs = configs
        self.captures = {}

    def connect(self) -> None:
        for name, config in self.configs.items():
            capture = self.cv2.VideoCapture(config.device, self.cv2.CAP_V4L2)
            capture.set(self.cv2.CAP_PROP_FRAME_WIDTH, config.width)
            capture.set(self.cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            if not capture.isOpened():
                raise RuntimeError(f"camera {name} failed to open: {config.device}")
            self.captures[name] = capture

    def capture_jpegs(self) -> dict[str, bytes]:
        result = {}
        for name, capture in self.captures.items():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"camera {name} failed to read")
            quality = self.configs[name].jpeg_quality
            ok, encoded = self.cv2.imencode(
                ".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            if not ok:
                raise RuntimeError(f"camera {name} failed to encode")
            result[name] = encoded.tobytes()
        return result

    def disconnect(self) -> None:
        for capture in self.captures.values():
            capture.release()
        self.captures.clear()
