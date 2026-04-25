#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a workspace of MCAP teleop episodes to LeRobot dataset format.

Expected source layout:

    <src>/
      <task_a>/
        <episode_dir_1>/output.mcap
        <episode_dir_2>/output.mcap
        ...
      <task_b>/
        ...

Each ``output.mcap`` is treated as one episode. One ``LeRobotDataset`` is
produced per task directory at ``<dst>/<task>/{meta,data,videos}/...``.

Topics consumed (foxglove protobuf):

  * ``/left-robot-state`` + ``/left-gripper-state``      → state[0:7]
  * ``/right-robot-state`` + ``/right-gripper-state``    → state[7:14]
  * ``/action-left-robot-state`` + ``/action-left-gripper-state``    → action[0:7]
  * ``/action-right-robot-state`` + ``/action-right-gripper-state``  → action[7:14]
  * ``/left_camera-images-rgb/image-raw``      → observation.images.left
  * ``/right_camera-images-rgb/image-raw``     → observation.images.right
  * ``/top_camera-images-right_rgb/image-raw`` → observation.images.top
  * ``/instruction``                           → task string

Master clock = top camera timestamps (≈30 Hz). State / action are picked by
nearest-timestamp lookup.

Usage:

    python examples/port_datasets/port_mcap_workspace.py \\
        --src /workspace \\
        --dst /workspace/lerobot \\
        --hf-user sagrawal0410 \\
        --image-size 512

Dependencies (one-off):

    pip install mcap mcap-protobuf-support av
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from lerobot.datasets import LeRobotDataset

# ── Topic configuration ──────────────────────────────────────────────────────
CAMERA_TOPICS: dict[str, str] = {
    "left": "/left_camera-images-rgb/image-raw",
    "right": "/right_camera-images-rgb/image-raw",
    "top": "/top_camera-images-right_rgb/image-raw",
}
MASTER_CAMERA = "top"  # frame clock anchor

LEFT_ROBOT = "/left-robot-state"
LEFT_GRIP = "/left-gripper-state"
RIGHT_ROBOT = "/right-robot-state"
RIGHT_GRIP = "/right-gripper-state"

ACT_LEFT_ROBOT = "/action-left-robot-state"
ACT_LEFT_GRIP = "/action-left-gripper-state"
ACT_RIGHT_ROBOT = "/action-right-robot-state"
ACT_RIGHT_GRIP = "/action-right-gripper-state"

INSTRUCTION = "/instruction"

ARM_DIM = 6      # joint positions per arm (verified from sample dump)
GRIP_DIM = 1     # single position per gripper
STATE_DIM = (ARM_DIM + GRIP_DIM) * 2  # = 14
ACTION_DIM = STATE_DIM

ALL_TOPICS = [
    *CAMERA_TOPICS.values(),
    LEFT_ROBOT, LEFT_GRIP, RIGHT_ROBOT, RIGHT_GRIP,
    ACT_LEFT_ROBOT, ACT_LEFT_GRIP, ACT_RIGHT_ROBOT, ACT_RIGHT_GRIP,
    INSTRUCTION,
]

logger = logging.getLogger("port_mcap")


# ── Small helpers ────────────────────────────────────────────────────────────
def to_array(x) -> np.ndarray:
    """Coerce a protobuf scalar or repeated field into a 1-D float32 ndarray."""
    if hasattr(x, "__len__"):
        return np.asarray(list(x), dtype=np.float32)
    return np.asarray([x], dtype=np.float32)


def state_vec(left_arm, left_grip, right_arm, right_grip) -> np.ndarray:
    """Build a 14-D vector from one decoded message of each topic."""
    parts = [
        to_array(left_arm.position)[:ARM_DIM],
        to_array(left_grip.position)[:GRIP_DIM],
        to_array(right_arm.position)[:ARM_DIM],
        to_array(right_grip.position)[:GRIP_DIM],
    ]
    # Pad if a message is shorter than expected (defensive — never expected).
    parts = [
        np.pad(p, (0, n - len(p))) if len(p) < n else p
        for p, n in zip(parts, [ARM_DIM, GRIP_DIM, ARM_DIM, GRIP_DIM])
    ]
    return np.concatenate(parts).astype(np.float32)


def nearest_indices(target_ts: np.ndarray, source_ts: np.ndarray) -> np.ndarray:
    """For each ``target_ts``, return the index in ``source_ts`` of the nearest value."""
    if len(source_ts) == 0:
        raise ValueError("source_ts is empty")
    src = np.asarray(source_ts, dtype=np.int64)
    tgt = np.asarray(target_ts, dtype=np.int64)
    idx = np.searchsorted(src, tgt)
    idx = np.clip(idx, 0, len(src) - 1)
    # Compare to idx-1; pick whichever is closer.
    prev = np.clip(idx - 1, 0, len(src) - 1)
    use_prev = (idx > 0) & (np.abs(src[prev] - tgt) <= np.abs(src[idx] - tgt))
    return np.where(use_prev, prev, idx)


# ── MCAP reading ─────────────────────────────────────────────────────────────
def read_mcap(mcap_path: Path) -> dict[str, list[tuple[int, object]]]:
    """Read all required topics from one MCAP. Returns dict[topic] -> [(ts_ns, msg)]."""
    out: dict[str, list[tuple[int, object]]] = defaultdict(list)
    with mcap_path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages(topics=ALL_TOPICS):
            out[channel.topic].append((int(message.publish_time), decoded))
    for topic in out:
        out[topic].sort(key=lambda kv: kv[0])
    return out


def decode_h265_frames(chunks: list[tuple[int, bytes]]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Decode an ordered list of H.265 chunks.

    Returns ``(timestamps_ns, [HxWx3 uint8 RGB ndarray, ...])``. The two lists
    are aligned 1:1 by decode order.
    """
    if not chunks:
        return np.zeros(0, dtype=np.int64), []
    codec = av.CodecContext.create("hevc", "r")
    out_frames: list[np.ndarray] = []
    out_ts: list[int] = []
    for ts, data in chunks:
        try:
            packets = codec.parse(bytes(data))
        except Exception as e:  # noqa: BLE001
            logger.debug("parse error at ts=%d: %s", ts, e)
            continue
        for pkt in packets:
            try:
                frames = codec.decode(pkt)
            except Exception as e:  # noqa: BLE001
                logger.debug("decode error at ts=%d: %s", ts, e)
                continue
            for fr in frames:
                out_frames.append(fr.to_ndarray(format="rgb24"))
                out_ts.append(ts)
    try:
        for fr in codec.decode(None):
            out_frames.append(fr.to_ndarray(format="rgb24"))
            out_ts.append(chunks[-1][0])
    except Exception:
        pass
    return np.asarray(out_ts, dtype=np.int64), out_frames


def resize_rgb(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Letterbox-free resize to (h, w) using PyAV (no extra deps)."""
    h, w = hw
    if img.shape[:2] == (h, w):
        return img
    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    frame = frame.reformat(width=w, height=h, format="rgb24")
    return frame.to_ndarray(format="rgb24")


# ── Per-episode conversion ───────────────────────────────────────────────────
def convert_episode(
    mcap_path: Path,
    dataset: LeRobotDataset,
    image_size: tuple[int, int],
    fallback_task: str,
) -> int:
    """Convert one MCAP into one episode of ``dataset``. Returns frames written."""
    msgs = read_mcap(mcap_path)

    required = [
        CAMERA_TOPICS[MASTER_CAMERA],
        LEFT_ROBOT, LEFT_GRIP, RIGHT_ROBOT, RIGHT_GRIP,
        ACT_LEFT_ROBOT, ACT_LEFT_GRIP, ACT_RIGHT_ROBOT, ACT_RIGHT_GRIP,
    ]
    missing = [t for t in required if not msgs.get(t)]
    if missing:
        raise RuntimeError(f"missing topics: {missing}")

    # Decode each camera (independently — frame counts may differ slightly).
    cam_ts: dict[str, np.ndarray] = {}
    cam_frames: dict[str, list[np.ndarray]] = {}
    for name, topic in CAMERA_TOPICS.items():
        chunks = [(ts, m.data) for ts, m in msgs.get(topic, [])]
        ts_arr, frames = decode_h265_frames(chunks)
        if len(frames) == 0:
            raise RuntimeError(f"camera {name!r} ({topic}) decoded 0 frames")
        cam_ts[name] = ts_arr
        cam_frames[name] = frames

    # Master timeline = master camera timestamps.
    master_ts = cam_ts[MASTER_CAMERA]
    master_frames = cam_frames[MASTER_CAMERA]
    n = min(len(master_ts), len(master_frames))
    master_ts = master_ts[:n]

    # Pre-extract topic timestamp arrays.
    def ts_array(topic: str) -> np.ndarray:
        return np.asarray([t for t, _ in msgs[topic]], dtype=np.int64)

    state_idx = {
        "lr": nearest_indices(master_ts, ts_array(LEFT_ROBOT)),
        "lg": nearest_indices(master_ts, ts_array(LEFT_GRIP)),
        "rr": nearest_indices(master_ts, ts_array(RIGHT_ROBOT)),
        "rg": nearest_indices(master_ts, ts_array(RIGHT_GRIP)),
    }
    action_idx = {
        "lr": nearest_indices(master_ts, ts_array(ACT_LEFT_ROBOT)),
        "lg": nearest_indices(master_ts, ts_array(ACT_LEFT_GRIP)),
        "rr": nearest_indices(master_ts, ts_array(ACT_RIGHT_ROBOT)),
        "rg": nearest_indices(master_ts, ts_array(ACT_RIGHT_GRIP)),
    }
    cam_idx = {
        name: nearest_indices(master_ts, cam_ts[name][: len(cam_frames[name])])
        for name in CAMERA_TOPICS
    }

    # Language task.
    if msgs.get(INSTRUCTION):
        task = str(msgs[INSTRUCTION][0][1].data) or fallback_task
    else:
        task = fallback_task

    # Drop a frame if any camera has no decoded frame at that index (defensive).
    for k, idx in cam_idx.items():
        if np.any(idx >= len(cam_frames[k])):
            n_safe = int(np.argmax(idx >= len(cam_frames[k])))
            n = min(n, n_safe)
    n = max(0, n)
    if n == 0:
        raise RuntimeError("zero frames after alignment")

    written = 0
    for i in range(n):
        try:
            s = state_vec(
                msgs[LEFT_ROBOT][state_idx["lr"][i]][1],
                msgs[LEFT_GRIP][state_idx["lg"][i]][1],
                msgs[RIGHT_ROBOT][state_idx["rr"][i]][1],
                msgs[RIGHT_GRIP][state_idx["rg"][i]][1],
            )
            a = state_vec(
                msgs[ACT_LEFT_ROBOT][action_idx["lr"][i]][1],
                msgs[ACT_LEFT_GRIP][action_idx["lg"][i]][1],
                msgs[ACT_RIGHT_ROBOT][action_idx["rr"][i]][1],
                msgs[ACT_RIGHT_GRIP][action_idx["rg"][i]][1],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("frame %d: state/action build failed: %s", i, e)
            continue

        frame = {
            "observation.state": s.astype(np.float32),
            "action": a.astype(np.float32),
            "task": task,
        }
        for name in CAMERA_TOPICS:
            img = cam_frames[name][cam_idx[name][i]]
            frame[f"observation.images.{name}"] = resize_rgb(img, image_size)

        dataset.add_frame(frame)
        written += 1

    if written == 0:
        return 0
    dataset.save_episode()
    return written


# ── Per-task driver ──────────────────────────────────────────────────────────
def features_dict(image_size: tuple[int, int]) -> dict:
    h, w = image_size
    feats: dict = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": {
                "axes": [
                    *(f"left_arm_j{i}" for i in range(ARM_DIM)),
                    "left_gripper",
                    *(f"right_arm_j{i}" for i in range(ARM_DIM)),
                    "right_gripper",
                ],
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": {
                "axes": [
                    *(f"left_arm_j{i}_cmd" for i in range(ARM_DIM)),
                    "left_gripper_cmd",
                    *(f"right_arm_j{i}_cmd" for i in range(ARM_DIM)),
                    "right_gripper_cmd",
                ],
            },
        },
    }
    for name in CAMERA_TOPICS:
        feats[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return feats


def convert_task(
    task_dir: Path,
    dst_root: Path,
    repo_id: str,
    fps: int,
    image_size: tuple[int, int],
    vcodec: str,
    skip_existing: bool,
) -> None:
    task_name = task_dir.name
    out_root = dst_root / task_name
    if skip_existing and (out_root / "meta" / "info.json").exists():
        logger.info("[skip] %s — already converted at %s", task_name, out_root)
        return

    mcaps = sorted(task_dir.glob("*/output.mcap"))
    if not mcaps:
        logger.warning("[skip] %s — no */output.mcap found", task_name)
        return

    logger.info("[task=%s] %d episodes → %s", task_name, len(mcaps), out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features_dict(image_size),
        root=out_root,
        robot_type="bimanual_robot",
        use_videos=True,
        vcodec=vcodec,
    )

    n_ok = 0
    try:
        for k, mcap in enumerate(mcaps):
            try:
                n = convert_episode(
                    mcap_path=mcap,
                    dataset=dataset,
                    image_size=image_size,
                    fallback_task=task_name.replace("_", " "),
                )
                logger.info("  [%d/%d] %s → %d frames", k + 1, len(mcaps), mcap.parent.name, n)
                if n > 0:
                    n_ok += 1
            except Exception as e:  # noqa: BLE001
                logger.error("  [%d/%d] %s FAILED: %s", k + 1, len(mcaps), mcap, e)
                logger.debug("%s", traceback.format_exc())
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
    finally:
        dataset.finalize()

    logger.info("[task=%s] done — %d/%d episodes converted", task_name, n_ok, len(mcaps))


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=Path("/workspace"),
                   help="Root of MCAP workspace (default: /workspace).")
    p.add_argument("--dst", type=Path, default=Path("/workspace/lerobot"),
                   help="Output root, one subdir per task (default: /workspace/lerobot).")
    p.add_argument("--hf-user", required=True,
                   help="HF Hub username — used to namespace each task's repo_id (no upload happens here).")
    p.add_argument("--fps", type=int, default=30, help="Master FPS (camera rate). Default: 30.")
    p.add_argument("--image-size", type=int, default=512,
                   help="Square resize for stored video frames. Default: 512.")
    p.add_argument("--vcodec", default="libsvtav1",
                   help="Output video codec for LeRobotDataset (default: libsvtav1; try 'h264' for speed).")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Optional explicit list of task subdir names. Default: all subdirs of --src.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip task dirs that already have meta/info.json under --dst.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    src: Path = args.src
    if not src.is_dir():
        logger.error("--src %s does not exist or is not a directory", src)
        return 2

    if args.tasks:
        task_dirs = [src / t for t in args.tasks]
    else:
        task_dirs = sorted(
            d for d in src.iterdir()
            if d.is_dir() and d.resolve() != args.dst.resolve()
            and not d.name.startswith(".") and d.name != "lerobot"
        )

    if not task_dirs:
        logger.error("No task dirs found under %s", src)
        return 2

    image_size = (args.image_size, args.image_size)

    for td in task_dirs:
        if not td.is_dir():
            logger.warning("[skip] %s — not a directory", td)
            continue
        repo_id = f"{args.hf_user}/{td.name}"
        try:
            convert_task(
                task_dir=td,
                dst_root=args.dst,
                repo_id=repo_id,
                fps=args.fps,
                image_size=image_size,
                vcodec=args.vcodec,
                skip_existing=args.skip_existing,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("[task=%s] FAILED: %s", td.name, e)
            logger.debug("%s", traceback.format_exc())

    logger.info("All done. Output root: %s", args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
