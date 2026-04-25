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

Two output modes (``--mode``):

* **merged** (default, recommended for SmolVLA-style base pretraining)

  Produces ONE LeRobot dataset under ``<dst>/<merged-name>/`` containing every
  ``output.mcap`` from every ``<task>/`` as one episode. Each episode keeps its
  own ``task`` string from ``/instruction`` (fallback = task dir name), so the
  resulting ``meta/tasks.parquet`` mirrors how ``lerobot/smolvla_base`` itself
  was trained: a flat pool of language-tagged episodes.

* **per-task**

  Produces ONE LeRobot dataset per task directory under ``<dst>/<task>/``.
  Use this when you want to train per-task expert policies, OR when you want
  the option to merge later via ``aggregate_per_task_datasets.py``.

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

Usage (default — single merged dataset, SmolVLA-style):

    python examples/port_datasets/port_mcap_workspace.py \\
        --src /workspace \\
        --dst /workspace/lerobot \\
        --hf-user sagrawal0410 \\
        --image-size 512
    # → /workspace/lerobot/_combined/{meta,data,videos}/...

Usage (per-task experts, multiple tasks in parallel):

    python examples/port_datasets/port_mcap_workspace.py \\
        --src /workspace --dst /workspace/lerobot --hf-user sagrawal0410 \\
        --mode per-task --num-workers 4

Usage (fast — GPU decode + GPU encode where available):

    python examples/port_datasets/port_mcap_workspace.py \\
        --src /workspace --dst /workspace/lerobot --hf-user sagrawal0410 \\
        --image-size 512 \\
        --vcodec h264_nvenc \\
        --hwaccel cuda \\
        --image-writer-threads 8

Speed knobs (rough order of impact on a typical CPU pod):
    --vcodec h264                       (~10× faster than libsvtav1; default)
    --vcodec h264_nvenc                 (GPU encode; needs FFmpeg w/ NVENC)
    --hwaccel cuda                      (GPU HEVC decode via NVDEC)
    --num-workers N                     (per-task mode only — parallel tasks)
    --image-writer-threads K            (threads inside the writer)
    --image-size 0                      (skip per-frame resize)

Dependencies (one-off):

    pip install mcap mcap-protobuf-support av
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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


def decode_h265_frames(
    chunks: list[tuple[int, bytes]],
    out_hw: tuple[int, int] | None = None,
    hwaccel: str | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Decode an ordered list of H.265 chunks.

    Args:
        chunks: ordered list of ``(timestamp_ns, encoded_bytes)``.
        out_hw: optional ``(h, w)``. When provided, decoded frames are resized
            inside the same PyAV pipeline (one ``sws_scale`` pass instead of a
            decode → ndarray → re-wrap → resize round-trip).
        hwaccel: optional PyAV hwaccel hint. Pass ``"cuda"`` to use NVDEC when
            available (requires an FFmpeg with ``hevc_cuvid``). Falls back to
            CPU decode silently if unsupported.

    Returns ``(timestamps_ns, [HxWx3 uint8 RGB ndarray, ...])``.
    """
    if not chunks:
        return np.zeros(0, dtype=np.int64), []

    codec = None
    if hwaccel == "cuda":
        try:
            codec = av.CodecContext.create("hevc_cuvid", "r")
        except Exception as e:  # noqa: BLE001
            logger.debug("hevc_cuvid unavailable, falling back to CPU: %s", e)
            codec = None
    if codec is None:
        codec = av.CodecContext.create("hevc", "r")
        # CPU HEVC decoder benefits a lot from threading.
        codec.thread_type = "FRAME"
        codec.thread_count = max(1, (os.cpu_count() or 4) // 2)

    out_frames: list[np.ndarray] = []
    out_ts: list[int] = []

    def _emit(fr: av.VideoFrame, ts: int) -> None:
        if out_hw is not None:
            h, w = out_hw
            fr = fr.reformat(width=w, height=h, format="rgb24")
            out_frames.append(fr.to_ndarray(format="rgb24"))
        else:
            out_frames.append(fr.to_ndarray(format="rgb24"))
        out_ts.append(ts)

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
                _emit(fr, ts)
    try:
        for fr in codec.decode(None):
            _emit(fr, chunks[-1][0])
    except Exception:
        pass
    return np.asarray(out_ts, dtype=np.int64), out_frames


# ── Per-episode conversion ───────────────────────────────────────────────────
def convert_episode(
    mcap_path: Path,
    dataset: LeRobotDataset,
    image_size: tuple[int, int] | None,
    fallback_task: str,
    hwaccel: str | None = None,
) -> int:
    """Convert one MCAP into one episode of ``dataset``. Returns frames written.

    ``image_size=None`` keeps native camera resolution (fastest — skips the
    sws_scale pass entirely).
    """
    msgs = read_mcap(mcap_path)

    required = [
        CAMERA_TOPICS[MASTER_CAMERA],
        LEFT_ROBOT, LEFT_GRIP, RIGHT_ROBOT, RIGHT_GRIP,
        ACT_LEFT_ROBOT, ACT_LEFT_GRIP, ACT_RIGHT_ROBOT, ACT_RIGHT_GRIP,
    ]
    missing = [t for t in required if not msgs.get(t)]
    if missing:
        raise RuntimeError(f"missing topics: {missing}")

    # Decode the 3 cameras in parallel threads. PyAV releases the GIL during
    # libavcodec calls, so this is a real ~3× speedup on a multicore CPU.
    def _decode_one(name: str) -> tuple[str, np.ndarray, list[np.ndarray]]:
        chunks = [(ts, m.data) for ts, m in msgs.get(CAMERA_TOPICS[name], [])]
        ts_arr, frames = decode_h265_frames(chunks, out_hw=image_size, hwaccel=hwaccel)
        if not frames:
            raise RuntimeError(f"camera {name!r} ({CAMERA_TOPICS[name]}) decoded 0 frames")
        return name, ts_arr, frames

    cam_ts: dict[str, np.ndarray] = {}
    cam_frames: dict[str, list[np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=len(CAMERA_TOPICS)) as ex:
        for name, ts_arr, frames in ex.map(_decode_one, CAMERA_TOPICS.keys()):
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
            frame[f"observation.images.{name}"] = cam_frames[name][cam_idx[name][i]]

        dataset.add_frame(frame)
        written += 1

    if written == 0:
        return 0
    dataset.save_episode()
    return written


# ── Per-task driver ──────────────────────────────────────────────────────────
def features_dict(image_size: tuple[int, int]) -> dict:
    """``image_size`` here is the (h, w) actually stored in the videos."""
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


def detect_native_size(mcap_path: Path, hwaccel: str | None = None) -> tuple[int, int]:
    """Decode just the first frame of the master camera to discover (h, w)."""
    msgs = read_mcap(mcap_path)
    chunks = [(ts, m.data) for ts, m in msgs.get(CAMERA_TOPICS[MASTER_CAMERA], [])]
    if not chunks:
        raise RuntimeError(f"no master-camera frames in {mcap_path}")
    # Only decode enough chunks to emit one frame.
    _, frames = decode_h265_frames(chunks[:8], out_hw=None, hwaccel=hwaccel)
    if not frames:
        # Fall back to decoding all chunks if 8 wasn't enough for a keyframe.
        _, frames = decode_h265_frames(chunks, out_hw=None, hwaccel=hwaccel)
    if not frames:
        raise RuntimeError(f"could not decode any frames from {mcap_path}")
    h, w = frames[0].shape[:2]
    return int(h), int(w)


def _open_dataset(
    out_root: Path,
    repo_id: str,
    fps: int,
    stored_hw: tuple[int, int],
    vcodec: str,
    streaming_encoding: bool,
    image_writer_threads: int,
    image_writer_processes: int,
) -> LeRobotDataset:
    out_root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features_dict(stored_hw),
        root=out_root,
        robot_type="bimanual_robot",
        use_videos=True,
        vcodec=vcodec,
        streaming_encoding=streaming_encoding,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def _convert_one_episode_into(
    mcap_path: Path,
    dataset: LeRobotDataset,
    convert_image_size: tuple[int, int] | None,
    fallback_task: str,
    hwaccel: str | None,
    label: str,
) -> int:
    """Wrap ``convert_episode`` with logging + buffer cleanup. Returns frames written."""
    try:
        n = convert_episode(
            mcap_path=mcap_path,
            dataset=dataset,
            image_size=convert_image_size,
            fallback_task=fallback_task,
            hwaccel=hwaccel,
        )
        logger.info("  %s %s → %d frames", label, mcap_path.parent.name, n)
        return n
    except Exception as e:  # noqa: BLE001
        logger.error("  %s %s FAILED: %s", label, mcap_path, e)
        logger.debug("%s", traceback.format_exc())
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        return 0


def convert_task(
    task_dir: Path,
    dst_root: Path,
    repo_id: str,
    fps: int,
    image_size: tuple[int, int] | None,
    vcodec: str,
    skip_existing: bool,
    streaming_encoding: bool = True,
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
    hwaccel: str | None = None,
    parallel_episode_encoding: bool = True,
    log_level: str = "INFO",
) -> tuple[str, int, int]:
    """Convert one task into ``<dst_root>/<task>/``. Returns (task_name, ok, total)."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        )

    task_name = task_dir.name
    out_root = dst_root / task_name
    if skip_existing and (out_root / "meta" / "info.json").exists():
        logger.info("[skip] %s — already converted at %s", task_name, out_root)
        return task_name, 0, 0

    mcaps = sorted(task_dir.glob("*/output.mcap"))
    if not mcaps:
        logger.warning("[skip] %s — no */output.mcap found", task_name)
        return task_name, 0, 0

    stored_hw = (
        image_size if image_size is not None else detect_native_size(mcaps[0], hwaccel=hwaccel)
    )
    if image_size is None:
        logger.info("[task=%s] using native image size %s", task_name, stored_hw)

    logger.info("[task=%s] %d episodes → %s (vcodec=%s, streaming=%s, hwaccel=%s)",
                task_name, len(mcaps), out_root, vcodec, streaming_encoding, hwaccel)

    dataset = _open_dataset(
        out_root=out_root, repo_id=repo_id, fps=fps, stored_hw=stored_hw,
        vcodec=vcodec, streaming_encoding=streaming_encoding,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )
    convert_image_size = stored_hw if image_size is not None else None

    n_ok = 0
    try:
        for k, mcap in enumerate(mcaps):
            label = f"[{k + 1}/{len(mcaps)}]"
            if _convert_one_episode_into(
                mcap_path=mcap,
                dataset=dataset,
                convert_image_size=convert_image_size,
                fallback_task=task_name.replace("_", " "),
                hwaccel=hwaccel,
                label=label,
            ) > 0:
                n_ok += 1
    finally:
        _ = parallel_episode_encoding  # reserved for future
        dataset.finalize()

    logger.info("[task=%s] done — %d/%d episodes converted", task_name, n_ok, len(mcaps))
    return task_name, n_ok, len(mcaps)


def convert_merged(
    task_dirs: list[Path],
    dst_root: Path,
    merged_name: str,
    repo_id: str,
    fps: int,
    image_size: tuple[int, int] | None,
    vcodec: str,
    skip_existing: bool,
    streaming_encoding: bool = True,
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
    hwaccel: str | None = None,
    log_level: str = "INFO",
) -> tuple[str, int, int]:
    """Convert ALL tasks into ONE LeRobot dataset under ``<dst_root>/<merged_name>/``.

    Each ``output.mcap`` becomes one episode whose ``task`` field is read from
    ``/instruction`` (fallback: parent directory name). This mirrors the data
    layout used to train ``lerobot/smolvla_base``: a single language-tagged
    episode pool sampled uniformly during training.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        )

    out_root = dst_root / merged_name
    if skip_existing and (out_root / "meta" / "info.json").exists():
        logger.info("[skip-merged] %s already exists at %s", merged_name, out_root)
        return merged_name, 0, 0

    jobs: list[tuple[str, Path]] = []
    per_task_count: dict[str, int] = {}
    for td in task_dirs:
        if not td.is_dir():
            continue
        mcaps = sorted(td.glob("*/output.mcap"))
        if not mcaps:
            logger.warning("[skip-task] %s — no */output.mcap found", td.name)
            continue
        per_task_count[td.name] = len(mcaps)
        for m in mcaps:
            jobs.append((td.name, m))

    if not jobs:
        logger.error("No MCAPs found under any task dir; nothing to do.")
        return merged_name, 0, 0

    stored_hw = (
        image_size if image_size is not None
        else detect_native_size(jobs[0][1], hwaccel=hwaccel)
    )
    if image_size is None:
        logger.info("[merged] using native image size %s", stored_hw)

    total = len(jobs)
    logger.info(
        "[merged=%s] %d tasks, %d episodes total → %s "
        "(vcodec=%s, streaming=%s, hwaccel=%s)",
        merged_name, len(per_task_count), total, out_root,
        vcodec, streaming_encoding, hwaccel,
    )
    for tname, n in per_task_count.items():
        logger.info("  · %-50s  %4d episodes", tname, n)

    dataset = _open_dataset(
        out_root=out_root, repo_id=repo_id, fps=fps, stored_hw=stored_hw,
        vcodec=vcodec, streaming_encoding=streaming_encoding,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )
    convert_image_size = stored_hw if image_size is not None else None

    n_ok = 0
    try:
        for k, (task_name, mcap) in enumerate(jobs):
            label = f"[{k + 1}/{total} task={task_name}]"
            if _convert_one_episode_into(
                mcap_path=mcap,
                dataset=dataset,
                convert_image_size=convert_image_size,
                fallback_task=task_name.replace("_", " "),
                hwaccel=hwaccel,
                label=label,
            ) > 0:
                n_ok += 1
    finally:
        dataset.finalize()

    logger.info(
        "[merged=%s] done — %d/%d episodes converted across %d tasks",
        merged_name, n_ok, total, len(per_task_count),
    )
    return merged_name, n_ok, total


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=Path("/workspace"),
                   help="Root of MCAP workspace (default: /workspace).")
    p.add_argument("--dst", type=Path, default=Path("/workspace/lerobot"),
                   help="Output root (default: /workspace/lerobot).")
    p.add_argument("--hf-user", required=True,
                   help="HF Hub username — used to namespace dataset repo_id(s) (no upload happens here).")
    p.add_argument(
        "--mode", choices=["merged", "per-task"], default="merged",
        help=(
            "Output layout. 'merged' (default) writes ONE LeRobot dataset under "
            "<dst>/<merged-name>/ containing every MCAP as an episode, each tagged "
            "with its task — this matches the SmolVLA base training data layout. "
            "'per-task' writes one dataset per task subdir under <dst>/<task>/."
        ),
    )
    p.add_argument(
        "--merged-name", default="_combined",
        help="In merged mode, name of the single output subdir under <dst>. Default: _combined.",
    )
    p.add_argument("--fps", type=int, default=30, help="Master FPS (camera rate). Default: 30.")
    p.add_argument("--image-size", type=int, default=512,
                   help="Square resize for stored video frames. Use 0 to keep native resolution. Default: 512.")
    p.add_argument(
        "--vcodec", default="h264",
        help=(
            "Output video codec for LeRobotDataset. Default: 'h264' (≈10× faster than 'libsvtav1'). "
            "Try 'h264_nvenc' for GPU-accelerated encode if FFmpeg has NVENC."
        ),
    )
    p.add_argument(
        "--hwaccel", default=None, choices=[None, "cuda"],
        help="HEVC decode acceleration. 'cuda' uses NVDEC (hevc_cuvid) when available.",
    )
    p.add_argument(
        "--num-workers", type=int, default=1,
        help=(
            "Per-task mode only: number of tasks to convert in parallel "
            "(each as a separate process). Ignored in merged mode (single writer). "
            "Default: 1."
        ),
    )
    p.add_argument(
        "--no-streaming-encoding", action="store_true",
        help="Disable streaming video encoding (slower, but simpler error model).",
    )
    p.add_argument(
        "--image-writer-threads", type=int, default=4,
        help="Threads used by LeRobotDataset for image/PNG writes. Default: 4.",
    )
    p.add_argument(
        "--image-writer-processes", type=int, default=0,
        help="Subprocesses for image writes (0 = use threads only). Default: 0.",
    )
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
        dst_resolved = args.dst.resolve()
        task_dirs = sorted(
            d for d in src.iterdir()
            if d.is_dir()
            and d.resolve() != dst_resolved
            and not d.name.startswith(".")
            and d.name != "lerobot"
            and d.name != args.merged_name
        )

    if not task_dirs:
        logger.error("No task dirs found under %s", src)
        return 2

    image_size: tuple[int, int] | None = (
        None if args.image_size == 0 else (args.image_size, args.image_size)
    )

    streaming_encoding = not args.no_streaming_encoding

    if args.mode == "merged":
        if args.num_workers > 1:
            logger.warning(
                "--num-workers=%d ignored in merged mode (one writer only); "
                "in-episode threading still applies.", args.num_workers,
            )
        repo_id = f"{args.hf_user}/{args.merged_name}"
        try:
            convert_merged(
                task_dirs=task_dirs,
                dst_root=args.dst,
                merged_name=args.merged_name,
                repo_id=repo_id,
                fps=args.fps,
                image_size=image_size,
                vcodec=args.vcodec,
                skip_existing=args.skip_existing,
                streaming_encoding=streaming_encoding,
                image_writer_threads=args.image_writer_threads,
                image_writer_processes=args.image_writer_processes,
                hwaccel=args.hwaccel,
                log_level=args.log_level,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("[merged] FAILED: %s", e)
            logger.debug("%s", traceback.format_exc())
        logger.info("All done. Merged dataset root: %s", args.dst / args.merged_name)
        return 0

    # ── per-task mode ──
    common_kwargs = dict(
        dst_root=args.dst,
        fps=args.fps,
        image_size=image_size,
        vcodec=args.vcodec,
        skip_existing=args.skip_existing,
        streaming_encoding=streaming_encoding,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        hwaccel=args.hwaccel,
        log_level=args.log_level,
    )

    jobs = []
    for td in task_dirs:
        if not td.is_dir():
            logger.warning("[skip] %s — not a directory", td)
            continue
        jobs.append((td, f"{args.hf_user}/{td.name}"))

    def _run(td: Path, repo_id: str) -> tuple[str, int, int]:
        return convert_task(task_dir=td, repo_id=repo_id, **common_kwargs)

    if args.num_workers <= 1:
        for td, repo_id in jobs:
            try:
                _run(td, repo_id)
            except Exception as e:  # noqa: BLE001
                logger.error("[task=%s] FAILED: %s", td.name, e)
                logger.debug("%s", traceback.format_exc())
    else:
        ctx = mp.get_context("spawn")
        worker = partial(_convert_task_entrypoint, common_kwargs=common_kwargs)
        with ctx.Pool(processes=args.num_workers) as pool:
            for res in pool.imap_unordered(worker, jobs):
                if isinstance(res, tuple) and len(res) == 3:
                    name, ok, total = res
                    logger.info("[task=%s] reported %d/%d episodes converted", name, ok, total)
                else:
                    logger.error("worker returned unexpected: %r", res)

    logger.info("All done. Per-task output root: %s", args.dst)
    return 0


def _convert_task_entrypoint(job: tuple[Path, str], common_kwargs: dict) -> tuple[str, int, int]:
    """Top-level helper so ``mp.Pool`` (spawn context) can pickle it."""
    td, repo_id = job
    try:
        return convert_task(task_dir=td, repo_id=repo_id, **common_kwargs)
    except Exception as e:  # noqa: BLE001
        logging.getLogger("port_mcap").error("[task=%s] FAILED in worker: %s", td.name, e)
        logging.getLogger("port_mcap").debug("%s", traceback.format_exc())
        return td.name, 0, -1


if __name__ == "__main__":
    sys.exit(main())
