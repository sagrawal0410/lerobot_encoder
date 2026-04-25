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

"""Aggregate multiple per-task LeRobot datasets into ONE merged dataset.

Use this when ``port_mcap_workspace.py`` was run with ``--mode per-task`` and
you now want the SmolVLA-style merged base WITHOUT re-running the converter.
This wraps LeRobot's official ``aggregate_datasets()`` which:

  * Validates that every source dataset shares ``fps`` / ``robot_type`` / ``features``.
  * Unifies tasks (deduplication by language string → single task table).
  * Copies videos and data files (no re-encode — fast, just I/O).
  * Re-numbers ``episode_index`` / ``index`` / ``task_index`` / chunk indices.
  * Recomputes per-feature stats correctly across the union.

Usage:

    python examples/port_datasets/aggregate_per_task_datasets.py \\
        --src /workspace/lerobot \\
        --dst /workspace/lerobot \\
        --merged-name _combined \\
        --hf-user sagrawal0410

Auto-discovers all subdirs of ``--src`` that look like a LeRobot dataset
(must contain ``meta/info.json``). Skips ``--merged-name`` itself.

Output: ``<dst>/<merged-name>/{meta,data,videos}/...`` — point Stage 1 at it
exactly as you would the converter's native ``--mode merged`` output.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lerobot.datasets import aggregate_datasets

logger = logging.getLogger("aggregate_per_task")


def discover_task_dirs(src: Path, merged_name: str) -> list[Path]:
    """Return sorted list of subdirs of ``src`` that hold a LeRobot dataset."""
    out: list[Path] = []
    for d in sorted(src.iterdir()):
        if not d.is_dir():
            continue
        if d.name == merged_name or d.name.startswith("."):
            continue
        if (d / "meta" / "info.json").is_file():
            out.append(d)
        else:
            logger.debug("[skip] %s — no meta/info.json", d)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src", type=Path, default=Path("/workspace/lerobot"),
                   help="Root containing per-task LeRobot datasets (default: /workspace/lerobot).")
    p.add_argument("--dst", type=Path, default=None,
                   help="Output root for the merged dataset. Default: same as --src.")
    p.add_argument("--merged-name", default="_combined",
                   help="Subdir name of the aggregated dataset under --dst. Default: _combined.")
    p.add_argument("--hf-user", required=True,
                   help="HF Hub username — used to namespace the merged repo_id.")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Optional explicit list of task subdir names. Default: auto-discover.")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="Files per chunk dir in the aggregated dataset. Default: LeRobot default.")
    p.add_argument("--data-files-size-mb", type=float, default=None,
                   help="Max data parquet file size in MB. Default: LeRobot default.")
    p.add_argument("--video-files-size-mb", type=float, default=None,
                   help="Max video MP4 file size in MB. Default: LeRobot default.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    src: Path = args.src
    dst: Path = args.dst or args.src
    if not src.is_dir():
        logger.error("--src %s does not exist or is not a directory", src)
        return 2

    if args.tasks:
        roots = [src / t for t in args.tasks]
        bad = [r for r in roots if not (r / "meta" / "info.json").is_file()]
        if bad:
            logger.error("These --tasks dirs are not LeRobot datasets: %s", bad)
            return 2
    else:
        roots = discover_task_dirs(src, args.merged_name)

    if not roots:
        logger.error("No per-task LeRobot datasets found under %s", src)
        return 2

    repo_ids = [f"{args.hf_user}/{r.name}" for r in roots]
    aggr_root = dst / args.merged_name
    aggr_repo_id = f"{args.hf_user}/{args.merged_name}"

    if (aggr_root / "meta" / "info.json").is_file():
        logger.error(
            "Aggregated dataset already exists at %s. Delete it or pick a different "
            "--merged-name. (rm -rf %s)", aggr_root, aggr_root,
        )
        return 2

    logger.info("Aggregating %d per-task datasets → %s", len(roots), aggr_root)
    for r, rid in zip(roots, repo_ids):
        logger.info("  · %s   (repo_id=%s)", r, rid)

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=aggr_repo_id,
        roots=roots,
        aggr_root=aggr_root,
        data_files_size_in_mb=args.data_files_size_mb,
        video_files_size_in_mb=args.video_files_size_mb,
        chunk_size=args.chunk_size,
    )

    logger.info("Done. Merged dataset at %s", aggr_root)
    logger.info("Train Stage 1 with:")
    logger.info("  CUSTOM_DATASET_REPO_ID=%s", aggr_repo_id)
    logger.info("  CUSTOM_DATASET_ROOT=%s", aggr_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
