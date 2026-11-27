#!/usr/bin/env python3
"""valorant-api.com から競技 13 マップの displayIcon 透過アルファを取得し、
オフライン同梱パック `vreplay/data/map_alphas.npz` を生成する。

各マップの 512×512 2 値アルファマスク（alpha > 128）を `np.packbits(..., axis=1)`
で `(512, 64)` uint8 に圧縮して保存する。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

API_URL = "https://valorant-api.com/v1/maps"
COMPETITIVE_MAPS: tuple[str, ...] = (
    "abyss",
    "ascent",
    "bind",
    "breeze",
    "corrode",
    "fracture",
    "haven",
    "icebox",
    "lotus",
    "pearl",
    "split",
    "summit",
    "sunset",
)
DEFAULT_SIZE = 512
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "vreplay" / "data" / "map_alphas.npz"


def _fetch_bytes(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "vreplay-build-map-alphas/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _alpha_from_png_bytes(data: bytes, size: int = DEFAULT_SIZE) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[:, :, 3]
    if alpha.shape != (size, size):
        alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)
    bits = (alpha > 128).astype(np.uint8)
    return np.packbits(bits, axis=1)


def build_map_alphas(
    out_path: Path = DEFAULT_OUT,
    *,
    maps: tuple[str, ...] = COMPETITIVE_MAPS,
    size: int = DEFAULT_SIZE,
    timeout: float = 30.0,
) -> dict[str, np.ndarray]:
    if size <= 0 or size % 8 != 0:
        raise ValueError(f"size must be a positive multiple of 8, got {size}")
    payload = json.loads(_fetch_bytes(API_URL, timeout=timeout).decode("utf-8"))
    entries = payload.get("data") or []
    by_name: dict[str, dict] = {}
    for item in entries:
        name = str(item.get("displayName") or "").strip().lower()
        icon = item.get("displayIcon")
        if name and icon:
            by_name[name] = item

    packed: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for m in maps:
        key = m.strip().lower()
        item = by_name.get(key)
        if not item:
            missing.append(key)
            continue
        png_bytes = _fetch_bytes(item["displayIcon"], timeout=timeout)
        packed[key] = _alpha_from_png_bytes(png_bytes, size=size)
        unpacked = np.unpackbits(packed[key], axis=1)
        print(f"  {key:10s}: shape={unpacked.shape}, floor_frac={unpacked.mean():.4f}")

    if missing:
        raise RuntimeError(f"Missing maps from valorant-api: {missing}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **packed)
    fsize = out_path.stat().st_size
    print(f"Saved {len(packed)} maps to {out_path} ({fsize:,} bytes, {fsize / 1024:.1f} KB)")
    return packed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build offline 13-map alpha mask pack (vreplay/data/map_alphas.npz)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output .npz path")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Square mask resolution (multiple of 8)")
    ap.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    args = ap.parse_args(argv)
    build_map_alphas(args.out, size=args.size, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
