#!/usr/bin/env python3
"""Back-project a perspective image onto an ERP panorama.

Inverse of erp2pers.py: takes a pinhole image plus its projection parameters
(fov / yaw / pitch / optional roll) and pastes it back into equirectangular
space. Parameters are parsed from the filename when it follows the erp2pers
naming scheme, e.g.

    pers_20260717_164006_fov90_yaw146.4_pitch-12.3.png

Run:
    python pers2erp.py PERS.png --erp PANO.png            # composite onto pano
    python pers2erp.py PERS.png --erp-size 1024 2048      # blank canvas + mask
    python pers2erp.py PERS.png --erp PANO.png --fov 85   # override parsed fov

The output ERP (and optionally the coverage mask) is written next to the
input unless --out is given.
"""

from __future__ import annotations

import argparse
import os
import re

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Core back-projection (conventions identical to erp2pers.py)
# --------------------------------------------------------------------------- #
def _rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """R = Ry(yaw) @ Rx(pitch) @ Rz(roll); +yaw looks right, +pitch looks up."""
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def _erp_rays(erp_hw: tuple[int, int]) -> np.ndarray:
    """Unit world ray (x right, y up, z forward) for each ERP pixel centre."""
    eh, ew = erp_hw
    lon = ((np.arange(ew, dtype=np.float64) + 0.5) / ew - 0.5) * 2.0 * np.pi
    lat = (0.5 - (np.arange(eh, dtype=np.float64) + 0.5) / eh) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
        np.cos(lat) * np.cos(lon),
    ], axis=-1)


def perspective_to_erp(
    pers: np.ndarray,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float = 0.0,
    erp_hw: tuple[int, int] = (1024, 2048),
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a pinhole image into ERP space.

    Returns (warped, mask): `warped` is an ERP-sized image that is zero outside
    the view frustum, `mask` is a float32 coverage map in [0, 1] (soft only on
    the 1-px bilinear border).
    """
    h, w = pers.shape[:2]
    f = 0.5 * w / np.tan(np.deg2rad(fov_deg) / 2.0)

    # World ray -> camera frame (rows @ R == R.T @ ray), then z=1 plane.
    cam = _erp_rays(erp_hw) @ _rotation(yaw_deg, pitch_deg, roll_deg)
    x, y, z = cam[..., 0], cam[..., 1], cam[..., 2]
    in_front = z > 1e-9
    zs = np.where(in_front, z, 1.0)
    map_x = np.where(in_front, (x / zs) * f + w / 2.0 - 0.5, -1e6)
    map_y = np.where(in_front, (-y / zs) * f + h / 2.0 - 0.5, -1e6)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    warped = cv2.remap(
        pers, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    mask = cv2.remap(
        np.ones((h, w), np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped, mask


def composite_onto_erp(base: np.ndarray, warped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Alpha-blend the warped view over the base panorama."""
    m = mask[..., None].astype(np.float32)
    out = warped.astype(np.float32) * m + base.astype(np.float32) * (1.0 - m)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Filename parsing / CLI
# --------------------------------------------------------------------------- #
def params_from_name(path: str) -> dict[str, float]:
    """Extract fov/yaw/pitch(/roll) from an erp2pers-style filename."""
    stem = os.path.splitext(os.path.basename(path))[0]
    found = {}
    for key in ("fov", "yaw", "pitch", "roll"):
        m = re.search(rf"{key}(-?\d+(?:\.\d+)?)", stem)
        if m:
            found[key] = float(m.group(1))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Perspective -> ERP back-projection")
    parser.add_argument("pers", help="Perspective image (params parsed from its name)")
    parser.add_argument("--erp", help="Base ERP panorama to composite onto")
    parser.add_argument("--erp-size", type=int, nargs=2, metavar=("H", "W"),
                        help="ERP size if no base panorama is given")
    parser.add_argument("--fov", type=float, help="Horizontal FOV in degrees (overrides name)")
    parser.add_argument("--yaw", type=float, help="Yaw in degrees (overrides name)")
    parser.add_argument("--pitch", type=float, help="Pitch in degrees (overrides name)")
    parser.add_argument("--roll", type=float, help="Roll in degrees (default 0)")
    parser.add_argument("--out", help="Output path (default: erp_from_<name>.png next to input)")
    parser.add_argument("--save-mask", action="store_true", help="Also write the coverage mask")
    args = parser.parse_args()

    parsed = params_from_name(args.pers)
    p = {k: getattr(args, k) if getattr(args, k) is not None else parsed.get(k)
         for k in ("fov", "yaw", "pitch", "roll")}
    p["roll"] = p["roll"] or 0.0
    missing = [k for k in ("fov", "yaw", "pitch") if p[k] is None]
    if missing:
        parser.error(f"could not parse {missing} from filename; pass --{missing[0]} etc.")

    pers = cv2.imread(args.pers, cv2.IMREAD_COLOR)
    if pers is None:
        parser.error(f"cannot read {args.pers}")

    base = None
    if args.erp:
        base = cv2.imread(args.erp, cv2.IMREAD_COLOR)
        if base is None:
            parser.error(f"cannot read {args.erp}")
        erp_hw = base.shape[:2]
    elif args.erp_size:
        erp_hw = tuple(args.erp_size)
    else:
        parser.error("need --erp or --erp-size")

    print(f"fov={p['fov']:g}  yaw={p['yaw']:g}  pitch={p['pitch']:g}  roll={p['roll']:g}  "
          f"pers={pers.shape[1]}x{pers.shape[0]}  erp={erp_hw[1]}x{erp_hw[0]}")

    warped, mask = perspective_to_erp(pers, p["fov"], p["yaw"], p["pitch"], p["roll"], erp_hw)
    out = composite_onto_erp(base, warped, mask) if base is not None else warped

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.pers)),
        f"erp_from_{os.path.splitext(os.path.basename(args.pers))[0]}.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)
    print(f"saved {out_path}")
    if args.save_mask:
        mask_path = os.path.splitext(out_path)[0] + "_mask.png"
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        print(f"saved {mask_path}")


if __name__ == "__main__":
    main()
