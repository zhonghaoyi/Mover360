#!/usr/bin/env python3
"""Interactive ERP -> perspective projection UI.

Upload an equirectangular panorama, click on it to aim the virtual camera
(the click sets yaw/pitch), pick a FOV and output size, preview the
perspective render live, then save the result to disk.

Run:
    python erp2pers.py [--port 7860] [--host 127.0.0.1] [--share]

The UI is served over HTTP so it works on headless machines (open the
forwarded port in your browser).
"""

from __future__ import annotations

import argparse
import datetime
import os

# /tmp/gradio on this shared machine belongs to another user; keep temp files local.
os.environ.setdefault("GRADIO_TEMP_DIR", "/vol/graphics-solar/zhonghaoy/.gradio_tmp")

import cv2
import gradio as gr
import numpy as np

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "erp2pers")
MAX_DISPLAY_WIDTH = 2048  # ERP shown in the browser is capped to this width


# --------------------------------------------------------------------------- #
# Core projection
# --------------------------------------------------------------------------- #
def _camera_rays(fov_deg: float, out_hw: tuple[int, int]) -> np.ndarray:
    """Unit-less ray directions (x right, y up, z forward) for each out pixel."""
    h, w = out_hw
    f = 0.5 * w / np.tan(np.deg2rad(fov_deg) / 2.0)
    u = (np.arange(w, dtype=np.float64) + 0.5) - w / 2.0
    v = (np.arange(h, dtype=np.float64) + 0.5) - h / 2.0
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu / f, -vv / f, np.ones_like(uu)], axis=-1)


def _rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """R = Ry(yaw) @ Rx(pitch) @ Rz(roll); +yaw looks right, +pitch looks up."""
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def _rays_to_erp_xy(rays: np.ndarray, erp_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Map world rays to (col, row) pixel coordinates in the ERP image."""
    eh, ew = erp_hw
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    lon = np.arctan2(x, z)                      # [-pi, pi], 0 = image centre
    lat = np.arctan2(y, np.hypot(x, z))         # [-pi/2, pi/2], + = up
    map_x = (lon / (2 * np.pi) + 0.5) * ew - 0.5
    map_y = (0.5 - lat / np.pi) * eh - 0.5
    return map_x, map_y


def erp_to_perspective(
    erp: np.ndarray,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float = 0.0,
    out_hw: tuple[int, int] = (720, 1280),
) -> np.ndarray:
    """Render a pinhole view (horizontal FOV `fov_deg`) from an ERP panorama."""
    rays = _camera_rays(fov_deg, out_hw) @ _rotation(yaw_deg, pitch_deg, roll_deg).T
    map_x, map_y = _rays_to_erp_xy(rays, erp.shape[:2])

    # Pad one wrapped column on each side so bilinear sampling blends across
    # the longitude seam instead of clamping at it.
    erp_pad = cv2.copyMakeBorder(erp, 0, 0, 1, 1, cv2.BORDER_WRAP)
    map_x = np.mod(map_x, erp.shape[1]) + 1.0
    map_y = np.clip(map_y, 0, erp.shape[0] - 1)
    return cv2.remap(
        erp_pad,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def frustum_border_points(
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    out_hw: tuple[int, int],
    erp_hw: tuple[int, int],
    samples_per_edge: int = 60,
) -> np.ndarray:
    """ERP pixel coords of the perspective image border (for the overlay)."""
    h, w = out_hw
    t = np.linspace(0, 1, samples_per_edge)
    edges = np.concatenate([
        np.stack([t * (w - 1), np.zeros_like(t)], axis=-1),          # top
        np.stack([np.full_like(t, w - 1), t * (h - 1)], axis=-1),    # right
        np.stack([(1 - t) * (w - 1), np.full_like(t, h - 1)], axis=-1),  # bottom
        np.stack([np.zeros_like(t), (1 - t) * (h - 1)], axis=-1),    # left
    ])
    f = 0.5 * w / np.tan(np.deg2rad(fov_deg) / 2.0)
    rays = np.stack([
        (edges[:, 0] + 0.5 - w / 2.0) / f,
        -(edges[:, 1] + 0.5 - h / 2.0) / f,
        np.ones(len(edges)),
    ], axis=-1) @ _rotation(yaw_deg, pitch_deg, roll_deg).T
    map_x, map_y = _rays_to_erp_xy(rays, erp_hw)
    return np.stack([np.mod(map_x, erp_hw[1]), np.clip(map_y, 0, erp_hw[0] - 1)], axis=-1)


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def _annotate_erp(erp_disp: np.ndarray, fov, yaw, pitch, roll, out_hw) -> np.ndarray:
    """Draw the view-frustum outline and centre crosshair on the display ERP."""
    img = erp_disp.copy()
    eh, ew = img.shape[:2]
    pts = frustum_border_points(fov, yaw, pitch, roll, out_hw, (eh, ew))
    # Dots instead of a polyline: immune to the horizontal wrap-around jump.
    for px, py in pts:
        cv2.circle(img, (int(round(px)), int(round(py))), max(2, ew // 1000), (0, 255, 90), -1)
    cx = int(round(np.mod(yaw / 360.0 + 0.5, 1.0) * ew))
    cy = int(round((0.5 - pitch / 180.0) * eh))
    s = max(8, ew // 120)
    cv2.line(img, (cx - s, cy), (cx + s, cy), (255, 60, 60), 2)
    cv2.line(img, (cx, cy - s), (cx, cy + s), (255, 60, 60), 2)
    return img


def _render(erp_full, fov, yaw, pitch, roll, out_w, out_h):
    """Returns (annotated display ERP, perspective render)."""
    if erp_full is None:
        return None, None
    out_hw = (int(out_h), int(out_w))
    pers = erp_to_perspective(erp_full, fov, yaw, pitch, roll, out_hw)

    scale = min(1.0, MAX_DISPLAY_WIDTH / erp_full.shape[1])
    if scale < 1.0:
        disp = cv2.resize(erp_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        disp = erp_full
    return _annotate_erp(disp, fov, yaw, pitch, roll, out_hw), pers


# --------------------------------------------------------------------------- #
# Gradio app
# --------------------------------------------------------------------------- #
def build_app() -> gr.Blocks:
    with gr.Blocks(title="ERP → Perspective") as demo:
        gr.Markdown(
            "## ERP → Perspective projection\n"
            "1. Upload an equirectangular panorama. "
            "2. **Click on it** to aim the camera (sets yaw/pitch). "
            "3. Tune FOV / size / roll. 4. Save the result."
        )
        erp_state = gr.State(None)   # full-resolution ERP (RGB uint8)
        pers_state = gr.State(None)  # last rendered perspective image

        with gr.Row():
            with gr.Column(scale=3):
                erp_in = gr.Image(
                    label="ERP panorama — click to set the view direction",
                    type="numpy", sources=["upload"], interactive=True,
                )
                pers_out = gr.Image(label="Perspective view", type="numpy", interactive=False)
            with gr.Column(scale=1):
                fov = gr.Slider(20, 150, value=90, step=1, label="Horizontal FOV (°)")
                yaw = gr.Slider(-180, 180, value=0, step=0.5, label="Yaw (°)")
                pitch = gr.Slider(-90, 90, value=0, step=0.5, label="Pitch (°)")
                roll = gr.Slider(-180, 180, value=0, step=0.5, label="Roll (°)")
                out_w = gr.Slider(128, 4096, value=1280, step=16, label="Output width (px)")
                out_h = gr.Slider(128, 4096, value=720, step=16, label="Output height (px)")
                out_dir = gr.Textbox(value=DEFAULT_OUTPUT_DIR, label="Output directory")
                fname = gr.Textbox(value="", label="Filename (blank = auto)")
                save_btn = gr.Button("💾 Save result", variant="primary")
                save_msg = gr.Markdown("")

        params = [fov, yaw, pitch, roll, out_w, out_h]

        def on_upload(img, *p):
            if img is None:
                return None, gr.update(), None, None
            erp = _to_rgb_uint8(img)
            disp, pers = _render(erp, *p)
            return erp, disp, pers, pers

        def on_change(erp, *p):
            disp, pers = _render(erp, *p)
            return disp, pers, pers

        def on_click(erp, evt: gr.SelectData, *p):
            if erp is None:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            scale = min(1.0, MAX_DISPLAY_WIDTH / erp.shape[1])
            disp_w, disp_h = erp.shape[1] * scale, erp.shape[0] * scale
            x, y = evt.index
            new_yaw = round((x / disp_w - 0.5) * 360.0, 1)
            new_pitch = round((0.5 - y / disp_h) * 180.0, 1)
            p = list(p)
            p[1], p[2] = new_yaw, new_pitch
            disp, pers = _render(erp, *p)
            return new_yaw, new_pitch, disp, pers, pers

        def on_save(pers, directory, name, *p):
            if pers is None:
                return "⚠️ Nothing to save — upload an ERP image first."
            directory = os.path.expanduser(directory.strip() or DEFAULT_OUTPUT_DIR)
            os.makedirs(directory, exist_ok=True)
            name = name.strip()
            if not name:
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"pers_{stamp}_fov{p[0]:g}_yaw{p[1]:g}_pitch{p[2]:g}.png"
            if not os.path.splitext(name)[1]:
                name += ".png"
            path = os.path.join(directory, name)
            cv2.imwrite(path, cv2.cvtColor(pers, cv2.COLOR_RGB2BGR))
            return f"✅ Saved `{path}`"

        erp_in.upload(on_upload, [erp_in] + params, [erp_state, erp_in, pers_out, pers_state])
        erp_in.clear(lambda: (None, None, None), None, [erp_state, pers_out, pers_state])
        erp_in.select(on_click, [erp_state] + params, [yaw, pitch, erp_in, pers_out, pers_state])
        for comp in params:
            comp.release(on_change, [erp_state] + params, [erp_in, pers_out, pers_state])
        save_btn.click(on_save, [pers_state, out_dir, fname] + params, [save_msg])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive ERP -> perspective UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public gradio link")
    args = parser.parse_args()
    build_app().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
