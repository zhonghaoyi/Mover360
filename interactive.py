#!/usr/bin/env python3
"""Lightweight browser UI for Mover360_depth inference.

The page lets a user upload an ERP panorama, draw a source box plus either a
target point or target box, optionally upload a reference image for add, then
runs DA-2 depth prediction followed by Mover360_depth.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import cgi


REPO_ROOT = Path(__file__).resolve().parent
DA2_DIR = REPO_ROOT / "depth" / "DA-2"
DEFAULT_CKPT = REPO_ROOT / "logs" / "360mover_flux2_depth" / "checkpoints" / "epoch=7-step=30219.ckpt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "interactive"
# Optional override for the Hugging Face cache; unset -> standard HF cache.
_hf_cache_env = os.environ.get("MOVER360_HF_CACHE")
DEFAULT_HF_CACHE = Path(_hf_cache_env) if _hf_cache_env else None
MAX_INTERACTIVE_RESULTS = 8
DEFAULT_AUTO_GPU_MAX_USED_MB = 128
DEFAULT_AUTO_GPU_MAX_UTIL = 20
POINT_GUIDANCE_RADIUS_PX = 0
POINT_GUIDANCE_SIGMA_PX = 10.0
POINT_GUIDANCE_REFERENCE_WIDTH = 2048


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mover360</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #18202a;
      --muted: #637083;
      --blue: #2563eb;
      --orange: #d97706;
      --red: #dc2626;
      --green: #15803d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    h1 { font-size: 18px; margin: 0; font-weight: 650; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 1fr) 360px;
      gap: 16px;
      padding: 16px;
      max-width: 1600px;
      margin: 0 auto;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .canvas-wrap {
      min-height: 420px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 12px;
      background: #eef1f5;
    }
    canvas {
      max-width: 100%;
      max-height: calc(100vh - 120px);
      background: #111827;
      border: 1px solid #c7ceda;
      cursor: crosshair;
    }
    .side {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .section {
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    .side .panel .section:last-child { border-bottom: 0; }
    label {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    input, select, button {
      width: 100%;
      font: inherit;
    }
    input[type="number"], input[type="text"], select {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 8px 9px;
      color: var(--text);
    }
    .hidden-file {
      display: none;
    }
    .file-control {
      display: grid;
      grid-template-columns: minmax(92px, 112px) minmax(0, 1fr);
      gap: 8px;
      align-items: stretch;
    }
    .file-name {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--muted);
      padding: 8px 9px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .row.triple { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .tools { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 9px 10px;
      cursor: pointer;
      font-weight: 600;
    }
    button.active[data-tool="source"] { border-color: var(--blue); color: var(--blue); }
    button.active[data-tool="targetBox"] { border-color: var(--orange); color: var(--orange); }
    button.active[data-tool="targetPoint"] { border-color: var(--red); color: var(--red); }
    button.primary {
      border-color: #0f172a;
      background: #0f172a;
      color: #fff;
    }
    button.secondary { color: var(--muted); }
    button:disabled { opacity: .55; cursor: wait; }
    .status {
      min-height: 42px;
      padding: 10px 12px;
      border-radius: 6px;
      background: #f8fafc;
      border: 1px solid var(--line);
      color: var(--muted);
      white-space: pre-wrap;
      line-height: 1.35;
    }
    .status.ok { color: var(--green); }
    .status.err { color: var(--red); }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 8px;
      color: var(--muted);
      background: #fff;
      font-size: 12px;
    }
    .dot { width: 9px; height: 9px; border-radius: 999px; display: inline-block; }
    .source-dot { background: var(--blue); }
    .target-dot { background: var(--orange); }
    .point-dot { background: var(--red); }
    .outputs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 12px;
    }
    .output h2 {
      font-size: 13px;
      margin: 0 0 8px;
      color: var(--muted);
      font-weight: 650;
    }
    .output img {
      width: 100%;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    .result-gallery {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
    }
    .result-gallery.single {
      grid-template-columns: 1fr;
    }
    .result-item {
      min-width: 0;
    }
    .result-caption {
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.3;
      overflow-wrap: anywhere;
    }
    .result-item img {
      cursor: pointer;
      transition: border-color .15s ease, box-shadow .15s ease;
    }
    .result-item.selected img {
      border-color: var(--blue);
      box-shadow: 0 0 0 2px rgba(37, 99, 235, .22);
    }
    .viewer-output {
      grid-column: 1 / -1;
    }
    .viewer-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(280px, .8fr);
      gap: 12px;
      align-items: start;
    }
    .viewer-image-wrap {
      position: relative;
      background: #0f172a;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    .viewer-image-wrap img {
      width: 100%;
      display: block;
      border: 0;
      border-radius: 0;
      cursor: crosshair;
      background: #111827;
    }
    .view-marker {
      position: absolute;
      width: 14px;
      height: 14px;
      border: 2px solid #fff;
      border-radius: 999px;
      box-shadow: 0 0 0 2px var(--red);
      transform: translate(-50%, -50%);
      pointer-events: none;
    }
    .perspective-panel {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .perspective-panel canvas {
      width: 100%;
      aspect-ratio: 1 / 1;
      max-height: none;
      cursor: default;
    }
    .viewer-controls {
      display: grid;
      gap: 8px;
    }
    .slider-row {
      display: grid;
      grid-template-columns: 52px minmax(0, 1fr) 64px;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }
    .slider-row input[type="range"] {
      width: 100%;
    }
    .viewer-actions {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .hidden { display: none; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .outputs { grid-template-columns: 1fr; }
      .viewer-grid { grid-template-columns: 1fr; }
      canvas { max-height: 70vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Mover360</h1>
    <div class="chips">
      <span class="chip"><span class="dot source-dot"></span>Source</span>
      <span class="chip"><span class="dot target-dot"></span>Target Box</span>
      <span class="chip"><span class="dot point-dot"></span>Target Point</span>
    </div>
  </header>

  <main>
    <section class="panel">
      <div class="canvas-wrap">
        <canvas id="canvas" width="1024" height="512"></canvas>
      </div>
      <div class="outputs" id="outputs">
        <div class="output hidden" id="resultBox">
          <h2>Prediction</h2>
          <div id="resultGallery" class="result-gallery"></div>
        </div>
        <div class="output viewer-output hidden" id="viewerBox">
          <h2>Selected Result and Perspective Preview</h2>
          <div class="viewer-grid">
            <div>
              <div class="viewer-image-wrap" id="viewerImageWrap">
                <img id="selectedImage" alt="Selected prediction">
                <span class="view-marker" id="viewMarker"></span>
              </div>
              <div class="result-caption" id="selectedCaption">Click a prediction to inspect it. Click the panorama to choose the perspective center.</div>
            </div>
            <div class="perspective-panel">
              <canvas id="perspectiveCanvas" width="512" height="512"></canvas>
              <div class="viewer-controls">
                <div class="slider-row">
                  <span>FOV</span>
                  <input id="fovSlider" type="range" min="20" max="120" value="75">
                  <span id="fovValue">75 deg</span>
                </div>
                <div class="slider-row">
                  <span>Yaw</span>
                  <input id="yawSlider" type="range" min="-180" max="180" value="0">
                  <span id="yawValue">0 deg</span>
                </div>
                <div class="slider-row">
                  <span>Pitch</span>
                  <input id="pitchSlider" type="range" min="-85" max="85" value="0">
                  <span id="pitchValue">0 deg</span>
                </div>
                <div class="viewer-actions">
                  <button type="button" class="secondary" id="resetViewBtn">Reset View</button>
                  <button type="button" class="secondary" id="useAsInputBtn">Use as Input</button>
                  <button type="button" class="primary" id="saveSelectionBtn">Save Selected</button>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="output hidden" id="depthBox">
          <h2>Near-Focused Depth</h2>
          <img id="depthImage" alt="Near-focused depth visualization">
        </div>
        <div class="output hidden" id="maskBox">
          <h2>Guidance</h2>
          <img id="maskImage" alt="Guidance mask">
        </div>
        <div class="output hidden" id="refBox">
          <h2>Reference</h2>
          <img id="refImage" alt="Reference image">
        </div>
      </div>
    </section>

    <aside class="side">
      <section class="panel">
        <div class="section">
          <label>Panorama</label>
          <input id="imageInput" class="hidden-file" type="file" accept="image/*">
          <div class="file-control">
            <button type="button" class="secondary" id="imageBrowse">Browse</button>
            <div class="file-name" id="imageName" title="No file selected">No file selected</div>
          </div>
        </div>
        <div class="section">
          <label>Reference</label>
          <input id="referenceInput" class="hidden-file" type="file" accept="image/*">
          <div class="file-control">
            <button type="button" class="secondary" id="referenceBrowse">Browse</button>
            <div class="file-name" id="referenceName" title="No file selected">No file selected</div>
          </div>
        </div>
        <div class="section">
          <label>Draw</label>
          <div class="tools">
            <button type="button" data-tool="source" class="active">Source</button>
            <button type="button" data-tool="targetBox">Target</button>
            <button type="button" data-tool="targetPoint">Point</button>
          </div>
        </div>
        <div class="section row">
          <button type="button" class="secondary" id="clearActive">Clear Tool</button>
          <button type="button" class="secondary" id="clearAll">Clear All</button>
        </div>
      </section>

      <section class="panel">
        <div class="section row">
          <div>
            <label for="task">Task</label>
            <select id="task">
              <option value="move">Move</option>
              <option value="add">Add</option>
              <option value="remove">Remove</option>
            </select>
          </div>
          <div>
            <label for="guidance">Guidance</label>
            <select id="guidance">
              <option value="auto">Auto</option>
              <option value="point">Point</option>
              <option value="bbox">Box</option>
              <option value="both">Both</option>
              <option value="mask">Mask</option>
            </select>
          </div>
        </div>
        <div class="section row triple">
          <div>
            <label for="steps">Steps</label>
            <input id="steps" type="number" min="1" max="100" value="50">
          </div>
          <div>
            <label for="seed">Seed</label>
            <input id="seed" type="number" value="-1">
          </div>
          <div>
            <label for="sampleCount">Results</label>
            <input id="sampleCount" type="number" min="1" max="8" value="1">
          </div>
        </div>
        <div class="section row">
          <button type="button" class="primary" id="runBtn">Run</button>
          <button type="button" class="secondary" id="clearResultsBtn">Clear Results</button>
        </div>
      </section>

      <section class="panel">
        <div class="section">
          <label>Status</label>
          <div class="status" id="status">Ready</div>
        </div>
      </section>
    </aside>
  </main>

  <script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const imageInput = document.getElementById('imageInput');
    const referenceInput = document.getElementById('referenceInput');
    const imageBrowse = document.getElementById('imageBrowse');
    const referenceBrowse = document.getElementById('referenceBrowse');
    const imageName = document.getElementById('imageName');
    const referenceName = document.getElementById('referenceName');
    const statusEl = document.getElementById('status');
    const runBtn = document.getElementById('runBtn');
    const clearResultsBtn = document.getElementById('clearResultsBtn');
    const toolButtons = Array.from(document.querySelectorAll('[data-tool]'));
    const resultGallery = document.getElementById('resultGallery');
    const depthImage = document.getElementById('depthImage');
    const maskImage = document.getElementById('maskImage');
    const refImage = document.getElementById('refImage');
    const resultBox = document.getElementById('resultBox');
    const depthBox = document.getElementById('depthBox');
    const maskBox = document.getElementById('maskBox');
    const refBox = document.getElementById('refBox');
    const viewerBox = document.getElementById('viewerBox');
    const viewerImageWrap = document.getElementById('viewerImageWrap');
    const selectedImage = document.getElementById('selectedImage');
    const selectedCaption = document.getElementById('selectedCaption');
    const viewMarker = document.getElementById('viewMarker');
    const perspectiveCanvas = document.getElementById('perspectiveCanvas');
    const perspectiveCtx = perspectiveCanvas.getContext('2d');
    const fovSlider = document.getElementById('fovSlider');
    const yawSlider = document.getElementById('yawSlider');
    const pitchSlider = document.getElementById('pitchSlider');
    const fovValue = document.getElementById('fovValue');
    const yawValue = document.getElementById('yawValue');
    const pitchValue = document.getElementById('pitchValue');
    const resetViewBtn = document.getElementById('resetViewBtn');
    const useAsInputBtn = document.getElementById('useAsInputBtn');
    const saveSelectionBtn = document.getElementById('saveSelectionBtn');

    let img = null;
    let currentPanoramaFile = null;
    let activeTool = 'source';
    let isDrawing = false;
    let dragStart = null;
    let previewBox = null;
    let annotations = {
      sourceBox: null,
      targetBox: null,
      targetPoint: null
    };
    let resultItems = [];
    let selectedResult = null;
    let selectedResultImage = null;
    let perspectiveRenderPending = false;
    let viewState = { yaw: 0, pitch: 0, fov: 75 };

    function setStatus(text, kind) {
      statusEl.textContent = text;
      statusEl.className = 'status' + (kind ? ' ' + kind : '');
    }

    function setActiveTool(tool) {
      activeTool = tool;
      toolButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.tool === tool));
    }

    function canvasPoint(evt) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(canvas.width, (evt.clientX - rect.left) * canvas.width / rect.width)),
        y: Math.max(0, Math.min(canvas.height, (evt.clientY - rect.top) * canvas.height / rect.height))
      };
    }

    function normalizeBox(a, b) {
      const x1 = Math.min(a.x, b.x);
      const y1 = Math.min(a.y, b.y);
      const x2 = Math.max(a.x, b.x);
      const y2 = Math.max(a.y, b.y);
      return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
    }

    function drawBox(box, color, label) {
      if (!box) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, canvas.width / 640);
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.16;
      ctx.fillRect(box.x, box.y, box.w, box.h);
      ctx.globalAlpha = 1;
      ctx.font = `${Math.max(14, canvas.width / 80)}px sans-serif`;
      const metrics = ctx.measureText(label);
      const labelH = Math.max(20, canvas.width / 48);
      ctx.fillRect(box.x, Math.max(0, box.y - labelH), metrics.width + 12, labelH);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, box.x + 6, Math.max(14, box.y - 6));
      ctx.restore();
    }

    function drawPoint(point) {
      if (!point) return;
      ctx.save();
      const r = Math.max(7, canvas.width / 140);
      ctx.fillStyle = '#dc2626';
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = Math.max(2, canvas.width / 640);
      ctx.beginPath();
      ctx.arc(point.x, point.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(point.x - r * 1.8, point.y);
      ctx.lineTo(point.x + r * 1.8, point.y);
      ctx.moveTo(point.x, point.y - r * 1.8);
      ctx.lineTo(point.x, point.y + r * 1.8);
      ctx.stroke();
      ctx.restore();
    }

    function render() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (img) {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      } else {
        ctx.fillStyle = '#111827';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#d1d5db';
        ctx.font = '18px sans-serif';
        ctx.fillText('Upload a panorama', 24, 42);
      }
      drawBox(annotations.sourceBox, '#2563eb', 'Source');
      drawBox(annotations.targetBox, '#d97706', 'Target');
      drawPoint(annotations.targetPoint);
      if (previewBox) {
        drawBox(previewBox, activeTool === 'source' ? '#2563eb' : '#d97706', 'Preview');
      }
    }

    function loadFileToCanvas(file, statusText) {
      const reader = new FileReader();
      reader.onload = () => {
        const nextImg = new Image();
        nextImg.onload = () => {
          img = nextImg;
          currentPanoramaFile = file;
          canvas.width = nextImg.naturalWidth;
          canvas.height = nextImg.naturalHeight;
          annotations = { sourceBox: null, targetBox: null, targetPoint: null };
          previewBox = null;
          render();
          setStatus(statusText || `${nextImg.naturalWidth} x ${nextImg.naturalHeight}`, 'ok');
        };
        nextImg.src = reader.result;
      };
      reader.readAsDataURL(file);
    }

    function setFileName(labelEl, file) {
      const name = file ? file.name : 'No file selected';
      labelEl.textContent = name;
      labelEl.title = name;
    }

    imageBrowse.addEventListener('click', () => imageInput.click());
    referenceBrowse.addEventListener('click', () => referenceInput.click());

    imageInput.addEventListener('change', () => {
      const file = imageInput.files[0];
      setFileName(imageName, file);
      if (file) loadFileToCanvas(file);
    });

    referenceInput.addEventListener('change', () => {
      setFileName(referenceName, referenceInput.files[0]);
    });

    toolButtons.forEach(btn => btn.addEventListener('click', () => setActiveTool(btn.dataset.tool)));

    canvas.addEventListener('mousedown', evt => {
      if (!img) return;
      const p = canvasPoint(evt);
      if (activeTool === 'targetPoint') {
        annotations.targetPoint = p;
        previewBox = null;
        render();
        return;
      }
      isDrawing = true;
      dragStart = p;
      previewBox = null;
    });

    canvas.addEventListener('mousemove', evt => {
      if (!isDrawing || !dragStart) return;
      previewBox = normalizeBox(dragStart, canvasPoint(evt));
      render();
    });

    window.addEventListener('mouseup', evt => {
      if (!isDrawing || !dragStart) return;
      const box = normalizeBox(dragStart, canvasPoint(evt));
      if (box.w >= 3 && box.h >= 3) {
        if (activeTool === 'source') annotations.sourceBox = box;
        if (activeTool === 'targetBox') annotations.targetBox = box;
      }
      isDrawing = false;
      dragStart = null;
      previewBox = null;
      render();
    });

    document.getElementById('clearActive').addEventListener('click', () => {
      if (activeTool === 'source') annotations.sourceBox = null;
      if (activeTool === 'targetBox') annotations.targetBox = null;
      if (activeTool === 'targetPoint') annotations.targetPoint = null;
      render();
    });

    document.getElementById('clearAll').addEventListener('click', () => {
      annotations = { sourceBox: null, targetBox: null, targetPoint: null };
      render();
    });

    function jsonOrEmpty(value) {
      return value ? JSON.stringify(value) : '';
    }

    function showImage(box, imgEl, url) {
      if (!url) {
        box.classList.add('hidden');
        return;
      }
      imgEl.src = url + '?t=' + Date.now();
      box.classList.remove('hidden');
    }

    function cleanOutputUrl(url) {
      return String(url || '').split('?')[0];
    }

    function predictionInputFileName(item, blob) {
      const seed = item.seed === undefined ? 'selected' : `seed-${item.seed}`;
      const path = cleanOutputUrl(item.url).split('/').pop() || '';
      const urlExt = (path.match(/\.(png|jpe?g|webp)$/i) || [])[0];
      const typeExt = blob.type === 'image/jpeg' ? '.jpg'
        : blob.type === 'image/webp' ? '.webp'
          : '.png';
      return `prediction-${seed}${urlExt || typeExt}`;
    }

    function clearPredictResults() {
      resultGallery.textContent = '';
      resultGallery.classList.remove('single');
      resultItems = [];
      selectedResult = null;
      selectedResultImage = null;
      resultBox.classList.add('hidden');
      viewerBox.classList.add('hidden');
      depthBox.classList.add('hidden');
      maskBox.classList.add('hidden');
      refBox.classList.add('hidden');
      selectedImage.removeAttribute('src');
      depthImage.removeAttribute('src');
      maskImage.removeAttribute('src');
      refImage.removeAttribute('src');
      viewMarker.style.display = 'none';
      selectedCaption.textContent = 'Click a prediction to inspect it. Click the panorama to choose the perspective center.';
      perspectiveCtx.clearRect(0, 0, perspectiveCanvas.width, perspectiveCanvas.height);
      setStatus('Ready', '');
    }

    function setViewState(next) {
      viewState = {
        yaw: Number(next.yaw),
        pitch: Number(next.pitch),
        fov: Number(next.fov)
      };
      viewState.yaw = Math.max(-180, Math.min(180, viewState.yaw));
      viewState.pitch = Math.max(-85, Math.min(85, viewState.pitch));
      viewState.fov = Math.max(20, Math.min(120, viewState.fov));
      yawSlider.value = String(Math.round(viewState.yaw));
      pitchSlider.value = String(Math.round(viewState.pitch));
      fovSlider.value = String(Math.round(viewState.fov));
      yawValue.textContent = `${Math.round(viewState.yaw)} deg`;
      pitchValue.textContent = `${Math.round(viewState.pitch)} deg`;
      fovValue.textContent = `${Math.round(viewState.fov)} deg`;
      updateViewMarker();
      schedulePerspectiveRender();
    }

    function updateViewMarker() {
      if (!selectedResultImage || !selectedResultImage.naturalWidth) {
        viewMarker.style.display = 'none';
        return;
      }
      const xPct = ((viewState.yaw + 180) / 360) * 100;
      const yPct = ((90 - viewState.pitch) / 180) * 100;
      viewMarker.style.left = `${xPct}%`;
      viewMarker.style.top = `${Math.max(0, Math.min(100, yPct))}%`;
      viewMarker.style.display = 'block';
    }

    function selectResult(item, index) {
      selectedResult = item;
      selectedCaption.textContent = item.seed === undefined
        ? 'Selected prediction'
        : `Selected prediction: seed ${item.seed}`;
      resultItems.forEach((entry, idx) => {
        entry.element.classList.toggle('selected', idx === index);
      });

      const resultImage = new Image();
      selectedResultImage = resultImage;
      resultImage.onload = () => {
        if (selectedResultImage !== resultImage) return;
        selectedImage.src = resultImage.src;
        viewerBox.classList.remove('hidden');
        setViewState(viewState);
      };
      resultImage.src = cleanOutputUrl(item.url) + '?t=' + Date.now();
    }

    function schedulePerspectiveRender() {
      if (!selectedResultImage || !selectedResultImage.complete || !selectedResultImage.naturalWidth) return;
      if (perspectiveRenderPending) return;
      perspectiveRenderPending = true;
      requestAnimationFrame(() => {
        perspectiveRenderPending = false;
        renderPerspective();
      });
    }

    function renderPerspective() {
      const source = selectedResultImage;
      if (!source || !source.naturalWidth) return;
      const size = perspectiveCanvas.width;
      const sourceCanvas = document.createElement('canvas');
      sourceCanvas.width = source.naturalWidth;
      sourceCanvas.height = source.naturalHeight;
      const sourceCtx = sourceCanvas.getContext('2d');
      sourceCtx.drawImage(source, 0, 0);
      const sourceData = sourceCtx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
      const out = perspectiveCtx.createImageData(size, size);
      const src = sourceData.data;
      const dst = out.data;
      const srcW = sourceCanvas.width;
      const srcH = sourceCanvas.height;
      const yaw = viewState.yaw * Math.PI / 180;
      const pitch = viewState.pitch * Math.PI / 180;
      const halfFovTan = Math.tan((viewState.fov * Math.PI / 180) / 2);
      const sinYaw = Math.sin(yaw);
      const cosYaw = Math.cos(yaw);
      const sinPitch = Math.sin(pitch);
      const cosPitch = Math.cos(pitch);
      const forward = [cosPitch * sinYaw, sinPitch, cosPitch * cosYaw];
      const right = [cosYaw, 0, -sinYaw];
      const up = [-sinPitch * sinYaw, cosPitch, -sinPitch * cosYaw];

      for (let y = 0; y < size; y += 1) {
        const v = (1 - 2 * ((y + 0.5) / size)) * halfFovTan;
        for (let x = 0; x < size; x += 1) {
          const u = (2 * ((x + 0.5) / size) - 1) * halfFovTan;
          let dx = forward[0] + u * right[0] + v * up[0];
          let dy = forward[1] + u * right[1] + v * up[1];
          let dz = forward[2] + u * right[2] + v * up[2];
          const invLen = 1 / Math.hypot(dx, dy, dz);
          dx *= invLen;
          dy *= invLen;
          dz *= invLen;
          const lon = Math.atan2(dx, dz);
          const lat = Math.asin(Math.max(-1, Math.min(1, dy)));
          let sx = ((lon + Math.PI) / (2 * Math.PI)) * srcW;
          let sy = ((Math.PI / 2 - lat) / Math.PI) * srcH;
          sx = ((Math.floor(sx) % srcW) + srcW) % srcW;
          sy = Math.max(0, Math.min(srcH - 1, Math.floor(sy)));
          const srcIdx = (sy * srcW + sx) * 4;
          const dstIdx = (y * size + x) * 4;
          dst[dstIdx] = src[srcIdx];
          dst[dstIdx + 1] = src[srcIdx + 1];
          dst[dstIdx + 2] = src[srcIdx + 2];
          dst[dstIdx + 3] = 255;
        }
      }
      perspectiveCtx.putImageData(out, 0, 0);
    }

    function showResults(results) {
      const items = Array.isArray(results) ? results.filter(item => item && item.url) : [];
      resultGallery.textContent = '';
      resultItems = [];
      selectedResult = null;
      selectedResultImage = null;
      viewerBox.classList.add('hidden');
      resultGallery.classList.toggle('single', items.length <= 1);
      if (!items.length) {
        resultBox.classList.add('hidden');
        return;
      }
      items.forEach((item, index) => {
        const wrap = document.createElement('div');
        wrap.className = 'result-item';
        const image = document.createElement('img');
        image.alt = item.seed === undefined ? 'Prediction' : `Prediction seed ${item.seed}`;
        image.src = item.url + '?t=' + Date.now();
        image.addEventListener('click', () => selectResult(item, index));
        const caption = document.createElement('div');
        caption.className = 'result-caption';
        caption.textContent = item.seed === undefined ? 'Prediction' : `Seed ${item.seed}`;
        wrap.appendChild(image);
        wrap.appendChild(caption);
        resultGallery.appendChild(wrap);
        resultItems.push({ element: wrap, item });
      });
      resultBox.classList.remove('hidden');
      selectResult(items[0], 0);
    }

    viewerImageWrap.addEventListener('click', evt => {
      if (!selectedResultImage || !selectedResultImage.naturalWidth) return;
      const rect = selectedImage.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width, evt.clientX - rect.left));
      const y = Math.max(0, Math.min(rect.height, evt.clientY - rect.top));
      const yaw = (x / rect.width) * 360 - 180;
      const pitch = 90 - (y / rect.height) * 180;
      setViewState({ ...viewState, yaw, pitch });
    });

    fovSlider.addEventListener('input', () => setViewState({ ...viewState, fov: Number(fovSlider.value) }));
    yawSlider.addEventListener('input', () => setViewState({ ...viewState, yaw: Number(yawSlider.value) }));
    pitchSlider.addEventListener('input', () => setViewState({ ...viewState, pitch: Number(pitchSlider.value) }));
    resetViewBtn.addEventListener('click', () => setViewState({ yaw: 0, pitch: 0, fov: 75 }));
    clearResultsBtn.addEventListener('click', clearPredictResults);

    useAsInputBtn.addEventListener('click', async () => {
      if (!selectedResult || !selectedResult.url) {
        setStatus('No selected prediction to use as input', 'err');
        return;
      }
      useAsInputBtn.disabled = true;
      setStatus('Loading selected prediction as new panorama input...', '');
      try {
        const response = await fetch(cleanOutputUrl(selectedResult.url));
        if (!response.ok) {
          throw new Error(response.statusText || 'Failed to load selected prediction');
        }
        const blob = await response.blob();
        const file = new File([blob], predictionInputFileName(selectedResult, blob), {
          type: blob.type || 'image/png'
        });
        imageInput.value = '';
        setFileName(imageName, file);
        loadFileToCanvas(file, 'Selected prediction is now the panorama input');
        clearPredictResults();
      } catch (err) {
        setStatus(err.message, 'err');
      } finally {
        useAsInputBtn.disabled = false;
      }
    });

    saveSelectionBtn.addEventListener('click', async () => {
      if (!selectedResult || !selectedResult.url || !selectedResultImage) {
        setStatus('No selected prediction to save', 'err');
        return;
      }
      saveSelectionBtn.disabled = true;
      setStatus('Saving selected prediction and perspective view...', '');
      try {
        renderPerspective();
        const payload = {
          image_url: cleanOutputUrl(selectedResult.url),
          perspective_png: perspectiveCanvas.toDataURL('image/png'),
          view: {
            yaw: viewState.yaw,
            pitch: viewState.pitch,
            fov: viewState.fov,
            square_size: perspectiveCanvas.width,
            seed: selectedResult.seed,
            device: selectedResult.device
          }
        };
        const response = await fetch('/api/save-selection', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.error || response.statusText);
        }
        setStatus(`Saved selected image and perspective view to:\n${data.saved_dir}`, 'ok');
      } catch (err) {
        setStatus(err.message, 'err');
      } finally {
        saveSelectionBtn.disabled = false;
      }
    });

    runBtn.addEventListener('click', async () => {
      if (!currentPanoramaFile) {
        setStatus('Missing panorama', 'err');
        return;
      }
      const form = new FormData();
      form.append('image', currentPanoramaFile);
      if (referenceInput.files[0]) form.append('reference', referenceInput.files[0]);
      form.append('task', document.getElementById('task').value);
      form.append('guidance_mode', document.getElementById('guidance').value);
      form.append('steps', document.getElementById('steps').value);
      form.append('seed', document.getElementById('seed').value);
      form.append('sample_count', document.getElementById('sampleCount').value);
      form.append('source_bbox', jsonOrEmpty(annotations.sourceBox));
      form.append('target_bbox', jsonOrEmpty(annotations.targetBox));
      form.append('target_point', jsonOrEmpty(annotations.targetPoint));
      form.append('image_width', String(canvas.width));
      form.append('image_height', String(canvas.height));

      runBtn.disabled = true;
      setStatus('Running DA-2 depth and Mover360_depth...', '');
      try {
        const response = await fetch('/api/infer', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.error || response.statusText);
        }
        showResults(data.results || (data.result_url ? [{ url: data.result_url, seed: data.seed }] : []));
        showImage(depthBox, depthImage, data.depth_vis_url);
        showImage(maskBox, maskImage, data.mask_vis_url);
        showImage(refBox, refImage, data.reference_url);
        setStatus(data.message || 'Done', 'ok');
      } catch (err) {
        setStatus(err.message, 'err');
      } finally {
        runBtn.disabled = false;
      }
    });

    render();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Mover360_depth interactive UI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get(
            "MOVER360_CUDA_VISIBLE_DEVICES",
            os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
        ),
        help=(
            "Physical CUDA devices exposed to the UI. Use `auto` to select mostly idle GPUs, "
            "`cpu` to hide CUDA, or an explicit list like `0,2,3`."
        ),
    )
    parser.add_argument(
        "--depth-cuda-visible-devices",
        default=os.environ.get("DEPTH_CUDA_VISIBLE_DEVICES", "auto"),
        help=(
            "CUDA_VISIBLE_DEVICES for the DA-2 subprocess. Use `auto` to choose a mostly-free "
            "GPU different from --cuda-visible-devices when possible; use `cpu` to hide CUDA."
        ),
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pano-height", type=int, default=512)
    parser.add_argument("--refs-resolution", type=int, default=384)
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--huggingface-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--compile-models", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--inference-devices",
        default=os.environ.get("MOVER360_INFERENCE_DEVICES", "auto"),
        help=(
            "Logical torch devices for Mover360 inference, e.g. `auto` or `cuda:0,cuda:1`. "
            "When --cuda-visible-devices is `2,3`, logical `cuda:0,cuda:1` map to physical `2,3`."
        ),
    )
    parser.add_argument(
        "--max-parallel-gpus",
        type=int,
        default=int(os.environ.get("MOVER360_MAX_PARALLEL_GPUS", "0") or 0),
        help="Limit the number of CUDA model replicas used for one request. 0 means all selected devices.",
    )
    parser.add_argument(
        "--auto-gpu-max-used-mb",
        type=int,
        default=int(os.environ.get("MOVER360_AUTO_GPU_MAX_USED_MB", DEFAULT_AUTO_GPU_MAX_USED_MB)),
        help="Maximum used GPU memory for `--cuda-visible-devices auto`.",
    )
    parser.add_argument(
        "--auto-gpu-max-util",
        type=int,
        default=int(os.environ.get("MOVER360_AUTO_GPU_MAX_UTIL", DEFAULT_AUTO_GPU_MAX_UTIL)),
        help="Maximum GPU utilization percent for `--cuda-visible-devices auto`.",
    )
    parser.add_argument("--depth-config", type=Path, default=DA2_DIR / "configs" / "infer.json")
    parser.add_argument("--depth-infer", type=Path, default=DA2_DIR / "infer_npy.py")
    parser.add_argument("--max-upload-mb", type=int, default=80)
    return parser.parse_args()


def normalize_prompt_text(prompt: str) -> str:
    prompt = prompt.strip()
    if prompt.endswith("."):
        prompt = prompt[:-1]
    return prompt


def ensure_prompt_sentence(prompt: str) -> str:
    prompt = prompt.strip()
    if prompt and prompt[-1] not in ".!?":
        prompt = prompt + "."
    return prompt


def default_task_prompt(task: str) -> str:
    if task == "add":
        return "add the object"
    if task == "remove":
        return "remove the object"
    return "move the object"


def build_flux2_text_only_prompt(prompt: str) -> str:
    return ensure_prompt_sentence(normalize_prompt_text(prompt))


def build_flux2_add_text_only_prompt(prompt: str) -> str:
    base_prompt = build_flux2_text_only_prompt(prompt)
    normalized = normalize_prompt_text(prompt).lower()
    explicit = "Add the object at the target region."
    if normalized in {"", "move the object", "move object", "add the object", "add object"}:
        return explicit
    return f"{explicit} {base_prompt}"


def build_flux2_move_text_only_prompt(prompt: str) -> str:
    base_prompt = build_flux2_text_only_prompt(prompt)
    normalized = normalize_prompt_text(prompt).lower()
    if "source" in normalized and "target" in normalized:
        return base_prompt
    explicit = "Move the object from the source region to the target region."
    if normalized in {"", "move the object", "move object"}:
        return explicit
    return f"{explicit} {base_prompt}"


def build_flux2_remove_text_only_prompt(prompt: str) -> str:
    base_prompt = build_flux2_text_only_prompt(prompt)
    normalized = normalize_prompt_text(prompt).lower()
    explicit = "Remove the object from the source region."
    if normalized in {"", "move the object", "move object", "remove the object", "remove object"}:
        return explicit
    return f"{explicit} {base_prompt}"


def build_prompt_variants(task: str) -> dict[str, list[str]]:
    prompt = default_task_prompt(task)
    if task == "add":
        text_only = build_flux2_add_text_only_prompt(prompt)
        input_prompt = f"Image 1 is the panorama to edit. {text_only}"
        mask_prompt = f"Image 1 is the panorama to edit. Image 2 marks the target region. {text_only}"
        ref_prompt = f"Image 1 is the panorama to edit. Image 2 is the object reference. {text_only}"
        ref_mask_prompt = (
            "Image 1 is the panorama to edit. "
            "Image 2 is the object reference. "
            f"Image 3 marks the target region. {text_only}"
        )
    elif task == "remove":
        text_only = build_flux2_remove_text_only_prompt(prompt)
        input_prompt = f"Image 1 is the panorama to edit. {text_only}"
        mask_prompt = f"Image 1 is the panorama to edit. Image 2 marks the object to remove. {text_only}"
        ref_prompt = input_prompt
        ref_mask_prompt = mask_prompt
    else:
        text_only = build_flux2_move_text_only_prompt(prompt)
        input_prompt = f"Image 1 is the panorama to edit. {text_only}"
        mask_prompt = (
            "Image 1 is the panorama to edit. "
            f"Image 2 marks the source and target regions. {text_only}"
        )
        ref_prompt = input_prompt
        ref_mask_prompt = mask_prompt

    return {
        "pano_prompt": [input_prompt],
        "pano_prompt_without_img": [text_only],
        "pano_prompt_with_mask": [mask_prompt],
        "pano_prompt_with_ref": [ref_prompt],
        "pano_prompt_with_ref_and_mask": [ref_mask_prompt],
    }


def safe_json_loads(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def clamp_box(box: dict[str, Any] | None, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not box:
        return None
    x = float(box.get("x", 0.0))
    y = float(box.get("y", 0.0))
    w = float(box.get("w", 0.0))
    h = float(box.get("h", 0.0))
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + w, 0.0), float(width))
    y2 = min(max(y + h, 0.0), float(height))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


def clamp_point(point: dict[str, Any] | None, width: int, height: int) -> tuple[float, float] | None:
    if not point:
        return None
    x = min(max(float(point.get("x", 0.0)), 0.0), float(max(width - 1, 0)))
    y = min(max(float(point.get("y", 0.0)), 0.0), float(max(height - 1, 0)))
    return x, y


def validate_annotations(
    task: str,
    source_box: tuple[float, float, float, float] | None,
    target_box: tuple[float, float, float, float] | None,
    target_point: tuple[float, float] | None,
) -> None:
    if task not in {"add", "move", "remove"}:
        raise ValueError("Task must be add, move, or remove.")
    if task in {"move", "remove"} and source_box is None:
        raise ValueError("Please draw a source box.")
    if task in {"add", "move"} and target_box is None and target_point is None:
        raise ValueError("Please draw a target box or target point.")


def resolve_guidance_mode(raw_mode: str, task: str, target_box: Any, target_point: Any) -> str:
    mode = str(raw_mode or "auto").strip().lower()
    if mode == "auto":
        if task == "remove":
            return "bbox"
        if target_point is not None:
            return "point"
        return "bbox"
    if mode not in {"point", "bbox", "both", "mask"}:
        raise ValueError("Guidance mode must be auto, point, bbox, both, or mask.")
    return mode


def image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def _parse_csv_tokens(value: str | None) -> list[str]:
    if value is None:
        return []
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _query_gpu_usage() -> list[tuple[int, int, int]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return []

    rows = []
    if completed.returncode != 0:
        return rows
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return rows


def resolve_cuda_visible_devices_arg(value: str | None, max_used_mb: int, max_util: int) -> str:
    configured = str(value or "").strip()
    normalized = configured.lower()
    if normalized in {"cpu", "none", "off", "-1"}:
        return ""
    if normalized not in {"", "auto"}:
        return configured

    rows = _query_gpu_usage()
    if not rows:
        return ""

    idle_rows = [
        row
        for row in rows
        if row[1] <= int(max_used_mb) and row[2] <= int(max_util)
    ]
    selected_rows = idle_rows or [min(rows, key=lambda row: (row[1], row[2], row[0]))]
    selected_rows = sorted(selected_rows, key=lambda row: row[0])
    return ",".join(str(row[0]) for row in selected_rows)


def _parse_cuda_device_list(value: str | None) -> set[str]:
    if value is None:
        return set()
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"cpu", "none", "off", "-1"}:
        return set()
    return {
        token.strip()
        for token in normalized.split(",")
        if token.strip() and token.strip() != "-1"
    }


class InteractiveRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = args.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.depth_cache_dir = self.output_dir / "_depth_cache"
        self.depth_cache_dir.mkdir(parents=True, exist_ok=True)
        self._models = {}
        self._inference_devices = None
        self._model_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def run(
        self,
        image_bytes: bytes,
        reference_bytes: bytes | None,
        fields: dict[str, str],
    ) -> dict[str, Any]:
        with self._infer_lock:
            return self._run_locked(image_bytes, reference_bytes, fields)

    def _run_locked(
        self,
        image_bytes: bytes,
        reference_bytes: bytes | None,
        fields: dict[str, str],
    ) -> dict[str, Any]:
        from PIL import Image
        import numpy as np
        import torch

        task = str(fields.get("task", "move")).strip().lower()
        orig_width = int(float(fields.get("image_width") or 0))
        orig_height = int(float(fields.get("image_height") or 0))
        if orig_width <= 0 or orig_height <= 0:
            from io import BytesIO

            with Image.open(BytesIO(image_bytes)) as probe_image:
                orig_width, orig_height = probe_image.size

        source_box = clamp_box(safe_json_loads(fields.get("source_bbox")), orig_width, orig_height)
        target_box = clamp_box(safe_json_loads(fields.get("target_bbox")), orig_width, orig_height)
        target_point = clamp_point(safe_json_loads(fields.get("target_point")), orig_width, orig_height)
        validate_annotations(task, source_box, target_box, target_point)
        guidance_mode = resolve_guidance_mode(fields.get("guidance_mode", "auto"), task, target_box, target_point)

        steps = int(float(fields.get("steps") or self.args.timesteps))
        steps = max(1, min(100, steps))
        sample_count = int(float(fields.get("sample_count") or 1))
        sample_count = max(1, min(MAX_INTERACTIVE_RESULTS, sample_count))
        seed = int(float(fields.get("seed") or -1))
        seed_modulus = 2**31
        if seed < 0:
            seed = random.randint(0, seed_modulus - 1)
        base_seed = seed % seed_modulus
        seeds = [(base_seed + index) % seed_modulus for index in range(sample_count)]

        input_hash = image_hash(image_bytes)
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{input_hash}"
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        image_path = run_dir / "input_original.png"
        resized_image_path = run_dir / "input_resized.png"
        depth_path = run_dir / "depth.npy"
        depth_vis_path = run_dir / "depth_vis.png"
        mask_vis_path = run_dir / "guidance.png"
        reference_path = None

        image_path.write_bytes(image_bytes)
        src_image = Image.open(image_path).convert("RGB")
        if orig_width <= 0 or orig_height <= 0:
            orig_width, orig_height = src_image.size

        height = int(self.args.pano_height)
        width = height * 2
        bicubic = getattr(Image, "Resampling", Image).BICUBIC
        resized = src_image.resize((width, height), bicubic)
        resized.save(resized_image_path)

        self._ensure_da2_depth(
            image_path=resized_image_path,
            image_hash_value=input_hash,
            pano_height=height,
            depth_path=depth_path,
            vis_path=depth_vis_path,
        )

        source_mask, target_mask, point_mask = self._build_masks(
            source_box=source_box,
            target_box=target_box,
            target_point=target_point,
            orig_size=(orig_width, orig_height),
            target_size=(width, height),
        )
        self._save_guidance_visual(
            image=np.asarray(resized),
            source_mask=source_mask,
            target_mask=target_mask,
            point_mask=point_mask,
            output_path=mask_vis_path,
        )

        input_tensor = self._image_to_pano_tensor(np.asarray(resized))
        mask_tensor = self._mask_to_pano_tensor(source_mask, target_mask, point_mask)
        depth_tensor = self._depth_to_pano_tensor(np.load(depth_path))

        batch = {
            "scene_id": ["interactive"],
            "camera_id": ["uploaded"],
            "object_id": ["interactive_object"],
            "function": [task],
            "pano_id": [run_id],
            "height": [height],
            "width": [width],
            "pano": input_tensor.clone(),
            "remove_pano": input_tensor.clone(),
            "pano_mask": mask_tensor,
            "pano_full_mask": mask_tensor.clone(),
            "input_depth": depth_tensor,
            "inference_source_mask_type": ["bbox"],
            "has_ref": [False],
            "ref_img_path": [""],
            "ref_source_id": [""],
        }
        fixed_prompt = default_task_prompt(task)
        batch.update(build_prompt_variants(task))

        if reference_bytes:
            reference_path = run_dir / "reference.png"
            reference_path.write_bytes(reference_bytes)
            ref = Image.open(reference_path).convert("RGB")
            ref = self._resize_reference_with_padding(ref, int(self.args.refs_resolution), bicubic)
            ref.save(reference_path)
            ref_tensor = self._reference_to_tensor(np.asarray(ref))
            batch["refs"] = ref_tensor
            if task == "add":
                batch["has_ref"] = [True]
                batch["ref_img_path"] = [str(reference_path)]
                batch["ref_source_id"] = ["uploaded_reference"]

        inference_devices = self._select_inference_devices(len(seeds))
        results = self._run_seed_inference_jobs(
            batch=batch,
            seeds=seeds,
            run_dir=run_dir,
            sample_count=sample_count,
            guidance_mode=guidance_mode,
            steps=steps,
            inference_devices=inference_devices,
        )

        metadata = {
            "task": task,
            "guidance_mode": guidance_mode,
            "steps": steps,
            "seed": seeds[0],
            "base_seed": base_seed,
            "seeds": seeds,
            "sample_count": sample_count,
            "source_bbox": source_box,
            "target_bbox": target_box,
            "target_point": target_point,
            "height": height,
            "width": width,
            "prompt": fixed_prompt,
            "inference_devices": [str(device) for device in inference_devices],
            "parallel_workers": len(inference_devices),
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        seed_text = str(seeds[0]) if len(seeds) == 1 else f"{seeds[0]}..{seeds[-1]} ({len(seeds)} results)"
        worker_text = ",".join(str(device) for device in inference_devices)
        message = f"Done: task={task}, guidance={guidance_mode}, steps={steps}, seed={seed_text}, devices={worker_text}"
        return {
            "ok": True,
            "message": message,
            "seed": seeds[0],
            "seeds": seeds,
            "results": results,
            "result_url": results[0]["url"] if results else None,
            "depth_vis_url": self._output_url(depth_vis_path) if depth_vis_path.exists() else None,
            "mask_vis_url": self._output_url(mask_vis_path),
            "reference_url": self._output_url(reference_path) if reference_path and reference_path.exists() else None,
        }

    def _resolve_inference_devices(self):
        import torch

        if self._inference_devices is not None:
            return self._inference_devices

        if self.args.device:
            devices = [self._canonical_torch_device(str(self.args.device), torch)]
        else:
            configured = str(self.args.inference_devices or "auto").strip()
            if configured.lower() in {"", "auto"}:
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    devices = [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())]
                else:
                    devices = [torch.device("cpu")]
            else:
                devices = [
                    self._canonical_torch_device(token, torch)
                    for token in _parse_csv_tokens(configured)
                ]

        max_parallel_gpus = max(0, int(getattr(self.args, "max_parallel_gpus", 0) or 0))
        if max_parallel_gpus > 0:
            limited_devices = []
            cuda_count = 0
            for device in devices:
                if device.type == "cuda":
                    if cuda_count >= max_parallel_gpus:
                        continue
                    cuda_count += 1
                limited_devices.append(device)
            devices = limited_devices

        unique_devices = []
        seen = set()
        for device in devices:
            key = str(device)
            if key in seen:
                continue
            seen.add(key)
            unique_devices.append(device)

        if not unique_devices:
            raise ValueError("No inference devices are available.")

        self._inference_devices = unique_devices
        return self._inference_devices

    @staticmethod
    def _canonical_torch_device(raw_device: str, torch_module):
        value = str(raw_device).strip().lower()
        if value in {"cpu", "none", "off", "-1"}:
            return torch_module.device("cpu")
        if value.isdigit():
            value = f"cuda:{value}"
        device = torch_module.device(value)
        if device.type == "cuda":
            if not torch_module.cuda.is_available():
                raise ValueError(f"CUDA device requested but CUDA is not available: {raw_device}")
            index = 0 if device.index is None else int(device.index)
            if index < 0 or index >= torch_module.cuda.device_count():
                raise ValueError(
                    f"CUDA device {raw_device} is not visible. "
                    f"Visible CUDA device count is {torch_module.cuda.device_count()}."
                )
            return torch_module.device(f"cuda:{index}")
        return device

    def _select_inference_devices(self, seed_count: int):
        devices = self._resolve_inference_devices()
        if seed_count <= 1:
            return devices[:1]
        return devices[: max(1, min(seed_count, len(devices)))]

    def _run_seed_inference_jobs(
        self,
        batch,
        seeds: list[int],
        run_dir: Path,
        sample_count: int,
        guidance_mode: str,
        steps: int,
        inference_devices,
    ):
        if not seeds:
            return []

        for device in inference_devices:
            self._load_model_for_device(device)

        assignments = [[] for _ in inference_devices]
        for index, seed in enumerate(seeds):
            assignments[index % len(inference_devices)].append((index, seed))

        result_slots = [None] * len(seeds)
        if len(inference_devices) == 1:
            for index, result in self._run_seed_chunk(
                batch=batch,
                assignments=assignments[0],
                device=inference_devices[0],
                run_dir=run_dir,
                sample_count=sample_count,
                guidance_mode=guidance_mode,
                steps=steps,
            ):
                result_slots[index] = result
        else:
            with ThreadPoolExecutor(max_workers=len(inference_devices)) as executor:
                futures = [
                    executor.submit(
                        self._run_seed_chunk,
                        batch=batch,
                        assignments=device_assignments,
                        device=device,
                        run_dir=run_dir,
                        sample_count=sample_count,
                        guidance_mode=guidance_mode,
                        steps=steps,
                    )
                    for device, device_assignments in zip(inference_devices, assignments)
                    if device_assignments
                ]
                for future in as_completed(futures):
                    for index, result in future.result():
                        result_slots[index] = result

        return [result for result in result_slots if result is not None]

    def _run_seed_chunk(
        self,
        batch,
        assignments: list[tuple[int, int]],
        device,
        run_dir: Path,
        sample_count: int,
        guidance_mode: str,
        steps: int,
    ):
        from PIL import Image
        import numpy as np
        import torch

        if device.type == "cuda":
            torch.cuda.set_device(device)

        model, device = self._load_model_for_device(device)
        self._configure_model_for_request(model, steps, guidance_mode)
        batch_on_device = self._move_batch_to_device(batch, device)

        chunk_results = []
        with torch.inference_mode():
            for index, current_seed in assignments:
                generator = self._make_torch_generator(current_seed, device)
                pred = model.inference(
                    batch_on_device,
                    guidance_mode=guidance_mode,
                    generator=generator,
                )
                pred_img = self._tensor_to_image(pred[0])
                if sample_count == 1:
                    result_path = run_dir / "result.png"
                else:
                    result_path = run_dir / f"result_{index:02d}_seed_{current_seed}.png"
                Image.fromarray(np.asarray(pred_img).astype(np.uint8)).save(result_path)
                chunk_results.append(
                    (
                        index,
                        {
                            "seed": current_seed,
                            "url": self._output_url(result_path),
                            "device": str(device),
                        },
                    )
                )
                del pred

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return chunk_results

    @staticmethod
    def _make_torch_generator(seed: int, device):
        import torch

        if device.type == "cuda":
            generator = torch.Generator(device=device)
        else:
            generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator

    def _configure_model_for_request(self, model, steps: int, guidance_mode: str) -> None:
        model.hparams.inference_timesteps = int(steps)
        model.hparams.guidance_scale = float(self.args.guidance_scale)
        model.hparams.inference_target_guidance_mode = guidance_mode

    def _load_model_for_device(self, device):
        with self._model_lock:
            key = str(device)
            if key in self._models:
                return self._models[key], device

            import torch

            from models.Mover360_depth.mover360 import Mover360_depth

            device = torch.device(device)
            if not self.args.ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.args.ckpt}")
            if device.type == "cuda":
                torch.cuda.set_device(device)

            print(f"Loading Mover360_depth model on {device}")
            model = Mover360_depth(
                model_id=self.args.model_id,
                ckpt_path=str(self.args.ckpt),
                inference_timesteps=int(self.args.timesteps),
                guidance_scale=float(self.args.guidance_scale),
                compile_models=bool(self.args.compile_models),
                huggingface_cache=str(self.args.huggingface_cache) if self.args.huggingface_cache else None,
                use_mask_in_inference=True,
                use_ref_in_inference=True,
            )
            model.eval()
            model.to(device)
            # The frozen Qwen3 text encoder (~8 GB) is only needed for a moment per
            # request; parking it on CPU leaves that headroom for the attention pass.
            model.offload_text_encoder_after_encode = True
            model.offload_text_encoder()
            self._models[key] = model
            print(f"Loaded Mover360_depth model on {device}")
            return self._models[key], device

    def _ensure_da2_depth(
        self,
        image_path: Path,
        image_hash_value: str,
        pano_height: int,
        depth_path: Path,
        vis_path: Path,
    ) -> None:
        cache_dir = self.depth_cache_dir / f"h{int(pano_height)}_{image_hash_value}"
        cached_depth_path = cache_dir / "depth.npy"
        cached_vis_path = cache_dir / "depth_vis.png"

        if cached_depth_path.exists():
            shutil.copy2(cached_depth_path, depth_path)
            self._save_near_focused_depth_visual(depth_path, vis_path)
            if vis_path.exists():
                shutil.copy2(vis_path, cached_vis_path)
            (depth_path.parent / "da2.log").write_text(
                f"Reused cached DA-2 depth from {cache_dir}\n",
                encoding="utf-8",
            )
            return

        if self._copy_existing_depth_from_runs(
            image_hash_value=image_hash_value,
            pano_height=pano_height,
            depth_path=depth_path,
            vis_path=vis_path,
            cache_depth_path=cached_depth_path,
            cache_vis_path=cached_vis_path,
        ):
            self._save_near_focused_depth_visual(depth_path, vis_path)
            if vis_path.exists():
                shutil.copy2(vis_path, cached_vis_path)
            return

        self._run_da2_depth(image_path, depth_path, vis_path)
        self._save_near_focused_depth_visual(depth_path, vis_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(depth_path, cached_depth_path)
        if vis_path.exists():
            shutil.copy2(vis_path, cached_vis_path)

    def _copy_existing_depth_from_runs(
        self,
        image_hash_value: str,
        pano_height: int,
        depth_path: Path,
        vis_path: Path,
        cache_depth_path: Path,
        cache_vis_path: Path,
    ) -> bool:
        candidates = []
        for candidate_dir in self.output_dir.glob(f"*_{image_hash_value}"):
            if not candidate_dir.is_dir() or candidate_dir == depth_path.parent:
                continue
            candidate_depth = candidate_dir / "depth.npy"
            if not candidate_depth.exists():
                continue
            candidate_resized = candidate_dir / "input_resized.png"
            if candidate_resized.exists():
                try:
                    from PIL import Image

                    with Image.open(candidate_resized) as image:
                        if image.height != int(pano_height):
                            continue
                except Exception:
                    continue
            candidates.append((candidate_depth.stat().st_mtime, candidate_dir))

        if not candidates:
            return False

        _, source_dir = max(candidates)
        source_depth = source_dir / "depth.npy"
        source_vis = source_dir / "depth_vis.png"
        shutil.copy2(source_depth, depth_path)
        if source_vis.exists():
            shutil.copy2(source_vis, vis_path)

        cache_depth_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_depth, cache_depth_path)
        if source_vis.exists():
            shutil.copy2(source_vis, cache_vis_path)

        (depth_path.parent / "da2.log").write_text(
            f"Reused existing DA-2 depth from {source_dir}\n",
            encoding="utf-8",
        )
        return True

    @staticmethod
    def _save_near_focused_depth_visual(depth_path: Path, vis_path: Path) -> None:
        from PIL import Image
        import numpy as np
        import matplotlib

        depth = np.load(depth_path).astype(np.float32).squeeze()
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}")

        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            Image.fromarray(np.zeros((*depth.shape, 3), dtype=np.uint8)).save(vis_path)
            return

        valid_depth = depth[valid]
        lo = float(np.quantile(valid_depth, 0.02))
        hi = float(np.quantile(valid_depth, 0.95))
        if hi <= lo + 1e-6:
            hi = float(np.quantile(valid_depth, 0.98))
        if hi <= lo + 1e-6:
            normalized = np.zeros_like(depth, dtype=np.float32)
        else:
            normalized = (depth - lo) / (hi - lo)
        normalized = np.clip(normalized, 0.0, 1.0)

        normalized[~valid] = 1.0
        rgb = matplotlib.colormaps["Spectral"](normalized, bytes=False)[:, :, :3] * 255.0
        rgb[~valid] = 0
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(vis_path)

    def _run_da2_depth(self, image_path: Path, depth_path: Path, vis_path: Path) -> None:
        if not self.args.depth_infer.exists():
            raise FileNotFoundError(f"DA-2 infer script not found: {self.args.depth_infer}")
        env = os.environ.copy()
        depth_cuda_visible_devices = self._resolve_depth_cuda_visible_devices()
        env["CUDA_VISIBLE_DEVICES"] = depth_cuda_visible_devices
        python_paths = [str(DA2_DIR), str(DA2_DIR / "src")]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env.setdefault("WANDB_MODE", "disabled")
        cmd = [
            sys.executable,
            str(self.args.depth_infer),
            "--image_path",
            str(image_path),
            "--config_path",
            str(self.args.depth_config),
            "--output_path",
            str(depth_path),
            "--vis_path",
            str(vis_path),
            "--save_vis",
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(DA2_DIR),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_header = (
            f"DA-2 CUDA_VISIBLE_DEVICES={depth_cuda_visible_devices or '<hidden/cpu>'}\n"
            f"Command: {' '.join(cmd)}\n\n"
        )
        log_text = log_header + completed.stdout
        depth_configured = str(self.args.depth_cuda_visible_devices or "auto").strip().lower()
        if (
            completed.returncode != 0
            and depth_configured in {"", "auto"}
            and depth_cuda_visible_devices
            and "busy or unavailable" in completed.stdout
        ):
            cpu_env = env.copy()
            cpu_env["CUDA_VISIBLE_DEVICES"] = ""
            cpu_completed = subprocess.run(
                cmd,
                cwd=str(DA2_DIR),
                env=cpu_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            cpu_log_header = (
                "\n\n--- Retrying DA-2 on CPU because the selected CUDA device was busy/unavailable ---\n"
                "DA-2 CUDA_VISIBLE_DEVICES=<hidden/cpu>\n"
                f"Command: {' '.join(cmd)}\n\n"
            )
            log_text += cpu_log_header + cpu_completed.stdout
            completed = cpu_completed
        (depth_path.parent / "da2.log").write_text(
            log_text,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"DA-2 failed with code {completed.returncode}. See {depth_path.parent / 'da2.log'}")
        if not depth_path.exists():
            raise FileNotFoundError(f"DA-2 did not write depth file: {depth_path}")

    def _resolve_depth_cuda_visible_devices(self) -> str:
        configured = str(self.args.depth_cuda_visible_devices or "auto").strip()
        if configured.lower() not in {"", "auto"}:
            if configured.lower() in {"cpu", "none", "off", "-1"}:
                return ""
            return configured

        main_devices = _parse_cuda_device_list(str(self.args.cuda_visible_devices or ""))
        rows = _query_gpu_usage()
        if not rows:
            return str(self.args.cuda_visible_devices or "")

        max_used_mb = int(getattr(self.args, "auto_gpu_max_used_mb", DEFAULT_AUTO_GPU_MAX_USED_MB))
        max_util = int(getattr(self.args, "auto_gpu_max_util", DEFAULT_AUTO_GPU_MAX_UTIL))
        idle_rows = [
            row
            for row in rows
            if row[1] <= max_used_mb and row[2] <= max_util
        ]
        if not idle_rows:
            return ""

        non_main_idle_rows = [
            row
            for row in idle_rows
            if str(row[0]) not in main_devices
        ]
        candidates = non_main_idle_rows or idle_rows
        selected = min(candidates, key=lambda row: (row[1], row[2], row[0]))
        return str(selected[0])

    @staticmethod
    def _build_masks(
        source_box: tuple[float, float, float, float] | None,
        target_box: tuple[float, float, float, float] | None,
        target_point: tuple[float, float] | None,
        orig_size: tuple[int, int],
        target_size: tuple[int, int],
    ):
        import numpy as np

        orig_w, orig_h = orig_size
        target_w, target_h = target_size
        sx = float(target_w) / float(max(orig_w, 1))
        sy = float(target_h) / float(max(orig_h, 1))

        def box_mask(box):
            mask = np.zeros((target_h, target_w), dtype=np.float32)
            if box is None:
                return mask
            x1, y1, x2, y2 = box
            ix1 = int(round(x1 * sx))
            iy1 = int(round(y1 * sy))
            ix2 = int(round(x2 * sx))
            iy2 = int(round(y2 * sy))
            ix1 = max(0, min(target_w - 1, ix1))
            iy1 = max(0, min(target_h - 1, iy1))
            ix2 = max(ix1 + 1, min(target_w, ix2))
            iy2 = max(iy1 + 1, min(target_h, iy2))
            mask[iy1:iy2, ix1:ix2] = 1.0
            return mask

        def point_guidance_map(point):
            out = np.zeros((target_h, target_w), dtype=np.float32)
            if point is None:
                return out

            cx = min(max(float(point[0]) * sx, 0.0), float(max(target_w - 1, 0)))
            cy = min(max(float(point[1]) * sy, 0.0), float(max(target_h - 1, 0)))

            ys = np.arange(target_h, dtype=np.float32)[:, None]
            xs = np.arange(target_w, dtype=np.float32)[None, :]
            dx = np.abs(xs - cx)
            if target_w > 1:
                dx = np.minimum(dx, float(target_w) - dx)
            dy = np.abs(ys - cy)

            point_scale = float(target_w) / float(max(POINT_GUIDANCE_REFERENCE_WIDTH, 1))
            radius_px = max(int(round(float(POINT_GUIDANCE_RADIUS_PX) * point_scale)), 0)
            if radius_px > 0:
                core = ((dx <= radius_px) & (dy <= radius_px)).astype(np.float32)
                out = np.maximum(out, core)

            sigma_px = float(POINT_GUIDANCE_SIGMA_PX) * point_scale
            if sigma_px > 0:
                gaussian = np.exp(-(dx * dx + dy * dy) / max(2.0 * sigma_px * sigma_px, 1e-8))
                gaussian = gaussian / max(float(gaussian.max()), 1e-8)
                out = np.maximum(out, gaussian.astype(np.float32))

            return np.clip(out, 0.0, 1.0).astype(np.float32)

        source_mask = box_mask(source_box)
        target_mask = box_mask(target_box)
        point_mask = point_guidance_map(target_point)
        return source_mask, target_mask, point_mask

    @staticmethod
    def _save_guidance_visual(image, source_mask, target_mask, point_mask, output_path: Path) -> None:
        import numpy as np
        from PIL import Image

        base = image.astype(np.float32).copy()
        overlay = np.zeros_like(base)
        overlay[..., 0] += target_mask * 245.0
        overlay[..., 1] += target_mask * 155.0
        overlay[..., 2] += source_mask * 235.0
        overlay[..., 0] += point_mask * 220.0
        mask_any = np.clip(source_mask + target_mask + point_mask, 0.0, 1.0)[..., None]
        vis = base * (1.0 - 0.45 * mask_any) + overlay * (0.45 * mask_any)
        Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8)).save(output_path)

    @staticmethod
    def _image_to_pano_tensor(image):
        import numpy as np
        import torch

        arr = image.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _resize_reference_with_padding(image, size: int, resample):
        from PIL import Image

        size = max(1, int(size))
        src_w, src_h = image.size
        scale = float(size) / float(max(src_w, src_h, 1))
        dst_w = max(1, int(round(src_w * scale)))
        dst_h = max(1, int(round(src_h * scale)))
        resized = image.resize((dst_w, dst_h), resample)
        padded = Image.new("RGB", (size, size), (255, 255, 255))
        padded.paste(resized, ((size - dst_w) // 2, (size - dst_h) // 2))
        return padded

    @staticmethod
    def _reference_to_tensor(image):
        import numpy as np
        import torch

        arr = image.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _mask_to_pano_tensor(source_mask, target_mask, point_mask):
        import numpy as np
        import torch

        mask = np.stack([target_mask, source_mask, point_mask], axis=0).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _normalize_depth_for_vae(depth_map, lower_quantile=0.0, upper_quantile=0.98, use_log=False):
        import numpy as np

        depth = np.asarray(depth_map, dtype=np.float32).squeeze()
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}")
        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            return np.zeros_like(depth, dtype=np.float32)
        depth = depth.copy()
        if use_log:
            depth[valid] = np.log1p(depth[valid])
        valid_depth = depth[valid]
        lo = float(np.quantile(valid_depth, lower_quantile))
        hi = float(np.quantile(valid_depth, upper_quantile))
        normalized = np.zeros_like(depth, dtype=np.float32)
        if hi <= lo + 1e-6:
            normalized[valid] = 0.5
            return normalized
        normalized[valid] = (depth[valid] - lo) / (hi - lo)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def _depth_to_pano_tensor(self, depth):
        import cv2
        import numpy as np
        import torch

        h = int(self.args.pano_height)
        w = h * 2
        depth = self._normalize_depth_for_vae(depth)
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = np.repeat(depth[:, :, None], 3, axis=2)
        depth = depth * 2.0 - 1.0
        return torch.from_numpy(depth.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _tensor_to_image(image):
        import torch

        if image.dtype in {torch.bfloat16, torch.float16}:
            image = image.to(torch.float32)
        if image.dtype != torch.uint8:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = (image * 255).round()
        image = image.detach().cpu().numpy().astype("uint8")
        if image.ndim == 3:
            image = image.transpose(1, 2, 0)
        return image

    @staticmethod
    def _move_batch_to_device(batch, device):
        import torch

        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    @staticmethod
    def _seed_everything(seed: int) -> None:
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _output_url(self, path: Path | None) -> str | None:
        if path is None:
            return None
        rel = path.resolve().relative_to(self.output_dir)
        return "/outputs/" + "/".join(rel.parts)

    def _resolve_output_url_path(self, url: str) -> Path:
        parsed = urlparse(str(url or ""))
        rel_url = parsed.path
        if not rel_url.startswith("/outputs/"):
            raise ValueError("Selected image must be an /outputs/ URL.")
        rel_path = Path(unquote(rel_url[len("/outputs/"):]))
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError("Invalid selected image path.")
        path = (self.output_dir / rel_path).resolve()
        try:
            path.relative_to(self.output_dir)
        except ValueError as exc:
            raise ValueError("Selected image path is outside output dir.") from exc
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Selected image not found: {path}")
        return path

    def save_selection(self, payload: dict[str, Any]) -> dict[str, Any]:
        image_path = self._resolve_output_url_path(str(payload.get("image_url") or ""))
        perspective_png = str(payload.get("perspective_png") or "")
        prefix = "data:image/png;base64,"
        if not perspective_png.startswith(prefix):
            raise ValueError("Missing PNG perspective preview.")
        try:
            perspective_bytes = base64.b64decode(perspective_png[len(prefix):], validate=True)
        except Exception as exc:
            raise ValueError("Invalid PNG perspective preview.") from exc
        if not perspective_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Perspective preview is not a PNG image.")

        selected_root = self.output_dir / "selected_img"
        selected_root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{timestamp}_{int(time.time() * 1000) % 1000:03d}_{image_path.stem}"
        selected_dir = selected_root / stem
        selected_dir.mkdir(parents=True, exist_ok=False)

        selected_image_path = selected_dir / f"selected{image_path.suffix or '.png'}"
        perspective_path = selected_dir / "perspective.png"
        metadata_path = selected_dir / "view.json"

        shutil.copy2(image_path, selected_image_path)
        perspective_path.write_bytes(perspective_bytes)
        metadata = {
            "source_image": str(image_path),
            "selected_image": str(selected_image_path),
            "perspective_image": str(perspective_path),
            "view": payload.get("view") or {},
            "saved_at": timestamp,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "ok": True,
            "saved_dir": str(selected_dir),
            "selected_image_url": self._output_url(selected_image_path),
            "perspective_url": self._output_url(perspective_path),
            "metadata_url": self._output_url(metadata_path),
        }


class InteractiveHandler(BaseHTTPRequestHandler):
    runner: InteractiveRunner
    max_upload_bytes: int

    server_version = "Mover360Interactive/1.0"

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/outputs/"):
            self._serve_output(parsed.path[len("/outputs/"):])
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/save-selection":
            try:
                payload = self._read_json()
                result = self.runner.save_selection(payload)
                self._send_json(result)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"ok": False, "error": str(exc)}, status=500)
            return
        if parsed.path != "/api/infer":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > self.max_upload_bytes:
                raise ValueError("Upload is too large.")
            if content_length <= 0:
                raise ValueError("Missing panorama upload.")
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("Expected multipart/form-data upload.")
            fields, files = self._read_multipart()
            image_bytes = files.get("image")
            if not image_bytes:
                raise ValueError("Missing panorama upload.")
            result = self.runner.run(
                image_bytes=image_bytes,
                reference_bytes=files.get("reference"),
                fields=fields,
            )
            self._send_json(result)
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            traceback.print_exc()
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > self.max_upload_bytes:
            raise ValueError("Request is too large.")
        if content_length <= 0:
            raise ValueError("Missing JSON request body.")
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ValueError("Expected application/json request.")
        data = self.rfile.read(content_length)
        try:
            payload = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON request body.") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON request body must be an object.")
        return payload

    def _read_multipart(self) -> tuple[dict[str, str], dict[str, bytes]]:
        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
        }
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=environ)
        fields: dict[str, str] = {}
        files: dict[str, bytes] = {}
        for key in form.keys():
            item = form[key]
            if isinstance(item, list):
                item = item[0]
            if getattr(item, "filename", None):
                files[key] = item.file.read()
            else:
                fields[key] = item.value
        return fields, files

    def _serve_output(self, rel_url: str) -> None:
        rel_path = Path(unquote(rel_url))
        if rel_path.is_absolute() or ".." in rel_path.parts:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        path = (self.runner.output_dir / rel_path).resolve()
        try:
            path.relative_to(self.runner.output_dir)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), mime)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def enable_pre_ampere_attention_fallback() -> None:
    """On pre-Ampere GPUs (sm < 80, e.g. Quadro RTX 6000) PyTorch has no bf16
    flash/memory-efficient SDPA kernel, so Flux2's bf16 attention silently falls
    back to the math path and materializes the full attention matrix (~9 GiB per
    call at our sequence lengths -> OOM on 24 GB cards). Route those calls
    through the fp16 memory-efficient kernel instead. fp16 is numerically safe
    here: Flux2 RMS-normalizes Q/K before attention, and fp16 has more mantissa
    bits than bf16. No-op on Ampere+ (e.g. A5000), where bf16 kernels exist.
    """
    import torch

    if not torch.cuda.is_available():
        return
    if all(
        torch.cuda.get_device_capability(i) >= (8, 0)
        for i in range(torch.cuda.device_count())
    ):
        return

    from diffusers.models.transformers import transformer_flux2

    original = transformer_flux2.dispatch_attention_fn

    def dispatch_with_fp16_fallback(query, key, value, attn_mask=None, *args, **kwargs):
        if (
            query.dtype == torch.bfloat16
            and query.is_cuda
            and torch.cuda.get_device_capability(query.device) < (8, 0)
        ):
            if attn_mask is not None and attn_mask.is_floating_point():
                attn_mask = attn_mask.to(torch.float16)
            out = original(
                query.to(torch.float16),
                key.to(torch.float16),
                value.to(torch.float16),
                attn_mask,
                *args,
                **kwargs,
            )
            return out.to(query.dtype)
        return original(query, key, value, attn_mask, *args, **kwargs)

    transformer_flux2.dispatch_attention_fn = dispatch_with_fp16_fallback
    print(
        "Pre-Ampere GPU detected: routing bf16 attention through the fp16 "
        "memory-efficient kernel (bf16 SDPA has no efficient kernel on sm<80)."
    )


def main() -> None:
    args = parse_args()
    args.cuda_visible_devices = resolve_cuda_visible_devices_arg(
        args.cuda_visible_devices,
        max_used_mb=int(args.auto_gpu_max_used_mb),
        max_util=int(args.auto_gpu_max_util),
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")
    if args.huggingface_cache:
        os.environ.setdefault("HF_HOME", str(Path(args.huggingface_cache).parent))
        os.environ.setdefault("HF_HUB_CACHE", str(args.huggingface_cache))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(args.huggingface_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(args.huggingface_cache))
    os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES") or "<hidden/cpu>"
    print(f"CUDA_VISIBLE_DEVICES={visible_devices}")
    print(f"Mover360 inference devices={args.inference_devices}, max_parallel_gpus={args.max_parallel_gpus}")

    enable_pre_ampere_attention_fallback()

    handler_cls = type(
        "BoundInteractiveHandler",
        (InteractiveHandler,),
        {
            "runner": InteractiveRunner(args),
            "max_upload_bytes": int(args.max_upload_mb) * 1024 * 1024,
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Mover360 interactive UI: http://{args.host}:{args.port}")
    print("Model and DA-2 are loaded on first Run request.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
