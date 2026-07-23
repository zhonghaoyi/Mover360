/*
  Synced Pannellum viewer helpers for the Mover360 project page.

  Provides two globals:

  createSyncedViewerPair(container, options)
    - container: DOM node or selector string
    - options: {
        left:  { panorama: string, pannellum: {...} },
        right: { panorama: string, pannellum: {...} },
        labels?: [string, string],            // shown above each viewer
        size?: { width: number|'fluid', height: number|string },
        meta?: { title: string, description: string }
      }
    - returns { root, left, right, destroy() }

  createComparisonQuad(container, options)
    - options: {
        title?: string,
        labels?: [string, string, string, string],
        items:  [src, src, src, src],
        size?: { width: number|'fluid', height: number|string },
        pannellum?: {...}
      }
    - returns { root, cells, destroy() }

  Dragging or zooming any viewer drives all of its siblings so the
  cameras stay aligned for A/B inspection.
*/

(function () {
    'use strict';

    function resolveContainer(container) {
        if (!container) throw new Error('container is required');
        if (typeof container === 'string') {
            var el = document.querySelector(container);
            if (!el) throw new Error('Container not found: ' + container);
            return el;
        }
        return container;
    }

    function toPx(v) {
        if (v == null) return null;
        return typeof v === 'number' ? (v + 'px') : String(v);
    }

    function createEl(tag, className, styles) {
        var el = document.createElement(tag);
        if (className) el.className = className;
        if (styles) Object.assign(el.style, styles);
        return el;
    }

    function baseConfig(src) {
        return {
            type: 'equirectangular',
            panorama: src,
            autoLoad: true,
            showFullscreenCtrl: true,
            showZoomCtrl: true,
            showCompass: false,
            hfov: 120,
            minHfov: 50,
            maxHfov: 120,
            autoRotate: -5
        };
    }

    function mergeConfig(base, extra) {
        var out = Object.assign({}, base);
        if (!extra) return out;
        Object.keys(extra).forEach(function (k) {
            var v = extra[k];
            if (v && typeof v === 'object' && !Array.isArray(v)) {
                out[k] = mergeConfig(out[k] || {}, v);
            } else {
                out[k] = v;
            }
        });
        return out;
    }

    /* Shared camera-sync engine: one master drives every other viewer. */
    function makeSyncGroup(viewers) {
        var master = null;
        var running = false;

        function copyState(from, to) {
            try {
                to.lookAt(from.getPitch(), from.getYaw(), from.getHfov(), false);
            } catch (e) { /* viewer may not be ready yet */ }
        }

        function tick() {
            if (!running || master == null) return;
            for (var i = 0; i < viewers.length; i++) {
                if (i !== master && viewers[i]) copyState(viewers[master], viewers[i]);
            }
            requestAnimationFrame(tick);
        }

        function start(which) {
            master = which;
            /* stop auto-rotation everywhere so the pair cannot drift apart
               after the user has interacted once */
            viewers.forEach(function (v) {
                try { v.stopAutoRotate(); } catch (e) {}
            });
            if (!running) {
                running = true;
                requestAnimationFrame(tick);
            }
        }

        function stop() {
            if (master != null) {
                for (var i = 0; i < viewers.length; i++) {
                    if (i !== master && viewers[i]) copyState(viewers[master], viewers[i]);
                }
            }
            master = null;
            running = false;
        }

        function attach(el, which) {
            var down = false;
            el.addEventListener('pointerdown', function () { down = true; start(which); });
            window.addEventListener('pointerup', function () { if (down) { down = false; stop(); } });
            el.addEventListener('pointermove', function () { if (down) start(which); });
            el.addEventListener('wheel', function () {
                start(which);
                window.setTimeout(stop, 90);
            }, { passive: true });
            /* touch fallback for older mobile browsers without pointer events */
            el.addEventListener('touchstart', function () { down = true; start(which); }, { passive: true });
            window.addEventListener('touchend', function () { if (down) { down = false; stop(); } });
        }

        return { attach: attach, copyState: copyState };
    }

    function makeLabel(text) {
        return createEl('div', 'sv-label', {
            fontSize: '11.5px',
            fontWeight: '600',
            color: '#7c8a9c',
            letterSpacing: '0.07em',
            textTransform: 'uppercase',
            marginBottom: '6px',
            textAlign: 'center'
        }).appendChild(document.createTextNode(text)).parentNode;
    }

    function createSyncedViewerPair(container, options) {
        var root = resolveContainer(container);
        var opt = options || {};
        var size = opt.size || { width: 600, height: 400 };
        var isFluid = size.width === 'fluid' || size.width === '100%' || size.width === 'auto';
        var meta = opt.meta || {};
        var labels = opt.labels || null;

        var card = createEl('div', 'sv-card', {
            border: '1px solid rgba(15,23,42,0.08)',
            borderRadius: '14px',
            padding: '14px',
            marginBottom: '10px',
            background: '#ffffff',
            display: 'block',
            textAlign: 'center',
            transition: 'all 220ms ease',
            boxShadow: '0 1px 2px rgba(15,23,42,0.04), 0 10px 30px rgba(15,23,42,0.05)',
            maxWidth: '100%'
        });

        card.addEventListener('pointerenter', function () {
            card.style.transform = 'translateY(-3px)';
            card.style.borderColor = 'rgba(18,145,216,0.4)';
            card.style.boxShadow = '0 18px 44px rgba(15,23,42,0.12)';
        });
        card.addEventListener('pointerleave', function () {
            card.style.transform = 'translateY(0)';
            card.style.borderColor = 'rgba(15,23,42,0.08)';
            card.style.boxShadow = '0 1px 2px rgba(15,23,42,0.04), 0 10px 30px rgba(15,23,42,0.05)';
        });

        if (meta.title) {
            var titleEl = createEl('div', 'sv-card-title', {
                fontWeight: '600',
                fontSize: '15px',
                color: '#0f172a',
                marginBottom: '10px'
            });
            if (meta.accent) {
                var dot = createEl('span', 'sv-dot', {
                    display: 'inline-block',
                    width: '9px',
                    height: '9px',
                    borderRadius: '999px',
                    background: meta.accent,
                    marginRight: '8px',
                    verticalAlign: '1px'
                });
                titleEl.appendChild(dot);
            }
            titleEl.appendChild(document.createTextNode(meta.title));
            card.appendChild(titleEl);
        }

        var group = createEl('div', 'sv-group', {
            display: 'flex',
            gap: '8px',
            alignItems: 'stretch',
            width: isFluid ? '100%' : ''
        });

        var h = toPx(size.height) || '400px';
        var w = isFluid ? null : toPx(size.width);

        function makeSide(labelText) {
            var col = createEl('div', 'sv-col', {
                display: 'flex',
                flexDirection: 'column',
                flex: isFluid ? '1 1 0' : '0 0 auto',
                minWidth: '0'
            });
            if (labelText) col.appendChild(makeLabel(labelText));
            var host = createEl('div', 'sv-viewer', isFluid
                ? { height: h, width: '100%' }
                : { height: h, width: w });
            col.appendChild(host);
            return { col: col, host: host };
        }

        var leftSide = makeSide(labels ? labels[0] : null);
        var rightSide = makeSide(labels ? labels[1] : null);
        group.appendChild(leftSide.col);
        group.appendChild(rightSide.col);
        card.appendChild(group);

        if (meta.instruction) {
            /* ERP instruction map (with the reference object at its right, if any)
               shown under the viewer pair */
            var instrRow = createEl('div', 'sv-instr', {
                marginTop: '10px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '14px',
                flexWrap: 'wrap'
            });

            var instrCol = createEl('div', 'sv-instr-col', {
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                flex: '0 1 auto',
                minWidth: '220px',
                maxWidth: meta.reference ? '58%' : '62%'
            });
            instrCol.appendChild(makeLabel('ERP Instruction Map'));
            var instrLink = createEl('a', 'sv-instr-link', {
                display: 'block',
                width: '100%'
            });
            instrLink.href = meta.instruction;
            instrLink.target = '_blank';
            instrLink.rel = 'noopener';
            instrLink.title = 'Open the full-size instruction map';
            var instrImg = createEl('img', 'sv-instr-img', {
                width: '100%',
                height: 'auto',
                borderRadius: '8px',
                boxShadow: '0 2px 10px rgba(15,23,42,0.16)',
                display: 'block'
            });
            instrImg.src = meta.instruction;
            instrImg.alt = 'ERP instruction map';
            instrLink.appendChild(instrImg);
            instrCol.appendChild(instrLink);
            instrRow.appendChild(instrCol);

            if (meta.reference) {
                var refCol = createEl('div', 'sv-ref-col', {
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    flex: '0 0 auto'
                });
                refCol.appendChild(makeLabel('Reference'));
                var refLarge = createEl('img', 'sv-ref-large', {
                    height: '140px',
                    width: 'auto',
                    maxWidth: '170px',
                    objectFit: 'contain',
                    borderRadius: '8px',
                    boxShadow: '0 2px 10px rgba(15,23,42,0.16)',
                    background: '#ffffff',
                    display: 'block'
                });
                refLarge.src = meta.reference;
                refLarge.alt = 'Reference object';
                refCol.appendChild(refLarge);
                instrRow.appendChild(refCol);
            }
            card.appendChild(instrRow);
        }

        var footer = createEl('div', 'sv-card-footer', {
            marginTop: '12px',
            color: '#52627a',
            fontSize: '15px',
            lineHeight: '1.5',
            minHeight: '24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '12px',
            flexWrap: 'wrap'
        });
        if (meta.reference && !meta.instruction) {
            /* small thumbnail of the reference object — only when it is not
               already shown next to the instruction map above */
            var refBox = createEl('div', 'sv-ref-box', {
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '2px',
                flex: '0 0 auto'
            });
            var refImg = createEl('img', 'sv-ref-thumb', {
                height: '58px',
                width: 'auto',
                maxWidth: '96px',
                objectFit: 'contain',
                borderRadius: '8px',
                boxShadow: '0 2px 8px rgba(15,23,42,0.16)',
                background: '#ffffff'
            });
            refImg.src = meta.reference;
            refImg.alt = 'Reference object';
            var refLabel = createEl('div', 'sv-ref-label', {
                fontSize: '10px',
                fontWeight: '600',
                letterSpacing: '0.07em',
                textTransform: 'uppercase',
                color: '#8a97a8'
            });
            refLabel.textContent = 'Reference';
            refBox.appendChild(refImg);
            refBox.appendChild(refLabel);
            footer.appendChild(refBox);
        }
        if (meta.description) {
            var descEl = createEl('span', 'sv-desc');
            descEl.textContent = meta.description;
            footer.appendChild(descEl);
        }
        card.appendChild(footer);
        root.appendChild(card);

        var leftViewer = pannellum.viewer(leftSide.host, mergeConfig(baseConfig(opt.left && opt.left.panorama), opt.left && opt.left.pannellum));
        var rightViewer = pannellum.viewer(rightSide.host, mergeConfig(baseConfig(opt.right && opt.right.panorama), opt.right && opt.right.pannellum));

        var sync = makeSyncGroup([leftViewer, rightViewer]);
        sync.attach(leftSide.host, 0);
        sync.attach(rightSide.host, 1);

        var loaded = [false, false];
        function alignWhenReady(i) {
            return function () {
                loaded[i] = true;
                if (loaded[0] && loaded[1]) sync.copyState(leftViewer, rightViewer);
            };
        }
        leftViewer.on('load', alignWhenReady(0));
        rightViewer.on('load', alignWhenReady(1));

        return {
            root: card,
            left: leftViewer,
            right: rightViewer,
            destroy: function () {
                try { leftViewer.destroy(); } catch (e) {}
                try { rightViewer.destroy(); } catch (e) {}
                if (card.parentNode) card.parentNode.removeChild(card);
            }
        };
    }

    function createComparisonQuad(container, options) {
        var root = resolveContainer(container);
        var opt = options || {};
        var size = opt.size || { width: 300, height: 200 };
        var h = toPx(size.height) || '200px';

        var card = createEl('div', 'sv-card sv-quad', {
            border: '1px solid rgba(15,23,42,0.08)',
            borderRadius: '14px',
            padding: '14px',
            background: '#ffffff',
            boxShadow: '0 1px 2px rgba(15,23,42,0.04), 0 10px 30px rgba(15,23,42,0.05)'
        });

        if (opt.title) {
            var titleEl = createEl('div', 'sv-card-title', {
                fontWeight: '600',
                marginBottom: '10px',
                textAlign: 'center',
                color: '#e8edf4'
            });
            titleEl.textContent = opt.title;
            card.appendChild(titleEl);
        }

        var grid = createEl('div', 'sv-quad-grid', {
            display: 'flex',
            gap: '8px',
            alignItems: 'stretch',
            width: '100%'
        });

        var viewers = [];
        var hosts = [];

        function makeCell(labelText, src) {
            var box = createEl('div', 'sv-quad-cell', {
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'stretch',
                flex: '1 1 0',
                minWidth: '0'
            });
            box.appendChild(makeLabel(labelText || ''));
            var host = createEl('div', 'sv-viewer', { height: h, width: '100%' });
            box.appendChild(host);
            grid.appendChild(box);
            hosts.push(host);
            viewers.push(pannellum.viewer(host, mergeConfig(baseConfig(src), opt.pannellum)));
        }

        /* supports any number of synced cells (3-up, 4-up, ...) */
        var defaultLabels = ['Original', 'Method A', 'Method B', 'Ours'];
        var items = opt.items || [];
        var n = Math.max(2, items.length);
        for (var i = 0; i < n; i++) {
            makeCell((opt.labels && opt.labels[i]) || defaultLabels[i] || ('View ' + (i + 1)), items[i]);
        }

        card.appendChild(grid);
        root.appendChild(card);

        var sync = makeSyncGroup(viewers);
        hosts.forEach(function (hostEl, idx) { sync.attach(hostEl, idx); });

        return {
            root: card,
            cells: viewers,
            destroy: function () {
                viewers.forEach(function (v) { try { v.destroy(); } catch (e) {} });
                if (card.parentNode) card.parentNode.removeChild(card);
            }
        };
    }

    window.createSyncedViewerPair = createSyncedViewerPair;
    window.createComparisonQuad = createComparisonQuad;
})();
