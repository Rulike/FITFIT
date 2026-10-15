(function () {
  'use strict';

  const observers = new WeakMap();

  function setup(canvas, draw) {
    if (!canvas) return null;
    const old = observers.get(canvas);
    if (old) old.disconnect();

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw(ctx, rect.width, rect.height);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    observers.set(canvas, observer);
    resize();
    return resize;
  }

  function cssVar(name, fallback) {
    return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
  }

  function roundedRect(ctx, x, y, w, h, r) {
    const radius = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, radius);
  }

  function line(canvas, data, options = {}) {
    const series = options.series || [
      { key: 'total', color: '#2f75f6', fill: true },
      { key: 'high', color: '#ff9f2e', fill: false }
    ];
    setup(canvas, (ctx, width, height) => {
      const muted = cssVar('--muted', '#8295ad');
      const lineColor = cssVar('--line', 'rgba(154,180,211,.12)');
      const padding = { left: 38, right: 15, top: 14, bottom: 28 };
      const chartW = width - padding.left - padding.right;
      const chartH = height - padding.top - padding.bottom;
      const allValues = series.flatMap((item) => data[item.key] || []);
      const max = Math.max(...allValues, 10) * 1.15;
      ctx.clearRect(0, 0, width, height);
      ctx.font = '9px Inter, Microsoft YaHei, sans-serif';
      ctx.textBaseline = 'middle';
      for (let i = 0; i <= 4; i += 1) {
        const y = padding.top + chartH * i / 4;
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.beginPath(); ctx.moveTo(padding.left, y); ctx.lineTo(width - padding.right, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = muted;
        ctx.textAlign = 'right';
        ctx.fillText(Math.round(max * (1 - i / 4)), padding.left - 8, y);
      }
      data.labels.forEach((label, index) => {
        const x = padding.left + chartW * index / Math.max(data.labels.length - 1, 1);
        ctx.fillStyle = muted;
        ctx.textAlign = 'center';
        ctx.fillText(label, x, height - 9);
      });
      series.forEach((item) => {
        const values = data[item.key] || [];
        const points = values.map((value, index) => ({
          x: padding.left + chartW * index / Math.max(values.length - 1, 1),
          y: padding.top + chartH * (1 - value / max)
        }));
        if (item.fill && points.length) {
          const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
          gradient.addColorStop(0, `${item.color}3d`);
          gradient.addColorStop(1, `${item.color}00`);
          ctx.beginPath();
          ctx.moveTo(points[0].x, height - padding.bottom);
          points.forEach((point) => ctx.lineTo(point.x, point.y));
          ctx.lineTo(points[points.length - 1].x, height - padding.bottom);
          ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();
        }
        ctx.beginPath();
        points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
        ctx.strokeStyle = item.color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.stroke();
        points.forEach((point) => {
          ctx.beginPath(); ctx.arc(point.x, point.y, 2.4, 0, Math.PI * 2);
          ctx.fillStyle = item.color; ctx.fill();
        });
      });
    });
  }

  function spark(canvas, values, color) {
    setup(canvas, (ctx, width, height) => {
      ctx.clearRect(0, 0, width, height);
      const min = Math.min(...values); const max = Math.max(...values); const range = max - min || 1;
      const points = values.map((value, index) => ({
        x: 2 + (width - 4) * index / Math.max(values.length - 1, 1),
        y: 3 + (height - 7) * (1 - (value - min) / range)
      }));
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, `${color}40`); gradient.addColorStop(1, `${color}00`);
      ctx.beginPath(); ctx.moveTo(points[0].x, height);
      points.forEach((point) => ctx.lineTo(point.x, point.y));
      ctx.lineTo(points[points.length - 1].x, height); ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();
      ctx.beginPath(); points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
      ctx.strokeStyle = color; ctx.lineWidth = 1.7; ctx.stroke();
    });
  }

  function donut(canvas, items) {
    setup(canvas, (ctx, width, height) => {
      ctx.clearRect(0, 0, width, height);
      const total = items.reduce((sum, item) => sum + item.value, 0);
      const cx = width / 2; const cy = height / 2; const radius = Math.min(width, height) * .41;
      let angle = -Math.PI / 2;
      items.forEach((item) => {
        const sweep = item.value / total * Math.PI * 2;
        ctx.beginPath(); ctx.arc(cx, cy, radius, angle + .018, angle + sweep - .018);
        ctx.strokeStyle = item.color; ctx.lineWidth = Math.max(14, radius * .24); ctx.lineCap = 'butt'; ctx.stroke();
        angle += sweep;
      });
    });
  }

  function bars(canvas, data, options = {}) {
    setup(canvas, (ctx, width, height) => {
      const keys = options.keys || ['positive', 'negative'];
      const colors = options.colors || ['#2ee6a6', '#ff5d73'];
      const muted = cssVar('--muted', '#8295ad'); const grid = cssVar('--line', 'rgba(154,180,211,.12)');
      const pad = { left: 32, right: 12, top: 14, bottom: 27 };
      const chartW = width - pad.left - pad.right; const chartH = height - pad.top - pad.bottom;
      const max = Math.max(...keys.flatMap((key) => data[key]), 10) * 1.12;
      ctx.clearRect(0, 0, width, height); ctx.font = '8px Inter, Microsoft YaHei';
      for (let i = 0; i <= 4; i += 1) {
        const y = pad.top + chartH * i / 4;
        ctx.strokeStyle = grid; ctx.setLineDash([3,4]); ctx.beginPath(); ctx.moveTo(pad.left,y); ctx.lineTo(width-pad.right,y); ctx.stroke(); ctx.setLineDash([]);
      }
      const groupW = chartW / data.labels.length;
      data.labels.forEach((label, idx) => {
        keys.forEach((key, keyIdx) => {
          const value = data[key][idx]; const barW = Math.min(13, groupW * .22);
          const x = pad.left + groupW * idx + groupW / 2 + (keyIdx - (keys.length - 1) / 2) * (barW + 3) - barW / 2;
          const h = chartH * value / max; const y = pad.top + chartH - h;
          roundedRect(ctx, x, y, barW, h, 3); ctx.fillStyle = colors[keyIdx]; ctx.fill();
        });
        ctx.fillStyle = muted; ctx.textAlign = 'center'; ctx.fillText(label, pad.left + groupW * idx + groupW / 2, height - 8);
      });
    });
  }

  function graph(canvas, graphData, onSelect) {
    const state = { scale: 1, offsetX: 0, offsetY: 0, selected: graphData.selectedId || graphData.selected?.id || graphData.nodes[0]?.id, hovered: null, dragging: false, draggedNode: null, moved: false, lastX: 0, lastY: 0, width: 0, height: 0 };
    let redraw = () => {};

    // Keep the original free-form radial look, but use a golden-angle spiral so
    // larger graphs do not stack nodes on the same repeated angles.
    function arrangePositions() {
      const nodes = graphData.nodes || [];
      if (nodes.length < 3) return;
      const goldenAngle = Math.PI * (3 - Math.sqrt(5));
      const spread = Math.min(0.42, 0.16 + Math.sqrt(nodes.length) * 0.035);
      nodes.forEach((node, index) => {
        const fraction = (index + 0.8) / nodes.length;
        const radius = 0.06 + spread * Math.sqrt(fraction);
        const angle = index * goldenAngle;
        node.x = Math.max(0.06, Math.min(0.94, 0.5 + Math.cos(angle) * radius));
        node.y = Math.max(0.09, Math.min(0.91, 0.5 + Math.sin(angle) * radius));
      });
    }

    arrangePositions();

    function nodePoint(node) {
      return {
        x: (node.x * state.width - state.width / 2) * state.scale + state.width / 2 + state.offsetX,
        y: (node.y * state.height - state.height / 2) * state.scale + state.height / 2 + state.offsetY
      };
    }

    function hitNode(point) {
      return [...graphData.nodes].reverse().find((node) => {
        const p = nodePoint(node); const r = node.size * Math.sqrt(state.scale) + 8;
        return Math.hypot(point.x - p.x, point.y - p.y) <= r;
      });
    }

    redraw = setup(canvas, (ctx, width, height) => {
      state.width = width; state.height = height;
      ctx.clearRect(0, 0, width, height);
      graphData.links.forEach(([from, to, label]) => {
        const a = graphData.nodes.find((node) => node.id === from); const b = graphData.nodes.find((node) => node.id === to);
        if (!a || !b) return;
        const p1 = nodePoint(a); const p2 = nodePoint(b);
        ctx.beginPath(); ctx.moveTo(p1.x,p1.y); ctx.lineTo(p2.x,p2.y);
        const active = state.selected === from || state.selected === to || state.hovered === from || state.hovered === to;
        ctx.strokeStyle = active ? 'rgba(74,168,255,.68)' : 'rgba(132,157,187,.16)'; ctx.lineWidth = active ? 1.6 : 1; ctx.stroke();
        if (active) {
          const mx = (p1.x + p2.x) / 2; const my = (p1.y + p2.y) / 2;
          ctx.font = '8px Microsoft YaHei'; ctx.textAlign = 'center'; ctx.fillStyle = cssVar('--muted','#8295ad');
          ctx.fillText(label, mx, my - 5);
        }
      });
      graphData.nodes.forEach((node) => {
        const point = nodePoint(node); const radius = node.size * Math.sqrt(state.scale);
        if (state.selected === node.id || state.draggedNode === node) {
          ctx.beginPath(); ctx.arc(point.x, point.y, radius + 8, 0, Math.PI*2); ctx.strokeStyle = `${node.color}50`; ctx.lineWidth = 5; ctx.stroke();
        }
        const gradient = ctx.createRadialGradient(point.x-radius*.25,point.y-radius*.3,2,point.x,point.y,radius);
        gradient.addColorStop(0, `${node.color}ee`); gradient.addColorStop(1, `${node.color}83`);
        ctx.beginPath(); ctx.arc(point.x, point.y, radius, 0, Math.PI*2); ctx.fillStyle = gradient; ctx.fill();
        ctx.strokeStyle = `${node.color}cc`; ctx.lineWidth = 1; ctx.stroke();
        ctx.font = `${state.selected === node.id ? '600 ' : ''}${Math.max(8,10*Math.sqrt(state.scale))}px Microsoft YaHei`;
        ctx.fillStyle = cssVar('--text','#eef6ff'); ctx.textAlign = 'center'; ctx.fillText(node.label, point.x, point.y + radius + 15);
      });
    });

    function pointerPosition(event) {
      const rect = canvas.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }
    canvas.onpointerdown = (event) => {
      const point = pointerPosition(event); state.dragging = true; state.draggedNode = hitNode(point) || null; state.moved = false; state.lastX = point.x; state.lastY = point.y; canvas.classList.toggle('dragging-node', Boolean(state.draggedNode)); canvas.setPointerCapture(event.pointerId);
    };
    canvas.onpointermove = (event) => {
      const point = pointerPosition(event);
      if (!state.dragging) {
        const hovered = hitNode(point) || null;
        if (hovered !== state.hovered) { state.hovered = hovered; redraw(); }
        return;
      }
      const dx = point.x-state.lastX; const dy = point.y-state.lastY;
      if (Math.abs(dx)+Math.abs(dy)>2) state.moved = true;
      if (state.draggedNode) {
        state.draggedNode.x = Math.max(0.04, Math.min(0.96, ((point.x - state.offsetX - state.width / 2) / state.scale + state.width / 2) / state.width));
        state.draggedNode.y = Math.max(0.08, Math.min(0.92, ((point.y - state.offsetY - state.height / 2) / state.scale + state.height / 2) / state.height));
      } else {
        state.offsetX += dx; state.offsetY += dy;
      }
      state.lastX=point.x; state.lastY=point.y; redraw();
    };
    canvas.onpointerup = (event) => {
      state.dragging = false; canvas.classList.remove('dragging-node');
      if (state.moved) { state.draggedNode = null; return; }
      const point = pointerPosition(event);
      const hit = hitNode(point);
      if (hit) { state.selected=hit.id; redraw(); if (onSelect) onSelect(hit); }
      state.draggedNode = null;
    };
    canvas.onwheel = (event) => { event.preventDefault(); state.scale = Math.max(.65,Math.min(1.7,state.scale + (event.deltaY<0?.1:-.1))); redraw(); };

    return {
      zoom(delta) { state.scale=Math.max(.65,Math.min(1.7,state.scale+delta)); redraw(); },
      reset() { state.scale=1; state.offsetX=0; state.offsetY=0; redraw(); }
    };
  }

  window.FinCharts = { line, spark, donut, bars, graph };
})();
