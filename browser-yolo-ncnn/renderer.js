const COCO_SKELETON = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 1], [0, 2], [1, 3], [2, 4],
  [0, 5], [0, 6]
];

const PERSON_COLORS = [
  "#65e6dc", "#b58cff", "#ffca70", "#ff7f9d",
  "#8fe17c", "#75aaff", "#ff9d65", "#d9a6ff"
];

export class PoseRenderer {
  constructor(canvas, options = {}) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError("PoseRenderer requires a canvas element.");
    }

    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: true, desynchronized: true });
    this.keypointThreshold = options.keypointThreshold ?? 0.25;
    this.boxes = options.boxes ?? true;
    this.labels = options.labels ?? true;
    this.lineWidth = options.lineWidth ?? 2.5;
    this.pointRadius = options.pointRadius ?? 3.2;
    this.smoothed = new Map();
    this.smoothing = options.smoothing ?? 0.35;
    this.lastSource = { width: 0, height: 0 };
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }

  clear() {
    this.resize();
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  normalizePoint(point) {
    if (Array.isArray(point)) {
      return { x: Number(point[0]), y: Number(point[1]), score: Number(point[2] ?? 1) };
    }
    return {
      x: Number(point?.x),
      y: Number(point?.y),
      score: Number(point?.score ?? point?.confidence ?? point?.visibility ?? 1)
    };
  }

  smoothPoints(trackId, points) {
    const previous = this.smoothed.get(trackId);
    if (!previous || previous.length !== points.length) {
      this.smoothed.set(trackId, points.map(p => ({ ...p })));
      return points;
    }

    const amount = this.smoothing;
    const result = points.map((point, index) => {
      const old = previous[index];
      if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return point;
      if (!old || point.score < this.keypointThreshold) return point;
      return {
        x: old.x + (point.x - old.x) * amount,
        y: old.y + (point.y - old.y) * amount,
        score: point.score
      };
    });
    this.smoothed.set(trackId, result.map(p => ({ ...p })));
    return result;
  }

  draw(detections = [], sourceWidth, sourceHeight) {
    this.resize();
    const ctx = this.ctx;
    const canvasWidth = this.canvas.width;
    const canvasHeight = this.canvas.height;
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    if (!sourceWidth || !sourceHeight || !Array.isArray(detections)) return;

    const scale = Math.min(canvasWidth / sourceWidth, canvasHeight / sourceHeight);
    const dpr = canvasWidth / Math.max(1, this.canvas.getBoundingClientRect().width);
    const offsetX = (canvasWidth - sourceWidth * scale) / 2;
    const offsetY = (canvasHeight - sourceHeight * scale) / 2;

    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    detections.forEach((detection, index) => {
      const trackId = detection.trackId ?? detection.id ?? index;
      const color = PERSON_COLORS[Math.abs(Number(trackId) || index) % PERSON_COLORS.length];
      const rawPoints = Array.isArray(detection.keypoints) ? detection.keypoints.map(p => this.normalizePoint(p)) : [];
      const points = (detection.trackId != null || detection.id != null)
        ? this.smoothPoints(trackId, rawPoints)
        : rawPoints;

      if (this.boxes && detection.box) {
        const box = detection.box;
        const x1 = Number(box.x1 ?? box[0]);
        const y1 = Number(box.y1 ?? box[1]);
        const x2 = Number(box.x2 ?? box[2]);
        const y2 = Number(box.y2 ?? box[3]);
        if ([x1, y1, x2, y2].every(Number.isFinite)) {
          const width = Math.max(0, x2 - x1);
          const height = Math.max(0, y2 - y1);
          ctx.strokeStyle = color;
          ctx.lineWidth = this.lineWidth * dpr / scale;
          ctx.strokeRect(x1, y1, width, height);
          if (this.labels) {
            const confidence = Number(detection.score ?? detection.confidence);
            const title = detection.label || "Person";
            const suffix = Number.isFinite(confidence) ? ` ${Math.round(confidence * 100)}%` : "";
            this.drawLabel(`${title}${suffix}`, x1, y1, color, scale, dpr);
          }
        }
      }

      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = this.lineWidth / scale;

      for (const [a, b] of COCO_SKELETON) {
        const first = points[a];
        const second = points[b];
        if (!first || !second ||
            first.score < this.keypointThreshold ||
            second.score < this.keypointThreshold ||
            ![first.x, first.y, second.x, second.y].every(Number.isFinite)) continue;

        ctx.beginPath();
        ctx.moveTo(first.x, first.y);
        ctx.lineTo(second.x, second.y);
        ctx.stroke();
      }

      points.forEach(point => {
        if (!point || point.score < this.keypointThreshold ||
            !Number.isFinite(point.x) || !Number.isFinite(point.y)) return;
        ctx.beginPath();
        ctx.arc(point.x, point.y, this.pointRadius * dpr / scale, 0, Math.PI * 2);
        ctx.fill();
        ctx.lineWidth = 1.25 * dpr / scale;
        ctx.strokeStyle = "rgba(8, 8, 14, 0.9)";
        ctx.stroke();
        ctx.strokeStyle = color;
      });
    });

    ctx.restore();
  }

  drawLabel(text, x, y, color, scale, dpr) {
    const ctx = this.ctx;
    const fontSize = 12 * dpr / scale;
    const paddingX = 6 * dpr / scale;
    const height = 21 * dpr / scale;
    ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
    const width = ctx.measureText(text).width + paddingX * 2;
    const top = Math.max(0, y - height);
    ctx.fillStyle = "rgba(10, 10, 18, 0.88)";
    ctx.fillRect(x, top, width, height);
    ctx.fillStyle = color;
    ctx.fillRect(x, top, 3 * dpr / scale, height);
    ctx.fillStyle = "#ffffff";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + paddingX, top + height / 2);
  }

  destroy() {
    this.resizeObserver.disconnect();
    this.smoothed.clear();
    this.clear();
  }
}
