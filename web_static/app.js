const fields = [
  "building_path",
  "flood_folder",
  "section_path",
  "auxiliary_path",
  "river_network_path",
  "output_dir",
  "village_name",
  "threshold",
  "time_interval_hours",
  "value_type",
  "scenario_name",
  "section_reference_mode",
  "show_river_network",
  "export_corrected_river_network",
];

const state = {
  config: {},
  jobId: null,
  result: null,
  frameCells: new Map(),
  frameIndex: 0,
  playing: false,
  showShallow: true,
  showOriginalRiver: true,
  showCorrectedRiver: true,
  showSections: true,
  extent: null,
  homeExtent: null,
  drag: null,
};

const el = (id) => document.getElementById(id);
const canvas = el("mapCanvas");
const ctx = canvas.getContext("2d");

function getInputValue(key) {
  const node = el(key);
  if (!node) return "";
  if (node.type === "checkbox") return node.checked;
  return node.value;
}

function setInputValue(key, value) {
  const node = el(key);
  if (!node) return;
  if (node.type === "checkbox") {
    node.checked = Boolean(value);
    return;
  }
  node.value = value ?? "";
}

async function loadConfig() {
  const res = await fetch("/api/config");
  state.config = await res.json();
  for (const key of fields) setInputValue(key, state.config[key]);
  el("configHint").textContent = "已载入 config.yaml";
  setStatus("就绪", "idle");
}

function collectConfig() {
  const config = {};
  for (const key of fields) config[key] = getInputValue(key);
  return config;
}

async function runAnalysis() {
  setStatus("正在启动分析", "running");
  state.result = null;
  state.frameCells.clear();
  state.frameIndex = 0;
  render();
  const res = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(collectConfig()),
  });
  const data = await res.json();
  state.jobId = data.job_id;
  pollJob();
}

async function pollJob() {
  if (!state.jobId) return;
  const res = await fetch(`/api/job/${state.jobId}`);
  const job = await res.json();
  updateSteps(job.logs || []);
  if (job.status === "done") {
    setStatus("分析完成", "done");
    state.result = job.result;
    state.frameIndex = 0;
    state.playing = false;
    updateResultUi();
    setupMap();
    await loadFrame(0);
    render();
    return;
  }
  if (job.status === "error") {
    setStatus(job.error || "分析失败", "error");
    return;
  }
  setStatus("分析中", "running");
  setTimeout(pollJob, 1200);
}

function setStatus(text, mode) {
  el("statusText").textContent = text;
  const dot = el("statusDot");
  dot.className = `dot ${mode || "idle"}`;
}

function updateSteps(logs) {
  const list = el("stepList");
  list.innerHTML = "";
  const items = logs.slice(-8);
  for (const line of items) {
    const div = document.createElement("div");
    div.textContent = line.replace(/^\[[^\]]+\]\s*/, "");
    list.appendChild(div);
  }
}

function updateResultUi() {
  const summary = state.result?.summary || {};
  const map = state.result?.map || {};
  el("mapTitle").textContent = `${map.village_name || "村庄"} | ${map.scenario_name || "情景"}`;
  el("mapSubTitle").textContent = `首次受淹 ${map.first_flood_time || "未受淹"}，最近断面 ${map.nearest_section_id || "-"}`;
  el("firstFlood").textContent = map.first_flood_time || "未受淹";
  el("nearestSection").textContent = map.nearest_section_id || "-";
  el("outputPath").textContent = state.result?.xlsx_path || "-";
  const frames = map.frames || [];
  el("frameSlider").max = Math.max(frames.length - 1, 0);
  el("frameSlider").value = 0;
  updateFrameLabel();
  if (summary["校正后河道路径"]) {
    el("configHint").textContent = `校正河道：${summary["校正后河道路径"]}`;
  }
}

function setupMap() {
  const b = collectBounds();
  state.homeExtent = padBounds(b, 0.07);
  state.extent = state.homeExtent;
}

function collectBounds() {
  const bounds = [];
  const map = state.result?.map || {};
  for (const key of ["buildings", "reference_buildings", "sections", "rivers", "corrected_rivers"]) {
    collectGeoJsonBounds(map[key], bounds);
  }
  if (!bounds.length) return [0, 0, 100, 100];
  return [
    Math.min(...bounds.map((b) => b[0])),
    Math.min(...bounds.map((b) => b[1])),
    Math.max(...bounds.map((b) => b[2])),
    Math.max(...bounds.map((b) => b[3])),
  ];
}

function collectGeoJsonBounds(geojson, bounds) {
  for (const feature of geojson?.features || []) {
    const b = geometryBounds(feature.geometry);
    if (b) bounds.push(b);
  }
}

function geometryBounds(geom) {
  if (!geom) return null;
  const xs = [];
  const ys = [];
  walkCoords(geom.coordinates, (x, y) => {
    xs.push(x);
    ys.push(y);
  });
  if (!xs.length) return null;
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function walkCoords(coords, fn) {
  if (!Array.isArray(coords)) return;
  if (typeof coords[0] === "number" && typeof coords[1] === "number") {
    fn(coords[0], coords[1]);
    return;
  }
  for (const item of coords) walkCoords(item, fn);
}

function padBounds(bounds, ratio) {
  const [minx, miny, maxx, maxy] = bounds;
  const dx = Math.max(maxx - minx, 1);
  const dy = Math.max(maxy - miny, 1);
  const pad = Math.max(dx, dy) * ratio;
  return [minx - pad, miny - pad, maxx + pad, maxy + pad];
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width * scale));
  const height = Math.max(240, Math.floor(rect.height * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function worldToScreen(x, y) {
  const [minx, miny, maxx, maxy] = state.extent || [0, 0, 1, 1];
  const sx = ((x - minx) / (maxx - minx)) * canvas.width;
  const sy = canvas.height - ((y - miny) / (maxy - miny)) * canvas.height;
  return [sx, sy];
}

function screenToWorld(sx, sy) {
  const [minx, miny, maxx, maxy] = state.extent || [0, 0, 1, 1];
  return [minx + (sx / canvas.width) * (maxx - minx), miny + ((canvas.height - sy) / canvas.height) * (maxy - miny)];
}

async function loadFrame(index) {
  if (!state.jobId || state.frameCells.has(index)) return;
  const res = await fetch(`/api/job/${state.jobId}/frame/${index}`);
  if (!res.ok) return;
  state.frameCells.set(index, await res.json());
}

async function setFrame(index) {
  const frames = state.result?.map?.frames || [];
  if (!frames.length) return;
  state.frameIndex = Math.max(0, Math.min(index, frames.length - 1));
  el("frameSlider").value = state.frameIndex;
  await loadFrame(state.frameIndex);
  updateFrameLabel();
  render();
}

function updateFrameLabel() {
  const frames = state.result?.map?.frames || [];
  if (!frames.length) {
    el("frameLabel").textContent = "无动画帧";
    return;
  }
  const frame = frames[state.frameIndex] || frames[0];
  const maxValue = Number.isFinite(frame.max_value) ? frame.max_value.toFixed(3) : "-";
  el("frameLabel").textContent = `${frame.label || "-"} (${state.frameIndex + 1}/${frames.length}) | 最大值 ${maxValue}`;
}

async function playLoop() {
  if (!state.playing) return;
  const frames = state.result?.map?.frames || [];
  if (!frames.length) return;
  await setFrame((state.frameIndex + 1) % frames.length);
  setTimeout(playLoop, 700);
}

function render() {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f6f8fa";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!state.result || !state.extent) {
    drawEmptyState();
    return;
  }
  drawFlood();
  if (state.showOriginalRiver) drawGeoJson(state.result.map.rivers, { stroke: "#0077b6", width: 2 });
  if (state.showCorrectedRiver) drawCorrectedRivers();
  drawGeoJson(state.result.map.buildings, { stroke: "#25313d", width: 1, fill: "rgba(255, 248, 207, 0.38)" });
  drawGeoJson(state.result.map.reference_buildings, { stroke: "#8e44ad", width: 3 });
  if (state.showSections) drawSections();
}

function drawEmptyState() {
  ctx.fillStyle = "#667483";
  ctx.font = `${15 * (window.devicePixelRatio || 1)}px Microsoft YaHei UI`;
  ctx.textAlign = "center";
  ctx.fillText("开始分析后显示地图动画", canvas.width / 2, canvas.height / 2);
}

function drawFlood() {
  const frame = state.frameCells.get(state.frameIndex);
  if (!frame) return;
  const threshold = Number(state.result?.map?.threshold || 0);
  for (const cell of frame.cells || []) {
    const [x0, y0, x1, y1, value] = cell;
    if (value < threshold && !state.showShallow) continue;
    const a = worldToScreen(x0, y0);
    const b = worldToScreen(x1, y1);
    ctx.fillStyle = value >= threshold ? "rgba(47, 128, 237, 0.82)" : "rgba(169, 214, 255, 0.68)";
    ctx.fillRect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]) + 1, Math.abs(b[1] - a[1]) + 1);
  }
}

function drawGeoJson(geojson, style) {
  for (const feature of geojson?.features || []) {
    drawGeometry(feature.geometry, style);
    if (style.label) drawFeatureLabel(feature, style);
  }
}

function drawCorrectedRivers() {
  const geojson = state.result?.map?.corrected_rivers;
  const frame = state.frameCells.get(state.frameIndex) || {};
  const values = frame.corrected_river_values || {};
  for (const feature of geojson?.features || []) {
    drawGeometry(feature.geometry, { stroke: "#00a676", width: 3 });
    const props = feature.properties || {};
    const riverId = String(props.river_name || props.river_folder || feature.id || "");
    const value = values[riverId]?.max;
    const label = Number.isFinite(value) ? `校正河道 | 水深${value.toFixed(2)}m` : "校正河道";
    drawFeatureLabel(feature, { label, color: "#006d4f" });
  }
}

function drawGeometry(geom, style) {
  if (!geom) return;
  if (geom.type === "Polygon") drawPolygon(geom.coordinates, style);
  else if (geom.type === "MultiPolygon") for (const poly of geom.coordinates) drawPolygon(poly, style);
  else if (geom.type === "LineString") drawLine(geom.coordinates, style);
  else if (geom.type === "MultiLineString") for (const line of geom.coordinates) drawLine(line, style);
  else if (geom.type === "Point") drawPoint(geom.coordinates, style);
}

function drawLine(coords, style) {
  if (!coords || coords.length < 2) return;
  ctx.beginPath();
  coords.forEach(([x, y], i) => {
    const p = worldToScreen(x, y);
    if (i === 0) ctx.moveTo(p[0], p[1]);
    else ctx.lineTo(p[0], p[1]);
  });
  ctx.strokeStyle = style.stroke || "#333";
  ctx.lineWidth = (style.width || 1) * (window.devicePixelRatio || 1);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();
}

function drawPolygon(rings, style) {
  if (!rings || !rings.length) return;
  ctx.beginPath();
  for (const ring of rings) {
    ring.forEach(([x, y], i) => {
      const p = worldToScreen(x, y);
      if (i === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    });
    ctx.closePath();
  }
  if (style.fill) {
    ctx.fillStyle = style.fill;
    ctx.fill();
  }
  ctx.strokeStyle = style.stroke || "#333";
  ctx.lineWidth = (style.width || 1) * (window.devicePixelRatio || 1);
  ctx.stroke();
}

function drawPoint(coords, style) {
  const p = worldToScreen(coords[0], coords[1]);
  ctx.beginPath();
  ctx.arc(p[0], p[1], 4 * (window.devicePixelRatio || 1), 0, Math.PI * 2);
  ctx.fillStyle = style.fill || style.stroke || "#00a676";
  ctx.fill();
}

function drawSections() {
  const nearest = String(state.result?.map?.nearest_section_id || "");
  for (const feature of state.result?.map?.sections?.features || []) {
    const id = String(feature.properties?.section_id || "");
    const isNearest = id === nearest;
    drawGeometry(feature.geometry, { stroke: isNearest ? "#c0392b" : "#d99020", width: isNearest ? 3 : 2 });
    drawSectionLabel(feature, isNearest);
  }
}

function drawSectionLabel(feature, isNearest) {
  const p = geometryMidpoint(feature.geometry);
  if (!p) return;
  const screen = worldToScreen(p[0], p[1]);
  const props = feature.properties || {};
  const fullId = String(props.section_id || "");
  const id = shortSectionId(fullId);
  const length = Number(props.trimmed_length_m);
  const depth = Number(props.section_original_depth_m);
  const frame = state.frameCells.get(state.frameIndex) || {};
  const sectionValues = frame.section_values || {};
  const waterDepth = sectionValues[fullId]?.max ?? sectionValues[id]?.max;
  const parts = [id];
  if (Number.isFinite(length)) parts.push(`长${length.toFixed(0)}m`);
  if (Number.isFinite(depth)) parts.push(`原深${depth.toFixed(2)}m`);
  if (Number.isFinite(waterDepth)) parts.push(`水深${waterDepth.toFixed(2)}m`);
  drawHaloText(parts.join(" | "), screen[0] + 5, screen[1] - 5, isNearest ? "#8b1e12" : "#6b4300", "left");
}

function drawFeatureLabel(feature, style) {
  const p = geometryMidpoint(feature.geometry);
  if (!p) return;
  const screen = worldToScreen(p[0], p[1]);
  drawHaloText(style.label, screen[0], screen[1], style.color || "#006d4f", "center");
}

function geometryMidpoint(geom) {
  const pts = [];
  walkCoords(geom?.coordinates, (x, y) => pts.push([x, y]));
  if (!pts.length) return null;
  return pts[Math.floor(pts.length / 2)];
}

function shortSectionId(id) {
  const parts = id.split("_");
  return parts[parts.length - 1] || id;
}

function drawHaloText(text, x, y, fill, align) {
  const scale = window.devicePixelRatio || 1;
  ctx.font = `${12 * scale}px Microsoft YaHei UI`;
  ctx.textAlign = align || "center";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 4 * scale;
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = fill;
  ctx.fillText(text, x, y);
}

function zoom(factor, sx = canvas.width / 2, sy = canvas.height / 2) {
  if (!state.extent) return;
  const [wx, wy] = screenToWorld(sx, sy);
  const [minx, miny, maxx, maxy] = state.extent;
  const w = (maxx - minx) * factor;
  const h = (maxy - miny) * factor;
  const rx = (wx - minx) / (maxx - minx);
  const ry = (wy - miny) / (maxy - miny);
  state.extent = [wx - rx * w, wy - ry * h, wx + (1 - rx) * w, wy + (1 - ry) * h];
  render();
}

function bindEvents() {
  el("reloadConfig").addEventListener("click", loadConfig);
  el("runAnalysis").addEventListener("click", runAnalysis);
  el("zoomIn").addEventListener("click", () => zoom(0.75));
  el("zoomOut").addEventListener("click", () => zoom(1.35));
  el("fitView").addEventListener("click", () => {
    state.extent = state.homeExtent;
    render();
  });
  el("togglePlay").addEventListener("click", () => {
    state.playing = !state.playing;
    el("togglePlay").textContent = state.playing ? "暂停" : "播放";
    if (state.playing) playLoop();
  });
  el("frameSlider").addEventListener("input", (e) => setFrame(Number(e.target.value)));
  el("showShallow").addEventListener("change", (e) => {
    state.showShallow = e.target.checked;
    render();
  });
  el("showOriginalRiver").addEventListener("change", (e) => {
    state.showOriginalRiver = e.target.checked;
    render();
  });
  el("showCorrectedRiver").addEventListener("change", (e) => {
    state.showCorrectedRiver = e.target.checked;
    render();
  });
  el("showSections").addEventListener("change", (e) => {
    state.showSections = e.target.checked;
    render();
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoom(e.deltaY < 0 ? 0.82 : 1.22, e.offsetX * (window.devicePixelRatio || 1), e.offsetY * (window.devicePixelRatio || 1));
  });
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    canvas.classList.add("dragging");
    state.drag = { x: e.clientX, y: e.clientY, extent: [...(state.extent || [0, 0, 1, 1])] };
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!state.drag || !state.extent) return;
    const scale = window.devicePixelRatio || 1;
    const dx = (e.clientX - state.drag.x) * scale;
    const dy = (e.clientY - state.drag.y) * scale;
    const [minx, miny, maxx, maxy] = state.drag.extent;
    const worldDx = (-dx / canvas.width) * (maxx - minx);
    const worldDy = (dy / canvas.height) * (maxy - miny);
    state.extent = [minx + worldDx, miny + worldDy, maxx + worldDx, maxy + worldDy];
    render();
  });
  canvas.addEventListener("pointerup", () => {
    canvas.classList.remove("dragging");
    state.drag = null;
  });
  window.addEventListener("resize", render);
}

bindEvents();
loadConfig().then(render);
