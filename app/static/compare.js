// Auto-design & compare: prompt + scan -> 3 candidate sockets with trade-offs,
// each in a live 3D viewer that toggles between the solid socket and a
// rectification HEATMAP on the prosthesis (blue = carved in / loaded,
// red = relieved out) — the trade-off made visible.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (s) => document.querySelector(s);
const loader = new GLTFLoader();
const viewers = [];

function makeViewer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  const scene = new THREE.Scene(); // transparent -> CSS gradient shows through
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.1;
  scene.add(new THREE.HemisphereLight(0xffffff, 0xbcd3ff, 1.25));
  const key = new THREE.DirectionalLight(0xffffff, 1.5);
  key.position.set(1, 1.2, 1);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xeaf2ff, 0.6);
  fill.position.set(-1, -0.5, -1);
  scene.add(fill);
  const v = { renderer, scene, camera, controls, canvas, model: null, framed: false };
  viewers.push(v);
  resize(v);
  return v;
}

function resize(v) {
  const w = v.canvas.clientWidth || 320;
  const h = v.canvas.clientHeight || 280;
  v.renderer.setSize(w, h, false);
  v.camera.aspect = w / h;
  v.camera.updateProjectionMatrix();
}

function loadModel(v, url, { color = null, vertexColors = false } = {}) {
  loader.load(url, (gltf) => {
    if (v.model) v.scene.remove(v.model);
    const root = gltf.scene;
    root.traverse((o) => {
      if (o.isMesh) {
        o.material = new THREE.MeshStandardMaterial({
          color: vertexColors ? 0xffffff : color,
          vertexColors,
          metalness: 0.05,
          roughness: 0.7,
          side: THREE.DoubleSide,
          flatShading: false,
        });
      }
    });
    const box = new THREE.Box3().setFromObject(root);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    root.position.sub(center);
    v.model = root;
    v.scene.add(root);
    if (!v.framed) {
      const radius = Math.max(size.x, size.y, size.z);
      v.camera.position.set(radius * 1.1, radius * 0.35, radius * 1.7);
      v.controls.target.set(0, 0, 0);
      v.controls.update();
      v.framed = true;
    }
  });
}

function animate() {
  requestAnimationFrame(animate);
  for (const v of viewers) {
    v.controls.update();
    v.renderer.render(v.scene, v.camera);
  }
}
animate();
window.addEventListener("resize", () => viewers.forEach(resize));

const COLORS = { stability: 0xe07a5f, balanced: 0x69b48f, comfort: 0x5b8def };
let sourceMap = {}; // key -> {label, doi, url}, populated per response

function metricRow(label, value, sub) {
  return `<div class="m"><span>${label}</span><b>${value}</b>${sub || ""}</div>`;
}

function renderCard(c) {
  const m = c.metrics;
  const printable = m.printable
    ? '<span class="badge ok">printable ✓</span>'
    : '<span class="badge no">not printable</span>';
  const snugPct = Math.round(m.snugness * 100);
  const vmax = c.heatmap_vmax_mm;

  const card = document.createElement("div");
  card.className = "card " + (m.validation_passed ? "pass" : "fail");
  card.innerHTML = `
    <div class="viewwrap">
      <canvas class="viewer"></canvas>
      <div class="viewtoggle">
        <button data-v="heatmap" class="on">Heatmap</button>
        <button data-v="socket">Socket</button>
      </div>
    </div>
    <div class="legend">
      <span>carve in <b>−${vmax}mm</b></span>
      <span class="grad"></span>
      <span><b>+${vmax}mm</b> relief out</span>
    </div>
    <div class="annot">
      ${c.annotations
        .map(
          (a) =>
            `<span class="chip ${a.action}">${a.zone}: ${
              a.action === "carve" ? "carve" : "relief"
            } ${a.mm}mm</span>`
        )
        .join("")}
    </div>
    <h3>${c.title}</h3>
    <p class="sub">${c.summary}</p>
    <div class="metrics">
      ${metricRow("Philosophy", m.philosophy)}
      ${metricRow("Volume reduction", m.volume_reduction_pct + "%")}
      ${metricRow("Snugness", snugPct + "%", `<div class="bar"><span style="width:${snugPct}%"></span></div>`)}
      ${metricRow("Max carve / relief", m.max_carve_mm + " / " + m.max_relief_mm + " mm")}
      ${metricRow("Wall (mean)", (m.wall_mean_mm ?? "–") + " mm", `<div class="bar"><span style="width:${m.wall_in_tol_pct}%"></span></div>`)}
      ${metricRow("Socket volume", m.socket_volume_cm3 + " cm³", `<div>${printable}</div>`)}
    </div>
    <div class="proscons">
      <ul class="pro">${c.pros.map((x) => `<li>${x}</li>`).join("")}</ul>
      <ul class="con">${c.cons.map((x) => `<li>${x}</li>`).join("")}</ul>
    </div>
    <details class="grounded">
      <summary>Why this design — grounded <span class="vbadge">✓ evidence</span></summary>
      <ul>
        ${(c.grounding || [])
          .map((g) => {
            const s = sourceMap[g.source];
            const cite = s
              ? ` <a href="${s.doi ? "https://doi.org/" + s.doi : s.url}" target="_blank" rel="noopener">[${g.source}]</a>`
              : "";
            return `<li>${g.text}${cite}</li>`;
          })
          .join("")}
      </ul>
    </details>
    <div class="actions">
      <a class="dl" href="${c.stl_url}" download>⬇ Download STL</a>
      <button class="log">Operation history (${c.processlog.entries.length})</button>
      ${c.log_verified ? '<span class="vbadge" title="Every value in this history is computed from the model and checked against an independent recomputation in the test suite — not hand-written.">✓ computed from model</span>' : ""}
    </div>
    <div class="history hidden"></div>
  `;

  // Operation history (Mimics/ScanIP-style).
  const hist = card.querySelector(".history");
  hist.innerHTML =
    "<ol>" +
    c.processlog.entries
      .map(
        (e) =>
          `<li><span class="cat">${e.category}</span><span class="op">${e.operation}</span>` +
          `<div class="dt">${e.detail}</div></li>`
      )
      .join("") +
    "</ol>";
  card.querySelector(".log").addEventListener("click", () =>
    hist.classList.toggle("hidden")
  );

  // Viewer + heatmap/socket toggle (heatmap shown first).
  const v = makeViewer(card.querySelector(".viewer"));
  loadModel(v, c.heatmap_glb, { vertexColors: true });
  const toggle = card.querySelector(".viewtoggle");
  toggle.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    if (!btn) return;
    toggle.querySelectorAll("button").forEach((b) => b.classList.remove("on"));
    btn.classList.add("on");
    if (btn.dataset.v === "socket")
      loadModel(v, c.socket_glb, { color: COLORS[c.key] || 0x9aa5b1 });
    else loadModel(v, c.heatmap_glb, { vertexColors: true });
  });
  return card;
}

async function run() {
  const btn = $("#run");
  const status = $("#status");
  btn.disabled = true;
  status.textContent = "Sampling scan & auto-designing 3 candidates…";
  viewers.length = 0;
  $("#cards").innerHTML = "";
  try {
    const res = await fetch("/api/autodesign", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: $("#prompt").value,
        shape: $("#scan").value,
        is_left_limb: $("#isleft").checked,
      }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const intents = data.intents.length
      ? data.intents.map((i) => `<span class="intent">${i}</span>`).join("")
      : '<span class="intent">none detected — neutral defaults</span>';
    const info = $("#scaninfo");
    info.classList.remove("hidden");
    info.innerHTML =
      `<strong>Scan:</strong> ${data.scan.source} (${data.scan.version}), ` +
      `${data.scan.length_mm} mm, ${data.scan.n_vertices.toLocaleString()} vertices · ` +
      `<strong>Detected intent:</strong> ${intents}`;

    // Build the source lookup so per-design grounding can deep-link to papers.
    sourceMap = {};
    for (const s of data.sources || []) sourceMap[s.key] = s;

    // Sources & grounding panel (collapsible).
    const srcEl = $("#sources");
    srcEl.classList.remove("hidden");
    srcEl.innerHTML =
      `<details open><summary>Sources &amp; grounding ` +
      `<span class="vbadge">✓ verified</span></summary>` +
      `<p class="note">${data.grounding_note || ""}</p>` +
      "<ul>" +
      (data.sources || [])
        .map(
          (s) =>
            `<li><a href="${s.url}" target="_blank" rel="noopener">${s.label}</a>` +
            (s.doi ? ` · <a href="https://doi.org/${s.doi}" target="_blank" rel="noopener">doi:${s.doi}</a>` : "") +
            `<div class="grounds">${s.grounds}</div></li>`
        )
        .join("") +
      "</ul></details>";

    const cards = $("#cards");
    for (const c of data.candidates) cards.appendChild(renderCard(c));
    viewers.forEach(resize);
    status.textContent = "Done — drag a model to rotate; toggle Heatmap/Socket.";
  } catch (e) {
    status.textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

$("#run").addEventListener("click", run);
document.querySelectorAll(".ex").forEach((b) =>
  b.addEventListener("click", () => ($("#prompt").value = b.dataset.ex))
);
