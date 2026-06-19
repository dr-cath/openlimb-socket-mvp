// Auto-design & compare: prompt + scan -> 3 candidate sockets with trade-offs,
// each in its own live 3D viewer, plus a Mimics/ScanIP-style operation history.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (s) => document.querySelector(s);
const loader = new GLTFLoader();
const viewers = []; // keep references so resize/animate keeps working

function makeViewer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1116);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(1, 1, 1);
  scene.add(key);
  const v = { renderer, scene, camera, controls, canvas, model: null };
  viewers.push(v);
  resize(v);
  return v;
}

function resize(v) {
  const w = v.canvas.clientWidth || 300;
  const h = v.canvas.clientHeight || 260;
  v.renderer.setSize(w, h, false);
  v.camera.aspect = w / h;
  v.camera.updateProjectionMatrix();
}

function loadModel(v, url, color) {
  loader.load(url, (gltf) => {
    if (v.model) v.scene.remove(v.model);
    const root = gltf.scene;
    root.traverse((o) => {
      if (o.isMesh) {
        o.material = new THREE.MeshStandardMaterial({
          color, metalness: 0.1, roughness: 0.75, side: THREE.DoubleSide,
        });
      }
    });
    // Center & frame the model.
    const box = new THREE.Box3().setFromObject(root);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    root.position.sub(center);
    v.model = root;
    v.scene.add(root);
    const radius = Math.max(size.x, size.y, size.z);
    v.camera.position.set(radius * 1.1, radius * 0.4, radius * 1.6);
    v.controls.target.set(0, 0, 0);
    v.controls.update();
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

const COLORS = { stability: 0xe07a5f, balanced: 0x81b29a, comfort: 0x6699cc };

function metricRow(label, value, sub) {
  return `<div class="m"><span>${label}</span><b>${value}</b>${sub || ""}</div>`;
}

function renderCard(c) {
  const m = c.metrics;
  const printable = m.printable
    ? '<span class="badge ok">printable ✓</span>'
    : '<span class="badge no">not printable</span>';
  const snugPct = Math.round(m.snugness * 100);
  const card = document.createElement("div");
  card.className = "card " + (m.validation_passed ? "pass" : "fail");
  card.innerHTML = `
    <canvas class="viewer"></canvas>
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
    <div class="actions">
      <a class="dl" href="${c.stl_url}" download>⬇ Download STL</a>
      <button class="log">Operation history (${c.processlog.entries.length})</button>
    </div>
    <div class="history hidden"></div>
  `;
  // Wire the operation-history toggle.
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
  // 3D viewer for this candidate.
  const v = makeViewer(card.querySelector(".viewer"));
  loadModel(v, c.socket_glb, COLORS[c.key] || 0x9aa5b1);
  return card;
}

async function run() {
  const btn = $("#run");
  const status = $("#status");
  btn.disabled = true;
  status.textContent = "Sampling scan & auto-designing 3 candidates…";
  // Reset viewers/cards.
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
      : '<span class="intent">none detected — using neutral defaults</span>';
    const info = $("#scaninfo");
    info.classList.remove("hidden");
    info.innerHTML =
      `<strong>Scan:</strong> ${data.scan.source} (${data.scan.version}), ` +
      `${data.scan.length_mm} mm, ${data.scan.n_vertices.toLocaleString()} vertices · ` +
      `<strong>Detected intent:</strong> ${intents}`;

    const cards = $("#cards");
    for (const c of data.candidates) cards.appendChild(renderCard(c));
    viewers.forEach(resize);
    status.textContent = "Done — drag any model to rotate.";
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
