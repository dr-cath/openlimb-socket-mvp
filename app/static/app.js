import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

const viewerEl = document.getElementById("viewer");
const loader = new PLYLoader();

// ---- three.js scene ------------------------------------------------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1419);
const camera = new THREE.PerspectiveCamera(45, 1, 1, 5000);
camera.position.set(300, 120, 300);
const renderer = new THREE.WebGLRenderer({ antialias: true });
viewerEl.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 0.8);
key.position.set(1, 1, 1);
scene.add(key);
const back = new THREE.DirectionalLight(0xffffff, 0.4);
back.position.set(-1, -0.5, -1);
scene.add(back);
const markers = new THREE.Group();
scene.add(markers);

function resize() {
  const w = viewerEl.clientWidth, h = viewerEl.clientHeight;
  renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }

// ---- mesh management -----------------------------------------------------
const meshes = { limb: null, rectified: null, socket: null, heatmap: null };
let currentView = "limb";
let limbMesh = null; // THREE.Mesh used for raycasting / painting

function disposeMesh(m) { if (m) { m.geometry.dispose(); m.material.dispose(); scene.remove(m); } }

function frameCamera(geometry) {
  geometry.computeBoundingSphere();
  const s = geometry.boundingSphere;
  controls.target.copy(s.center);
  camera.position.set(s.center.x + s.radius * 2, s.center.y + s.radius * 0.6, s.center.z + s.radius * 2);
  camera.near = s.radius / 100; camera.far = s.radius * 20; camera.updateProjectionMatrix();
}

async function loadPly(url, colored) {
  const buf = await (await fetch(url + "?t=" + Date.now())).arrayBuffer();
  const geom = loader.parse(buf);
  geom.computeVertexNormals();
  const mat = colored
    ? new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, flatShading: false })
    : new THREE.MeshStandardMaterial({ color: 0xb9c4d0, roughness: 0.8, metalness: 0.0, flatShading: false });
  return new THREE.Mesh(geom, mat);
}

function showView(view) {
  currentView = view;
  for (const k of Object.keys(meshes)) if (meshes[k]) meshes[k].visible = (k === view);
  document.querySelectorAll("#tabs button").forEach(b =>
    b.classList.toggle("active", b.dataset.view === view));
  document.getElementById("legend").style.visibility = view === "heatmap" ? "visible" : "hidden";
}

async function setLimb(data) {
  for (const k of Object.keys(meshes)) { disposeMesh(meshes[k]); meshes[k] = null; }
  markers.clear();
  limbMesh = await loadPly(data.limb_url, false);
  meshes.limb = limbMesh; scene.add(limbMesh);
  frameCamera(limbMesh.geometry);
  showView("limb");
  document.getElementById("limbinfo").textContent =
    `${data.info.source} · ${data.n_vertices} verts · length ${data.length_mm.toFixed(0)} mm`;
  document.getElementById("generate").disabled = false;
  document.getElementById("export").disabled = true;
}

// ---- spec assembly -------------------------------------------------------
const patches = [];
function buildSpec() {
  const v = id => document.getElementById(id);
  return {
    philosophy: v("philosophy").value,
    snugness: parseFloat(v("snugness").value),
    is_left_limb: v("is_left").checked,
    relieve_fibular_head: v("relieve_fibular_head").checked,
    relieve_distal_tibial_end: v("relieve_distal_tibial_end").checked,
    relieve_tibial_crest: v("relieve_tibial_crest").checked,
    load_patellar_tendon: v("load_patellar_tendon").checked,
    load_posterior_calf: v("load_posterior_calf").checked,
    carve_paratibial: v("carve_paratibial").checked,
    carve_supracondylar: v("carve_supracondylar").checked,
    trimline_height_frac: parseFloat(v("trimline_height_frac").value),
    wall_thickness_mm: parseFloat(v("wall_thickness_mm").value),
    custom_regions: patches.map(p => ({
      point: p.point, radius_mm: p.radius, depth_mm: p.depth, carve: p.carve, label: p.label,
    })),
  };
}

function renderPatches() {
  const ul = document.getElementById("patches");
  ul.innerHTML = "";
  patches.forEach((p, i) => {
    const li = document.createElement("li");
    li.textContent = `${p.carve ? "carve" : "relief"} r${p.radius} d${p.depth}`;
    const x = document.createElement("button");
    x.textContent = "×"; x.className = "ghost";
    x.onclick = () => { patches.splice(i, 1); renderPatches(); rebuildMarkers(); };
    li.appendChild(x); ul.appendChild(li);
  });
}
function rebuildMarkers() {
  markers.clear();
  patches.forEach(p => {
    const s = new THREE.Mesh(new THREE.SphereGeometry(p.radius, 16, 12),
      new THREE.MeshBasicMaterial({ color: p.carve ? 0x3b6fd4 : 0xd4503b, transparent: true, opacity: 0.35 }));
    s.position.fromArray(p.point); markers.add(s);
  });
}

// ---- painting (raycast onto limb) ---------------------------------------
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
renderer.domElement.addEventListener("pointerdown", (e) => {
  if (!document.getElementById("paint").checked || !limbMesh || currentView !== "limb") return;
  const r = renderer.domElement.getBoundingClientRect();
  pointer.x = ((e.clientX - r.left) / r.width) * 2 - 1;
  pointer.y = -((e.clientY - r.top) / r.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(limbMesh)[0];
  if (!hit) return;
  patches.push({
    point: hit.point.toArray(),
    radius: parseFloat(document.getElementById("paint_radius").value),
    depth: parseFloat(document.getElementById("paint_depth").value),
    carve: document.getElementById("paint_carve").checked,
    label: "painted",
  });
  renderPatches(); rebuildMarkers();
  logEvent("paint_patch", { point: hit.point.toArray() });
});

// ---- API calls -----------------------------------------------------------
async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}) });
  return r.json();
}
function logEvent(event, payload) { post("/api/log", { event, ...payload }); }

document.querySelectorAll("button.sample").forEach(b => b.onclick = async () => {
  b.disabled = true;
  const data = await post("/api/sample", { shape: b.dataset.shape, is_left_limb: document.getElementById("is_left").checked });
  await setLimb(data); b.disabled = false;
});

document.getElementById("upload").onchange = async (e) => {
  const file = e.target.files[0]; if (!file) return;
  const fd = new FormData(); fd.append("file", file);
  const data = await (await fetch("/api/upload", { method: "POST", body: fd })).json();
  if (data.error) { alert(data.error); return; }
  await setLimb(data);
};

document.getElementById("generate").onclick = async () => {
  const btn = document.getElementById("generate");
  btn.disabled = true; btn.textContent = "Generating…";
  const data = await post("/api/generate", buildSpec());
  btn.textContent = "Generate"; btn.disabled = false;
  if (data.error) { alert(data.error); return; }
  disposeMesh(meshes.rectified); disposeMesh(meshes.socket); disposeMesh(meshes.heatmap);
  meshes.rectified = await loadPly(data.rectified_url, false); scene.add(meshes.rectified);
  meshes.socket = await loadPly(data.socket_url, false); scene.add(meshes.socket);
  meshes.heatmap = await loadPly(data.heatmap_url, true); scene.add(meshes.heatmap);
  renderValidation(data);
  showView("socket");
  document.getElementById("export").disabled = !data.validation.passed;
};

function renderValidation(data) {
  const el = document.getElementById("validation");
  const v = data.validation;
  const wt = data.validation.metrics.wall_thickness_mm;
  let rows = "";
  for (const [k, val] of Object.entries(v.checks))
    rows += `<tr><td>${k}</td><td class="${val ? "ok" : "fail"}">${val ? "✓" : "✗"}</td></tr>`;
  el.innerHTML =
    `<div class="${v.passed ? "ok" : "fail"}"><strong>${v.passed ? "PASS" : "FAIL"}</strong></div>` +
    `<table>${rows}</table>` +
    (wt ? `<p class="muted">wall ${wt.mean.toFixed(1)} mm (p05 ${wt.p05.toFixed(1)}–p95 ${wt.p95.toFixed(1)}); ` +
      `carve ${data.rectify.max_carve_mm.toFixed(1)} mm / relief ${data.rectify.max_relief_mm.toFixed(1)} mm; ` +
      `VR ${data.rectify.volume_reduction_pct}%</p>` : "");
  if (!v.passed && v.messages.length) el.innerHTML += `<p class="fail">${v.messages.join("<br>")}</p>`;
}

document.getElementById("export").onclick = async () => {
  const name = document.getElementById("exportname").value || "approved_socket";
  const data = await post("/api/export", { name, notes: "" });
  const el = document.getElementById("exportresult");
  el.innerHTML = data.stl_url
    ? `<a href="${data.stl_url}" download>Download ${name}.stl</a>`
    : `<span class="fail">${data.error}</span>`;
};

document.getElementById("clearpatches").onclick = () => { patches.length = 0; renderPatches(); rebuildMarkers(); };
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => { if (meshes[b.dataset.view]) showView(b.dataset.view); });
document.getElementById("snugness").addEventListener("input", e =>
  document.getElementById("snugval").textContent = e.target.value);

resize(); animate();
document.getElementById("snugval").textContent = "0.5";
document.getElementById("legend").style.visibility = "hidden";
