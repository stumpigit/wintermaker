import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
const tileSelect = document.getElementById("tile-select");
const textureSelect = document.getElementById("texture-select");
const exaggerationInput = document.getElementById("exaggeration");
const exaggerationValue = document.getElementById("exaggeration-value");

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd8dde3);

const camera = new THREE.PerspectiveCamera(
  50,
  window.innerWidth / window.innerHeight,
  0.5,
  50000,
);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.maxPolarAngle = Math.PI / 2.02;
controls.minDistance = 20;
controls.maxDistance = 12000;

let terrainMesh = null;
let baseHeights = null;
let sceneMeta = null;
const textureCache = new Map();

function createTerrainMaterial(texture) {
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.flipY = true;
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.anisotropy = Math.min(renderer.capabilities.getMaxAnisotropy(), 4);

  return new THREE.MeshBasicMaterial({
    map: texture,
    side: THREE.DoubleSide,
    transparent: false,
    opacity: 1,
    alphaTest: 0,
    toneMapped: false,
    depthWrite: true,
    depthTest: true,
  });
}

async function discoverTiles() {
  const params = new URLSearchParams(window.location.search);
  const requested = params.get("tile");

  let tiles = [];
  try {
    const manifest = await fetch("data/manifest.json");
    if (manifest.ok) {
      const data = await manifest.json();
      tiles = data.tiles || [];
    }
  } catch {
    /* manifest optional */
  }

  if (requested && !tiles.includes(requested)) {
    tiles.unshift(requested);
  }
  if (tiles.length === 0) {
    tiles = ["demo_test_001"];
  }

  tileSelect.replaceChildren(
    ...tiles.map((tile) => {
      const option = document.createElement("option");
      option.value = tile;
      option.textContent = tile;
      return option;
    }),
  );

  const initial = requested && tiles.includes(requested) ? requested : tiles[0];
  tileSelect.value = initial;
  return initial;
}

async function loadBinary(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Datei nicht gefunden: ${url}`);
  }
  return response.arrayBuffer();
}

async function loadTexture(url) {
  if (textureCache.has(url)) {
    return textureCache.get(url);
  }
  const loader = new THREE.TextureLoader();
  const texture = await loader.loadAsync(url);
  textureCache.set(url, texture);
  return texture;
}

function applyExaggeration(factor) {
  if (!terrainMesh || !baseHeights) {
    return;
  }
  const positions = terrainMesh.geometry.attributes.position.array;
  for (let vi = 0; vi < baseHeights.length; vi++) {
    positions[vi * 3 + 1] = baseHeights[vi] * factor;
  }
  terrainMesh.geometry.attributes.position.needsUpdate = true;
  terrainMesh.geometry.computeBoundingBox();
  terrainMesh.geometry.computeBoundingSphere();
}

function frameCamera(mesh) {
  const box = new THREE.Box3().setFromObject(mesh);
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);

  controls.target.copy(center);

  const maxDim = Math.max(size.x, size.y, size.z);
  camera.near = Math.max(maxDim / 5000, 0.5);
  camera.far = maxDim * 20;
  camera.updateProjectionMatrix();

  camera.position.set(
    center.x + maxDim * 0.55,
    center.y + maxDim * 0.45,
    center.z + maxDim * 0.65,
  );
  controls.update();
}

function disposeTerrain() {
  if (!terrainMesh) {
    return;
  }
  scene.remove(terrainMesh);
  terrainMesh.geometry.dispose();
  terrainMesh.material.map?.dispose();
  terrainMesh.material.dispose();
  terrainMesh = null;
  baseHeights = null;
}

async function loadTile(tileId) {
  statusEl.textContent = `Lade ${tileId} …`;
  textureCache.clear();
  disposeTerrain();

  const base = `data/${tileId}`;
  const metaResponse = await fetch(`${base}/scene.json`);
  if (!metaResponse.ok) {
    throw new Error(
      `Keine Viewer-Daten für „${tileId}“. Bitte zuerst exportieren:\n` +
        `winter-ortho viewer-export --tile-id ${tileId}`,
    );
  }
  sceneMeta = await metaResponse.json();

  const [positionsBuf, uvsBuf, indicesBuf, winterTexture] = await Promise.all([
    loadBinary(`${base}/${sceneMeta.files.positions}`),
    loadBinary(`${base}/${sceneMeta.files.uvs}`),
    loadBinary(`${base}/${sceneMeta.files.indices}`),
    loadTexture(`${base}/${sceneMeta.textures.winter}`),
  ]);

  const positions = new Float32Array(positionsBuf);
  const uvs = new Float32Array(uvsBuf);
  const IndexArray = sceneMeta.index_dtype === "uint16" ? Uint16Array : Uint32Array;
  const rawIndices = new IndexArray(indicesBuf);
  const indices = new Uint32Array(rawIndices.length);
  for (let i = 0; i < rawIndices.length; i++) {
    indices[i] = rawIndices[i];
  }

  baseHeights = new Float32Array(positions.length / 3);
  for (let vi = 0; vi < baseHeights.length; vi++) {
    baseHeights[vi] = positions[vi * 3 + 1];
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(new THREE.Uint32BufferAttribute(indices, 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  const material = createTerrainMaterial(winterTexture);
  terrainMesh = new THREE.Mesh(geometry, material);
  terrainMesh.frustumCulled = false;
  scene.add(terrainMesh);

  const exaggeration = parseFloat(exaggerationInput.value);
  applyExaggeration(exaggeration);
  frameCamera(terrainMesh);

  textureSelect.replaceChildren();
  const winterOption = document.createElement("option");
  winterOption.value = "winter";
  winterOption.textContent = "Winter";
  textureSelect.appendChild(winterOption);
  if (sceneMeta.textures.summer) {
    const summerOption = document.createElement("option");
    summerOption.value = "summer";
    summerOption.textContent = "Sommer";
    textureSelect.appendChild(summerOption);
  }
  textureSelect.value = "winter";

  const triangles = Math.floor(sceneMeta.index_count / 3).toLocaleString("de-CH");
  const vertices = sceneMeta.vertex_count.toLocaleString("de-CH");
  const texInfo = sceneMeta.texture_width
    ? ` · Textur ${sceneMeta.texture_width}×${sceneMeta.texture_height}`
    : "";
  statusEl.textContent = `${tileId} · ${vertices} Punkte · ${triangles} Dreiecke${texInfo}`;
}

async function switchTexture(kind) {
  if (!terrainMesh || !sceneMeta) {
    return;
  }
  const file = sceneMeta.textures[kind];
  if (!file) {
    return;
  }
  const oldTexture = terrainMesh.material.map;
  const texture = await loadTexture(`data/${sceneMeta.tile_id}/${file}`);
  terrainMesh.material.dispose();
  terrainMesh.material = createTerrainMaterial(texture);
  if (oldTexture && oldTexture !== texture) {
    oldTexture.dispose();
  }
}

tileSelect.addEventListener("change", () => {
  loadTile(tileSelect.value).catch((error) => {
    statusEl.textContent = error.message;
    console.error(error);
  });
});

textureSelect.addEventListener("change", () => {
  switchTexture(textureSelect.value).catch(console.error);
});

exaggerationInput.addEventListener("input", () => {
  const value = parseFloat(exaggerationInput.value);
  exaggerationValue.textContent = `${value.toFixed(1)}×`;
  applyExaggeration(value);
  frameCamera(terrainMesh);
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

discoverTiles()
  .then((tileId) => loadTile(tileId))
  .then(() => animate())
  .catch((error) => {
    statusEl.textContent = error.message;
    console.error(error);
  });
