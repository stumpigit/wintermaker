import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
const tileSelect = document.getElementById("tile-select");
const textureSelect = document.getElementById("texture-select");
const elevationLabel = document.getElementById("elevation-label");
const elevationSelect = document.getElementById("elevation-select");
const exaggerationInput = document.getElementById("exaggeration");
const exaggerationValue = document.getElementById("exaggeration-value");

// Wintermittag ~11 Uhr, typisch für die Schweiz (tief stehende Sonne im Süden)
const WINTER_SUN_AZIMUTH_DEG = 172;
const WINTER_SUN_ALTITUDE_DEG = 28;

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();

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
let heightModels = null;
let activeElevationModel = "base";
let sceneMeta = null;
let trackLines = [];
let trackSources = [];
const textureCache = new Map();

function createWinterSky() {
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(1, 32, 16),
    new THREE.ShaderMaterial({
      uniforms: {
        topColor: { value: new THREE.Color(0x1e7fd4) },
        midColor: { value: new THREE.Color(0x4aa8e8) },
        horizonColor: { value: new THREE.Color(0xb8d9f2) },
        sunDirection: { value: new THREE.Vector3() },
      },
      vertexShader: `
        varying vec3 vWorldPosition;
        void main() {
          vec4 worldPosition = modelMatrix * vec4(position, 1.0);
          vWorldPosition = worldPosition.xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_Position.z = gl_Position.w;
        }
      `,
      fragmentShader: `
        uniform vec3 topColor;
        uniform vec3 midColor;
        uniform vec3 horizonColor;
        uniform vec3 sunDirection;
        varying vec3 vWorldPosition;
        void main() {
          vec3 direction = normalize(vWorldPosition - cameraPosition);
          float elevation = clamp(direction.y, 0.0, 1.0);
          vec3 skyColor = elevation > 0.55
            ? mix(midColor, topColor, (elevation - 0.55) / 0.45)
            : mix(horizonColor, midColor, elevation / 0.55);
          float sunGlow = pow(max(dot(direction, normalize(sunDirection)), 0.0), 256.0);
          skyColor += vec3(1.0, 0.95, 0.82) * sunGlow * 0.85;
          gl_FragColor = vec4(skyColor, 1.0);
        }
      `,
      side: THREE.BackSide,
      depthWrite: false,
      toneMapped: false,
    }),
  );
  sky.scale.setScalar(450000);
  sky.frustumCulled = false;
  return sky;
}

const sky = createWinterSky();
scene.add(sky);
const skyUniforms = sky.material.uniforms;

const hemisphereLight = new THREE.HemisphereLight(0x4aa3e8, 0xd8e8f5, 0.62);
scene.add(hemisphereLight);

const ambientLight = new THREE.AmbientLight(0xc8ddf5, 0.12);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xfff6e8, 2.6);
sunLight.target.position.set(0, 0, 0);
scene.add(sunLight);
scene.add(sunLight.target);

const sunDirection = new THREE.Vector3();
updateWinterSun();

function updateWinterSun(target = new THREE.Vector3(0, 0, 0)) {
  const phi = THREE.MathUtils.degToRad(90 - WINTER_SUN_ALTITUDE_DEG);
  const theta = THREE.MathUtils.degToRad(WINTER_SUN_AZIMUTH_DEG);
  sunDirection.setFromSphericalCoords(1, phi, theta);
  skyUniforms.sunDirection.value.copy(sunDirection);
  sunLight.position.copy(sunDirection).multiplyScalar(8000).add(target);
  sunLight.target.position.copy(target);
}

function createTerrainMaterial(texture) {
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.flipY = true;
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.anisotropy = renderer.capabilities.getMaxAnisotropy();

  return new THREE.MeshStandardMaterial({
    map: texture,
    roughness: 0.92,
    metalness: 0.0,
    side: THREE.DoubleSide,
    toneMapped: true,
  });
}

function extractHeights(positions) {
  const heights = new Float32Array(positions.length / 3);
  for (let vi = 0; vi < heights.length; vi++) {
    heights[vi] = positions[vi * 3 + 1];
  }
  return heights;
}

function updateTerrainNormals() {
  if (!terrainMesh) {
    return;
  }
  terrainMesh.geometry.computeVertexNormals();
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

function currentHeights() {
  if (!heightModels) {
    return null;
  }
  return heightModels[activeElevationModel] || heightModels.base;
}

function applyExaggeration(factor) {
  if (!terrainMesh) {
    return;
  }
  const heights = currentHeights();
  if (!heights) {
    return;
  }
  const positions = terrainMesh.geometry.attributes.position.array;
  for (let vi = 0; vi < heights.length; vi++) {
    positions[vi * 3 + 1] = heights[vi] * factor;
  }
  terrainMesh.geometry.attributes.position.needsUpdate = true;
  updateTerrainNormals();
  terrainMesh.geometry.computeBoundingBox();
  terrainMesh.geometry.computeBoundingSphere();
  updateTrackHeights(factor);
}

function frameCamera(mesh) {
  const box = new THREE.Box3().setFromObject(mesh);
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);

  controls.target.copy(center);
  updateWinterSun(center);

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

function disposeTracks() {
  for (const line of trackLines) {
    scene.remove(line);
    line.geometry.dispose();
    line.material.dispose();
  }
  trackLines = [];
  trackSources = [];
}

function disposeTerrain() {
  disposeTracks();
  if (!terrainMesh) {
    return;
  }
  scene.remove(terrainMesh);
  terrainMesh.geometry.dispose();
  terrainMesh.material.map?.dispose();
  terrainMesh.material.dispose();
  terrainMesh = null;
  heightModels = null;
}

function trackHeightsForModel(source) {
  if (activeElevationModel === "snow_surface" && source.snowHeights) {
    return source.snowHeights;
  }
  return source.heights;
}

function trackPointsFromSource(source, factor) {
  const heights = trackHeightsForModel(source);
  const points = [];
  for (let vi = 0; vi < heights.length; vi++) {
    points.push(source.x[vi], heights[vi] * factor, source.z[vi]);
  }
  return points;
}

function updateTrackHeights(factor) {
  for (let ti = 0; ti < trackLines.length; ti++) {
    const source = trackSources[ti];
    if (!source) {
      continue;
    }
    trackLines[ti].geometry.setPositions(trackPointsFromSource(source, factor));
    trackLines[ti].computeLineDistances();
  }
}

async function loadTracks(tileId) {
  disposeTracks();
  if (!sceneMeta?.tracks_file) {
    return;
  }

  const response = await fetch(`data/${tileId}/${sceneMeta.tracks_file}`);
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  const tracks = payload.tracks || [];
  if (tracks.length === 0) {
    return;
  }

  const resolution = new THREE.Vector2(window.innerWidth, window.innerHeight);

  for (const track of tracks) {
    const raw = track.positions;
    if (!raw || raw.length < 6) {
      continue;
    }

    const pointCount = raw.length / 3;
    const source = {
      x: new Float32Array(pointCount),
      heights: new Float32Array(pointCount),
      z: new Float32Array(pointCount),
      snowHeights: null,
    };
    for (let vi = 0; vi < pointCount; vi++) {
      source.x[vi] = raw[vi * 3];
      source.heights[vi] = raw[vi * 3 + 1];
      source.z[vi] = raw[vi * 3 + 2];
    }
    if (track.snow_positions?.length >= 6) {
      const snowRaw = track.snow_positions;
      const snowCount = snowRaw.length / 3;
      source.snowHeights = new Float32Array(snowCount);
      for (let vi = 0; vi < snowCount; vi++) {
        source.snowHeights[vi] = snowRaw[vi * 3 + 1];
      }
    }
    trackSources.push(source);

    const exaggeration = parseFloat(exaggerationInput.value);
    const geometry = new LineGeometry();
    geometry.setPositions(trackPointsFromSource(source, exaggeration));
    const material = new LineMaterial({
      color: 0x2288ff,
      linewidth: 4,
      depthTest: true,
      depthWrite: false,
      transparent: true,
      opacity: 0.95,
      worldUnits: false,
    });
    material.resolution.copy(resolution);

    const line = new Line2(geometry, material);
    line.computeLineDistances();
    line.renderOrder = 2;
    line.frustumCulled = false;
    scene.add(line);
    trackLines.push(line);
  }
}

function configureElevationSelect() {
  const hasSnowSurface =
    sceneMeta?.has_snow_surface && sceneMeta?.elevation_models?.snow_surface;
  elevationLabel.hidden = !hasSnowSurface;
  if (!hasSnowSurface) {
    activeElevationModel = "base";
    return;
  }
  elevationSelect.value = activeElevationModel;
}

async function loadElevationModels(base, files) {
  const models = {
    base: null,
    snow_surface: null,
  };

  const baseBuf = await loadBinary(`${base}/${files.positions}`);
  models.base = extractHeights(new Float32Array(baseBuf));

  const snowFile = sceneMeta?.elevation_models?.snow_surface;
  if (snowFile) {
    try {
      const snowBuf = await loadBinary(`${base}/${snowFile}`);
      models.snow_surface = extractHeights(new Float32Array(snowBuf));
    } catch {
      models.snow_surface = null;
    }
  }

  return models;
}

function switchElevationModel(modelKey) {
  if (!terrainMesh || !heightModels || !heightModels[modelKey]) {
    return;
  }
  activeElevationModel = modelKey;
  applyExaggeration(parseFloat(exaggerationInput.value));
  updateTrackHeights(parseFloat(exaggerationInput.value));
  frameCamera(terrainMesh);
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
  activeElevationModel = sceneMeta.has_snow_surface ? "snow_surface" : "base";

  const [positionsBuf, uvsBuf, indicesBuf, winterTexture] = await Promise.all([
    loadBinary(`${base}/${sceneMeta.files.positions}`),
    loadBinary(`${base}/${sceneMeta.files.uvs}`),
    loadBinary(`${base}/${sceneMeta.files.indices}`),
    loadTexture(`${base}/${sceneMeta.textures.winter}`),
  ]);

  heightModels = await loadElevationModels(base, sceneMeta.files);

  const positions = new Float32Array(positionsBuf);
  const uvs = new Float32Array(uvsBuf);
  const IndexArray = sceneMeta.index_dtype === "uint16" ? Uint16Array : Uint32Array;
  const rawIndices = new IndexArray(indicesBuf);
  const indices = new Uint32Array(rawIndices.length);
  for (let i = 0; i < rawIndices.length; i++) {
    indices[i] = rawIndices[i];
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(new THREE.Uint32BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  const material = createTerrainMaterial(winterTexture);
  terrainMesh = new THREE.Mesh(geometry, material);
  terrainMesh.frustumCulled = false;
  scene.add(terrainMesh);

  configureElevationSelect();
  if (sceneMeta.has_snow_surface && heightModels.snow_surface) {
    elevationSelect.value = "snow_surface";
    activeElevationModel = "snow_surface";
  }

  const exaggeration = parseFloat(exaggerationInput.value);
  applyExaggeration(exaggeration);
  await loadTracks(tileId);
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
  const elevInfo = sceneMeta.has_snow_surface ? " · Schneeoberfläche verfügbar" : "";
  const trackInfo = trackLines.length > 0 ? ` · ${trackLines.length} GPX-Route(n)` : "";
  statusEl.textContent =
    `${tileId} · ${vertices} Punkte · ${triangles} Dreiecke${texInfo}${elevInfo}${trackInfo}` +
    ` · Wintersonne 11 Uhr`;
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

elevationSelect.addEventListener("change", () => {
  switchElevationModel(elevationSelect.value);
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
  const resolution = new THREE.Vector2(window.innerWidth, window.innerHeight);
  for (const line of trackLines) {
    line.material.resolution.copy(resolution);
  }
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
