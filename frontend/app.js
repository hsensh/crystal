import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js';
import RegionsPlugin from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.esm.js';

const RNN_MODELS = ['mp.rnnn', 'bd.rnnn', 'sh.rnnn', 'lq.rnnn', 'cb.rnnn'];

// each stage type + its compact controls and default params
const STAGE_SPECS = {
  leveler: {
    label: 'Leveler', defaults: { lvl_max_gain_db: 12, lvl_smooth_ms: 400 },
    controls: [
      { k: 'lvl_max_gain_db', type: 'range', label: 'max boost (dB)', min: 0, max: 30, step: 1 },
      { k: 'lvl_smooth_ms', type: 'range', label: 'smoothing (ms)', min: 50, max: 2000, step: 50 },
    ],
  },
  noisereduce: {
    label: 'noisereduce', defaults: { stationary: true, prop_decrease: 0.8 },
    controls: [
      { k: 'stationary', type: 'check', label: 'stationary noise' },
      { k: 'prop_decrease', type: 'range', label: 'strength', min: 0, max: 1, step: 0.05 },
    ],
  },
  deepfilternet: {
    label: 'DeepFilterNet', defaults: { atten_lim_db: 15, dfn_mix: 0.8 },
    controls: [
      { k: 'atten_lim_db', type: 'range', label: 'max atten (dB)', min: 0, max: 60, step: 1 },
      { k: 'dfn_mix', type: 'range', label: 'dry/wet mix', min: 0, max: 1, step: 0.05 },
    ],
  },
  rnnoise: {
    label: 'RNNoise', defaults: { rnn_model: 'mp.rnnn', rnn_mix: 1.0 },
    controls: [
      { k: 'rnn_model', type: 'select', label: 'model', opts: RNN_MODELS },
      { k: 'rnn_mix', type: 'range', label: 'wet/dry mix', min: 0, max: 1, step: 0.05 },
    ],
  },
};

const newStage = (type) => ({ type, params: { ...STAGE_SPECS[type].defaults } });
const defaultChain = () => [newStage('leveler'), newStage('deepfilternet')];

const state = {
  tracks: [], focus: 0, mergeable: false, mode: 'normal',
  excluded: [], cleaned: {}, mergeResult: null, mergeRaw: null, showCleaned: false,
  chains: {},                 // trackIndex -> stage[]
  mergeChain: defaultChain(), // merge mode's own chain
  micInclude: [],             // bool per track: include in merge
  fuseMode: 'blend',          // 'blend' | 'autopick'
  noiseStrength: 1.0,
  winMs: 46,
  smoothMs: 120,
  showNoise: false,
};
const noiseIds = new Set();   // wavesurfer region ids that are noise highlights

// the chain currently being edited (per-track in normal, shared in merge)
function activeChain() {
  if (state.mode === 'merge') return state.mergeChain;
  if (!state.chains[state.focus]) state.chains[state.focus] = defaultChain();
  return state.chains[state.focus];
}
function chainFor(i) {
  if (!state.chains[i]) state.chains[i] = defaultChain();
  return state.chains[i];
}

const $ = (s) => document.querySelector(s);

const apiJSON = (path, body) =>
  fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body) }).then(checkOk);
async function checkOk(r) {
  if (!r.ok) {
    let msg = r.statusText;
    try { const b = await r.json(); msg = b.message || b.detail || msg; } catch { /* noop */ }
    throw new Error(msg);                       // friendly message; full detail is in Logs
  }
  return r.json();
}

function setStatus(msg, cls = '') {
  const s = $('#status'); s.textContent = msg; s.className = 'status ' + cls;
}

const regions = RegionsPlugin.create();
const ws = WaveSurfer.create({
  container: '#waveform', waveColor: '#3b6ef5', progressColor: '#9db8ff',
  cursorColor: '#e8eefc', height: 150, normalize: true, plugins: [regions],
});
regions.enableDragSelection({ color: 'rgba(246,181,69,.22)' });
// keep only one TRIM region at a time (noise highlights are exempt)
regions.on('region-created', (r) => {
  if (noiseIds.has(r.id)) return;  // a highlight, not a user trim
  regions.getRegions().forEach((x) => { if (x !== r && !noiseIds.has(x.id)) x.remove(); });
});

const audioUrl = (path) => `/api/audio?path=${encodeURIComponent(path)}`;

// loading a new clip while one is still loading makes WaveSurfer abort the old
// fetch (AbortError) — that's expected, swallow it so it isn't an unhandled reject
function loadWave(url) {
  ws.load(url).catch((e) => { if (e && e.name !== 'AbortError') console.error(e); });
}

/* ---------- loading sessions ---------- */

function applySession(res) {
  state.tracks = res.tracks || [];
  state.mergeable = !!res.mergeable;
  state.excluded = [];
  state.cleaned = {};
  state.overrides = {};
  state.mergeResult = null;
  state.mergeRaw = null;
  state.focus = 0;
  state.micInclude = state.tracks.map(() => true);  // all mics in by default

  const badge = $('#session-badge');
  if (!state.tracks.length) {
    badge.className = 'badge warn'; badge.textContent = (res.reason || 'no tracks found');
    badge.classList.remove('hidden');
    $('#editor').classList.add('hidden');
    $('#empty-state').classList.remove('hidden');
    return;
  }
  if (state.mergeable) {
    badge.className = 'badge ok';
    badge.textContent = `${state.tracks.length} tracks · can merge`;
  } else {
    badge.className = 'badge warn';
    badge.textContent = `${state.tracks.length} tracks · merge off (${res.reason})`;
  }
  badge.classList.remove('hidden');
  $('#merge-btn').disabled = !state.mergeable;
  if (!state.mergeable && state.mode === 'merge') setMode('normal');

  $('#empty-state').classList.add('hidden');
  $('#editor').classList.remove('hidden');
  renderTrackList();
  focusTrack(0);
}

async function uploadFiles(fileList) {
  const wavs = Array.from(fileList).filter((f) => /\.wav$/i.test(f.name));
  if (!wavs.length) { flashBadge('warn', 'no .wav files in that selection'); return; }
  const fd = new FormData();
  wavs.forEach((f) => fd.append('files', f, f.name));
  flashBadge('ok', `uploading ${wavs.length} file(s)…`);
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd }).then(checkOk);
    applySession(res);
  } catch (e) { flashBadge('warn', 'upload failed: ' + e.message); }
}
function flashBadge(kind, msg) {
  const b = $('#session-badge'); b.className = 'badge ' + kind; b.textContent = msg;
  b.classList.remove('hidden');
}

/* ---------- track list + focus ---------- */

function renderTrackList() {
  const ul = $('#track-list'); ul.innerHTML = '';
  state.tracks.forEach((t, i) => {
    const li = document.createElement('li');
    if (i === state.focus) li.classList.add('active');
    if (state.cleaned[i]) li.classList.add('done');
    if (state.excluded.includes(i)) li.classList.add('excluded');
    const name = document.createElement('div'); name.className = 't-name'; name.textContent = t.name;
    const meta = document.createElement('div'); meta.className = 't-meta';
    meta.textContent = `${(t.duration ?? 0).toFixed(1)}s · ${t.channels}ch`;
    li.append(name, meta);
    li.onclick = () => focusTrack(i);
    ul.appendChild(li);
  });
}

function focusTrack(i) {
  state.focus = i;
  state.showCleaned = false;
  renderTrackList();
  $('#now-track').textContent = state.tracks[i].name;
  $('#ab-clean').disabled = !state.cleaned[i];
  setAB('orig');
  loadWave(audioUrl(state.tracks[i].path));
  noiseIds.clear();  // regions cleared on load; drop stale ids
  ws.once('ready', () => { if (state.showNoise) refreshNoise(); });
  renderParams();
}

/* ---------- params panel ---------- */

function renderParams() {
  const chain = activeChain();
  const box = $('#chain'); box.innerHTML = '';
  if (!chain.length) {
    const e = document.createElement('div'); e.className = 'chain-empty';
    e.textContent = 'No stages — audio passes through untouched. Add a stage below.';
    box.appendChild(e); return;
  }
  chain.forEach((st, idx) => box.appendChild(stageCard(st, idx, chain)));
}

function stageCard(st, idx, chain) {
  const spec = STAGE_SPECS[st.type];
  const card = document.createElement('div'); card.className = 'stage';

  const head = document.createElement('div'); head.className = 'stage-head';
  const numEl = document.createElement('span'); numEl.className = 'stage-num'; numEl.textContent = idx + 1;
  const name = document.createElement('span'); name.className = 'stage-name'; name.textContent = spec.label;
  const tools = document.createElement('div'); tools.className = 'stage-tools';
  const up = toolBtn('↑', idx === 0, () => move(chain, idx, -1));
  const down = toolBtn('↓', idx === chain.length - 1, () => move(chain, idx, +1));
  const rm = toolBtn('×', false, () => { chain.splice(idx, 1); renderParams(); });
  rm.classList.add('rm');
  tools.append(up, down, rm);
  head.append(numEl, name, tools);

  const body = document.createElement('div'); body.className = 'stage-body';
  spec.controls.forEach((c) => {
    if (c.type === 'check') body.appendChild(chk(c.label, st.params[c.k], (v) => { st.params[c.k] = v; }));
    else if (c.type === 'select') body.appendChild(sel(c.label, c.opts, st.params[c.k], (v) => { st.params[c.k] = v; }));
    else body.appendChild(num(c.label, c.min, c.max, c.step, st.params[c.k], (v) => { st.params[c.k] = v; }));
  });

  card.append(head, body); return card;
}

function toolBtn(label, disabled, onclick) {
  const b = document.createElement('button'); b.textContent = label; b.disabled = disabled;
  b.onclick = onclick; return b;
}
function move(chain, idx, dir) {
  const j = idx + dir;
  if (j < 0 || j >= chain.length) return;
  [chain[idx], chain[j]] = [chain[j], chain[idx]];
  renderParams();
}
function addStage(type) { activeChain().push(newStage(type)); renderParams(); }

/* ---------- merge mic panel ---------- */

function renderMicPanel() {
  const box = $('#mic-list'); box.innerHTML = '';
  const soloed = state.micInclude.filter(Boolean).length === 1;
  state.tracks.forEach((t, i) => {
    const row = document.createElement('div'); row.className = 'mic-row' + (state.micInclude[i] ? '' : ' off');
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = state.micInclude[i];
    cb.onchange = () => { state.micInclude[i] = cb.checked; renderMicPanel(); };
    const name = document.createElement('span'); name.className = 'm-name'; name.textContent = t.name;
    const meta = document.createElement('span'); meta.className = 'm-meta'; meta.textContent = `${(t.duration ?? 0).toFixed(1)}s · ${t.channels}ch`;
    const spacer = document.createElement('span'); spacer.className = 'spacer';
    const solo = document.createElement('button');
    solo.className = 'solo' + (soloed && state.micInclude[i] ? ' active' : '');
    solo.textContent = 'Solo';
    solo.onclick = () => { state.micInclude = state.tracks.map((_, j) => j === i); renderMicPanel(); };
    row.append(cb, name, meta, spacer, solo);
    box.appendChild(row);
  });
}
function mergeExclude() {
  return state.tracks.map((_, i) => i).filter((i) => !state.micInclude[i]);
}

function sel(label, opts, value, onchange) {
  const l = document.createElement('label'); l.textContent = label;
  const s = document.createElement('select');
  opts.forEach((o) => { const op = document.createElement('option'); op.value = o; op.textContent = o; if (o === value) op.selected = true; s.appendChild(op); });
  s.onchange = () => onchange(s.value); l.appendChild(s); return l;
}
function num(label, min, max, step, value, onchange) {
  const l = document.createElement('label');
  const row = document.createElement('div'); row.className = 'row';
  const cap = document.createElement('span'); cap.textContent = label;
  const val = document.createElement('span'); val.className = 'val'; val.textContent = (+value).toFixed(2);
  row.append(cap, val);
  const i = document.createElement('input');
  i.type = 'range'; i.min = min; i.max = max; i.step = step; i.value = value;
  i.oninput = () => { val.textContent = (+i.value).toFixed(2); onchange(parseFloat(i.value)); };
  l.append(row, i); return l;
}
function chk(label, value, onchange) {
  const l = document.createElement('label'); l.className = 'check';
  const i = document.createElement('input'); i.type = 'checkbox'; i.checked = value;
  i.onchange = () => onchange(i.checked);
  const span = document.createElement('span'); span.textContent = label;
  l.append(i, span); return l;
}

/* ---------- trim + transport ---------- */

function currentTrim() {
  const r = regions.getRegions().find((x) => !noiseIds.has(x.id));
  return r ? [r.start, r.end] : [0, 0];
}

/* ---------- noise highlight ---------- */

function clearNoiseRegions() {
  regions.getRegions().forEach((r) => { if (noiseIds.has(r.id)) r.remove(); });
  noiseIds.clear();
}
async function refreshNoise() {
  clearNoiseRegions();
  if (!state.showNoise || !state.tracks.length) return;
  const path = state.tracks[state.focus].path;
  try {
    const res = await fetch(`/api/noise?path=${encodeURIComponent(path)}`).then(checkOk);
    res.segments.forEach(([a, b]) => {
      const reg = regions.addRegion({ start: a, end: b, drag: false, resize: false,
        color: 'rgba(255,107,107,.25)' });
      noiseIds.add(reg.id);
    });
    setStatus(`${res.segments.length} noisy region(s) flagged on ${state.tracks[state.focus].name}`, '');
  } catch (err) { setStatus('noise detect failed: ' + err.message, 'err'); }
}
function setTrimEdge(edge) {
  const r = regions.getRegions()[0];
  const t = ws.getCurrentTime();
  if (!r) { regions.addRegion({ start: edge === 'start' ? t : 0, end: edge === 'end' ? t : ws.getDuration() }); return; }
  if (edge === 'start') r.setOptions({ start: t }); else r.setOptions({ end: t });
}

/* ---------- render / merge / export ---------- */

async function renderFocus() {
  const t = state.tracks[state.focus];
  setStatus('rendering…', 'busy');
  try {
    const res = await apiJSON('/api/render', { path: t.path, chain: chainFor(state.focus), trim: currentTrim(), mode: 'normal' });
    state.cleaned[state.focus] = res.out_path;
    showResult(res, `Rendered ${t.name}`);
    renderTrackList();
    $('#ab-clean').disabled = false;
    setAB('clean');
  } catch (err) { setStatus('error: ' + err.message, 'err'); }
}

async function renderAll() {
  setStatus('rendering all tracks…', 'busy');
  const tracks = state.tracks.map((t, i) => ({ path: t.path, chain: chainFor(i), trim: [0, 0], mode: 'normal' }));
  try {
    const res = await apiJSON('/api/render_all', { tracks });
    res.results.forEach((r, i) => { if (r.ok) state.cleaned[i] = r.out_path; });
    renderTrackList();
    document.querySelectorAll('#track-list li').forEach((li, i) => { if (!res.results[i].ok) li.classList.add('error'); });
    const fails = res.results.filter((r) => !r.ok).length;
    setStatus(`rendered ${res.results.length - fails}/${res.results.length}` + (fails ? ` · ${fails} failed` : ''), fails ? 'err' : '');
  } catch (err) { setStatus('error: ' + err.message, 'err'); }
}

async function renderMerge() {
  const exclude = mergeExclude();
  const active = state.tracks.length - exclude.length;
  if (!active) { setStatus('all mics excluded — include at least one', 'err'); return; }
  setStatus(`fusing ${active} mic(s)${$('#preclean').checked ? ' (cleaning each first)' : ''}…`, 'busy');
  try {
    const res = await apiJSON('/api/merge', {
      paths: state.tracks.map((t) => t.path),
      exclude, auto_exclude: exclude.length ? false : $('#auto-exclude').checked,
      preclean: $('#preclean').checked,
      fuse_mode: state.fuseMode, noise_strength: state.noiseStrength,
      win_ms: state.winMs, smooth_ms: state.smoothMs,
      chain: state.mergeChain, trim: currentTrim(),
    });
    state.mergeResult = res.out_path;
    state.mergeRaw = res.raw_path;
    const usedNames = (res.active || []).map((i) => state.tracks[i]?.name).join(', ');
    $('#now-track').textContent = 'Merged master';
    showResult(res, `Merged from: ${usedNames}`);
    $('#ab-clean').disabled = false;     // enable A/B for the merge result
    setAB('clean');                       // jump to the cleaned preview
  } catch (err) { setStatus('error: ' + err.message, 'err'); }
}

function showResult(res, title) {
  setStatus(title, '');
  const r = $('#result');
  const d = (x) => x.toFixed(1);
  r.innerHTML =
    `<span class="metric">noise floor <b>${d(res.before.noise_floor_dbfs)} → ${d(res.after.noise_floor_dbfs)} dB</b></span>` +
    `<span class="metric">RMS ${d(res.before.rms_dbfs)} → ${d(res.after.rms_dbfs)} dB</span>` +
    `<span class="metric">→ output/${state.mode}/${res.name}</span>`;
  r.classList.remove('hidden');
}

async function doExport() {
  const src = state.mode === 'merge' ? state.mergeResult : state.cleaned[state.focus];
  if (!src) { setStatus('nothing rendered yet — Render first', 'err'); return; }
  try { const res = await apiJSON('/api/export', { src, dest_dir: 'output/export' }); setStatus('exported → ' + res.dest, ''); }
  catch (err) { setStatus('error: ' + err.message, 'err'); }
}

/* ---------- A/B + mode ---------- */

function setAB(which) {
  // merge mode: Original = raw combined, Cleaned = cleaned merged master
  if (state.mode === 'merge') {
    if (which === 'clean' && !state.mergeResult) return;
    if (which === 'orig' && !state.mergeRaw) return;
    state.showCleaned = which === 'clean';
    document.querySelectorAll('#ab-toggle button').forEach((b) => b.classList.toggle('active', b.dataset.ab === which));
    loadWave(audioUrl(state.showCleaned ? state.mergeResult : state.mergeRaw));
    return;
  }
  // normal mode: per-track original vs cleaned
  const cleaned = state.cleaned[state.focus];
  if (which === 'clean' && !cleaned) return;
  state.showCleaned = which === 'clean';
  document.querySelectorAll('#ab-toggle button').forEach((b) => b.classList.toggle('active', b.dataset.ab === which));
  const t = state.tracks[state.focus];
  loadWave(audioUrl(state.showCleaned ? cleaned : t.path));
}

function setMode(mode) {
  if (mode === 'merge' && !state.mergeable) return;
  state.mode = mode;
  document.querySelectorAll('#mode-toggle button').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  $('#panel-title').textContent = mode === 'merge' ? 'Fuse mics → cleanup chain' : 'Cleanup chain';
  $('#render').innerHTML = (mode === 'merge' ? 'Merge + clean' : 'Render') + ' <kbd>r</kbd>';
  document.querySelectorAll('.hidden-merge').forEach((el) => el.classList.toggle('hidden', mode === 'merge'));
  $('#merge-panel').classList.toggle('hidden', mode !== 'merge');
  if (mode === 'merge') {
    renderMicPanel();
    $('#ab-clean').disabled = !state.mergeResult;  // enabled once a merge is rendered
  }
  renderParams();  // chain differs between per-track and merge
}

/* ---------- wiring ---------- */

$('#play').onclick = () => ws.playPause();
ws.on('play', () => { $('#play').textContent = 'Pause'; $('#play').classList.remove('play'); });
ws.on('pause', () => { $('#play').textContent = 'Play'; $('#play').classList.add('play'); });
$('#zoom-in').onclick = () => ws.zoom((ws.options.minPxPerSec || 0) + 30);
$('#zoom-out').onclick = () => ws.zoom(Math.max(0, (ws.options.minPxPerSec || 0) - 30));
$('#clear-trim').onclick = () => regions.clearRegions();
$('#render').onclick = () => (state.mode === 'merge' ? renderMerge() : renderFocus());
$('#render-all').onclick = renderAll;
$('#export').onclick = doExport;
document.querySelectorAll('#mode-toggle button').forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });
document.querySelectorAll('#ab-toggle button').forEach((b) => { b.onclick = () => setAB(b.dataset.ab); });
document.querySelectorAll('.add-btn').forEach((b) => { b.onclick = () => addStage(b.dataset.add); });
document.querySelectorAll('#fuse-mode button').forEach((b) => {
  b.onclick = () => {
    state.fuseMode = b.dataset.fuse;
    document.querySelectorAll('#fuse-mode button').forEach((x) => x.classList.toggle('active', x === b));
    $('#fuse-desc').textContent = state.fuseMode === 'autopick'
      ? 'picks the cleanest mic per moment, crossfades, level-matched — excludes bad parts'
      : 'sums every mic (weighted)';
    $('#autopick-adv').classList.toggle('hidden', state.fuseMode !== 'autopick');
  };
});
$('#noise-strength').oninput = (e) => {
  state.noiseStrength = parseFloat(e.target.value);
  $('#noise-val').textContent = state.noiseStrength.toFixed(2);
};
$('#win-ms').oninput = (e) => {
  state.winMs = parseFloat(e.target.value);
  $('#win-val').textContent = String(state.winMs);
};
$('#smooth-ms').oninput = (e) => {
  state.smoothMs = parseFloat(e.target.value);
  $('#smooth-val').textContent = String(state.smoothMs);
};
$('#noise-btn').onclick = () => {
  state.showNoise = !state.showNoise;
  $('#noise-btn').classList.toggle('active', state.showNoise);
  refreshNoise();
};

// dropzone + pickers
const dz = $('#dropzone');
$('#pick-files').onclick = (e) => { e.stopPropagation(); $('#file-input').click(); };
$('#pick-folder').onclick = (e) => { e.stopPropagation(); $('#folder-input').click(); };
dz.onclick = () => $('#file-input').click();
$('#file-input').onchange = (e) => uploadFiles(e.target.files);
$('#folder-input').onchange = (e) => uploadFiles(e.target.files);
['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
dz.addEventListener('drop', (e) => { if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files); });

// help overlay
const SHORT = [
  ['space', 'play / pause'], ['[ ]', 'zoom out / in'], ['← →', 'seek'],
  ['i / o', 'set trim in / out'], ['r', 'render'], ['Shift+R', 'render all'],
  ['n / p', 'next / prev track'], ['m', 'toggle merge'], ['a', 'A/B'],
  ['e', 'export'], ['?', 'this help'],
];
$('#shortcuts-body').innerHTML = SHORT.map(([k, d]) => `<div><kbd>${k}</kbd> ${d}</div>`).join('');
$('#help-btn').onclick = () => $('#shortcuts').classList.toggle('hidden');
$('#close-help').onclick = () => $('#shortcuts').classList.add('hidden');

async function loadLogs() {
  try { $('#logs-body').textContent = await fetch('/api/logs').then((r) => r.text()); }
  catch (e) { $('#logs-body').textContent = 'could not load logs: ' + e.message; }
}
$('#logs-btn').onclick = () => { $('#logs').classList.remove('hidden'); loadLogs(); };
$('#logs-refresh').onclick = loadLogs;
$('#logs-close').onclick = () => $('#logs').classList.add('hidden');

const SHORTCUTS = {
  ' ': () => ws.playPause(),
  '[': () => $('#zoom-out').onclick(),
  ']': () => $('#zoom-in').onclick(),
  'ArrowLeft': () => ws.setTime(Math.max(0, ws.getCurrentTime() - 2)),
  'ArrowRight': () => ws.setTime(ws.getCurrentTime() + 2),
  'i': () => setTrimEdge('start'),
  'o': () => setTrimEdge('end'),
  'r': () => $('#render').onclick(),
  'R': () => renderAll(),
  'n': () => state.tracks.length && focusTrack(Math.min(state.tracks.length - 1, state.focus + 1)),
  'p': () => state.tracks.length && focusTrack(Math.max(0, state.focus - 1)),
  'm': () => setMode(state.mode === 'merge' ? 'normal' : 'merge'),
  'a': () => setAB(state.showCleaned ? 'orig' : 'clean'),
  'e': () => doExport(),
  '?': () => $('#shortcuts').classList.toggle('hidden'),
};
window.addEventListener('keydown', (ev) => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(ev.target.tagName)) return;
  const key = ev.shiftKey && ev.key === 'R' ? 'R' : ev.key;
  const fn = SHORTCUTS[key];
  if (fn) { ev.preventDefault(); fn(); }
});

/* ---------- first-run resource setup ---------- */

const RES_ITEMS = [
  ['ffmpeg', 'Audio engine (ffmpeg)'],
  ['torch', 'Speech-AI runtime (PyTorch)'],
  ['deepfilternet', 'DeepFilterNet model'],
];

function renderResList(s, working) {
  $('#res-list').innerHTML = RES_ITEMS.map(([k, label]) => {
    const cls = s[k] ? 'ok' : (working ? 'work' : 'pending');
    const mark = s[k] ? 'ready' : (working ? 'installing…' : 'pending');
    return `<li class="${cls}">${label} — ${mark}</li>`;
  }).join('');
}

async function pollResources() {
  let s;
  try { s = await fetch('/api/resources').then((r) => r.json()); }
  catch { setTimeout(pollResources, 2000); return; }
  renderResList(s, s.install && s.install.running);
  $('#setup-log').textContent = (s.install && s.install.log || []).join('\n');
  if (s.ready) { setTimeout(() => $('#setup').classList.add('hidden'), 700); return; }
  if (s.install && s.install.error) {
    $('#setup-go').classList.remove('hidden');
    $('#setup-go').disabled = false; $('#setup-go').textContent = 'Retry';
    return;
  }
  setTimeout(pollResources, 1500);
}

async function beginInstall() {
  $('#setup-go').classList.add('hidden');
  await fetch('/api/resources/install', { method: 'POST' });
  pollResources();
}
$('#setup-go').onclick = beginInstall;

async function checkResources() {
  let s;
  try { s = await fetch('/api/resources').then((r) => r.json()); }
  catch { return; }                       // not the desktop build; skip
  if (s.ready) return;                     // everything present, no screen
  $('#setup').classList.remove('hidden');  // dedicated first-run download screen
  renderResList(s, false);
  beginInstall();                          // auto-start the download
}

checkResources();

export { state, ws, focusTrack };
