import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js';
import RegionsPlugin from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.esm.js';

const METHODS = ['noisereduce', 'deepfilternet', 'rnnoise',
  'noisereduce__deepfilternet', 'rnnoise__deepfilternet', 'noisereduce__rnnoise'];
const RNN_MODELS = ['mp.rnnn', 'bd.rnnn', 'sh.rnnn', 'lq.rnnn', 'cb.rnnn'];

const state = {
  tracks: [], focus: 0, mergeable: false, mode: 'normal',
  excluded: [], cleaned: {}, mergeResult: null, showCleaned: false,
  globalDefault: {
    method: 'deepfilternet',
    params: { stationary: true, prop_decrease: 0.8, atten_lim_db: 15,
              dfn_mix: 0.8, rnn_model: 'mp.rnnn', rnn_mix: 1.0 },
  },
  overrides: {},
};

function effective(i) {
  const o = state.overrides[i];
  return o ? o : { method: state.globalDefault.method,
                   params: { ...state.globalDefault.params } };
}

const $ = (s) => document.querySelector(s);

const apiJSON = (path, body) =>
  fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body) }).then(checkOk);
async function checkOk(r) {
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch { /* noop */ }
    throw new Error(`${r.status}: ${detail}`);
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
// keep only one trim region at a time
regions.on('region-created', (r) => {
  regions.getRegions().forEach((x) => { if (x !== r) x.remove(); });
});

const audioUrl = (path) => `/api/audio?path=${encodeURIComponent(path)}`;

/* ---------- loading sessions ---------- */

function applySession(res) {
  state.tracks = res.tracks || [];
  state.mergeable = !!res.mergeable;
  state.excluded = [];
  state.cleaned = {};
  state.overrides = {};
  state.mergeResult = null;
  state.focus = 0;

  const badge = $('#session-badge');
  if (!state.tracks.length) {
    badge.className = 'badge warn'; badge.textContent = '⚠ ' + (res.reason || 'no tracks found');
    badge.classList.remove('hidden');
    $('#editor').classList.add('hidden');
    $('#empty-state').classList.remove('hidden');
    return;
  }
  if (state.mergeable) {
    badge.className = 'badge ok';
    badge.textContent = `✓ ${state.tracks.length} tracks · can merge`;
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

async function loadDefault() {
  try { applySession(await fetch('/api/default_session').then(checkOk)); }
  catch { /* no default; wait for user */ }
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
  ws.load(audioUrl(state.tracks[i].path));
  renderParams();
}

/* ---------- params panel ---------- */

function renderParams() {
  const e = effective(state.focus);
  const p = e.params;
  const box = $('#params'); box.innerHTML = '';
  box.append(
    sel('Method', METHODS, e.method, (v) => { setEff('method', v); renderParams(); }),
    chk('Stationary noise (noisereduce)', p.stationary, (v) => setParam('stationary', v)),
    num('noisereduce strength', 0, 1, 0.05, p.prop_decrease, (v) => setParam('prop_decrease', v)),
    num('DeepFilterNet max atten (dB)', 0, 60, 1, p.atten_lim_db, (v) => setParam('atten_lim_db', v)),
    num('DeepFilterNet dry/wet mix', 0, 1, 0.05, p.dfn_mix, (v) => setParam('dfn_mix', v)),
    sel('RNNoise model', RNN_MODELS, p.rnn_model, (v) => setParam('rnn_model', v)),
    num('RNNoise wet/dry mix', 0, 1, 0.05, p.rnn_mix, (v) => setParam('rnn_mix', v)),
  );
}
function setEff(k, v) { const e = effective(state.focus); e[k] = v; state.overrides[state.focus] = e; }
function setParam(k, v) { const e = effective(state.focus); e.params[k] = v; state.overrides[state.focus] = e; }

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
  const r = regions.getRegions()[0];
  return r ? [r.start, r.end] : [0, 0];
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
  const e = effective(state.focus);
  setStatus('rendering…', 'busy');
  try {
    const res = await apiJSON('/api/render', { path: t.path, method: e.method, params: e.params, trim: currentTrim(), mode: 'normal' });
    state.cleaned[state.focus] = res.out_path;
    showResult(res, `Rendered ${t.name}`);
    renderTrackList();
    $('#ab-clean').disabled = false;
    setAB('clean');
  } catch (err) { setStatus('error: ' + err.message, 'err'); }
}

async function renderAll() {
  setStatus('rendering all tracks…', 'busy');
  const tracks = state.tracks.map((t, i) => { const e = effective(i); return { path: t.path, method: e.method, params: e.params, trim: [0, 0], mode: 'normal' }; });
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
  const e = effective(state.focus);
  setStatus('fusing mics + cleaning…', 'busy');
  try {
    const res = await apiJSON('/api/merge', { paths: state.tracks.map((t) => t.path), exclude: [], method: e.method, params: e.params, trim: currentTrim() });
    state.mergeResult = res.out_path;
    state.excluded = res.excluded || [];
    renderTrackList();
    $('#now-track').textContent = 'Merged master';
    showResult(res, 'Merged master' + (state.excluded.length ? ` (excluded ${state.excluded.map((i) => state.tracks[i]?.name).join(', ')})` : ''));
    ws.load(audioUrl(res.out_path));
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
  const cleaned = state.cleaned[state.focus];
  if (which === 'clean' && !cleaned) return;
  state.showCleaned = which === 'clean';
  document.querySelectorAll('#ab-toggle button').forEach((b) => b.classList.toggle('active', b.dataset.ab === which));
  const t = state.tracks[state.focus];
  ws.load(audioUrl(state.showCleaned ? cleaned : t.path));
}

function setMode(mode) {
  if (mode === 'merge' && !state.mergeable) return;
  state.mode = mode;
  document.querySelectorAll('#mode-toggle button').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  $('#panel-title').textContent = mode === 'merge' ? 'Fuse mics → one voice' : 'Cleanup';
  $('#render').innerHTML = (mode === 'merge' ? 'Merge + clean' : 'Render') + ' <kbd>r</kbd>';
  document.querySelectorAll('.hidden-merge').forEach((el) => el.classList.toggle('hidden', mode === 'merge'));
}

/* ---------- wiring ---------- */

$('#play').onclick = () => ws.playPause();
$('#zoom-in').onclick = () => ws.zoom((ws.options.minPxPerSec || 0) + 30);
$('#zoom-out').onclick = () => ws.zoom(Math.max(0, (ws.options.minPxPerSec || 0) - 30));
$('#clear-trim').onclick = () => regions.clearRegions();
$('#render').onclick = () => (state.mode === 'merge' ? renderMerge() : renderFocus());
$('#render-all').onclick = renderAll;
$('#export').onclick = doExport;
document.querySelectorAll('#mode-toggle button').forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });
document.querySelectorAll('#ab-toggle button').forEach((b) => { b.onclick = () => setAB(b.dataset.ab); });

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
  ['i / o', 'set trim in / out'], ['r', 'render'], ['⇧R', 'render all'],
  ['n / p', 'next / prev track'], ['m', 'toggle merge'], ['a', 'A/B'],
  ['e', 'export'], ['?', 'this help'],
];
$('#shortcuts-body').innerHTML = SHORT.map(([k, d]) => `<div><kbd>${k}</kbd> ${d}</div>`).join('');
$('#help-btn').onclick = () => $('#shortcuts').classList.toggle('hidden');
$('#close-help').onclick = () => $('#shortcuts').classList.add('hidden');

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

loadDefault();

export { state, ws, focusTrack };
