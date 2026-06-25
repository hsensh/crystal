import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js';
import RegionsPlugin from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.esm.js';

const state = { tracks: [], focus: 0, mergeable: false, mode: 'normal' };

const METHODS = ['noisereduce', 'deepfilternet', 'rnnoise',
  'noisereduce__deepfilternet', 'rnnoise__deepfilternet', 'noisereduce__rnnoise'];
const RNN_MODELS = ['mp.rnnn', 'bd.rnnn', 'sh.rnnn', 'lq.rnnn', 'cb.rnnn'];

state.globalDefault = {
  method: 'deepfilternet',
  params: { stationary: true, prop_decrease: 0.8, atten_lim_db: 15,
            dfn_mix: 0.8, rnn_model: 'mp.rnnn', rnn_mix: 1.0 },
};
state.overrides = {};   // index -> {method, params}
state.cleaned = {};     // index -> out_path
state.showCleaned = false;

function effective(i) {
  const o = state.overrides[i];
  return o ? o : { method: state.globalDefault.method,
                   params: { ...state.globalDefault.params } };
}

const $ = (s) => document.querySelector(s);
const api = (path, body) =>
  fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body) }).then((r) => {
    if (!r.ok) throw new Error(`API error ${r.status}: ${r.statusText}`);
    return r.json();
  });

const regions = RegionsPlugin.create();
const ws = WaveSurfer.create({
  container: '#waveform', waveColor: '#3b82f6', progressColor: '#1d4ed8',
  height: 160, normalize: true, plugins: [regions],
});

function audioUrl(path) {
  return `/api/audio?path=${encodeURIComponent(path)}`;
}

function renderTrackList() {
  const ul = $('#track-list');
  ul.innerHTML = '';
  state.tracks.forEach((t, i) => {
    const li = document.createElement('li');
    li.textContent = `${t.name}  ${(t.duration ?? 0).toFixed(1)}s`;
    if (i === state.focus) li.classList.add('active');
    li.onclick = () => focusTrack(i);
    ul.appendChild(li);
  });
}

function renderParams() {
  const e = effective(state.focus);
  const p = e.params;
  const box = $('#params');
  box.innerHTML = '';
  const method = sel('Method', METHODS, e.method, (v) => { setEff('method', v); });
  const prop = num('nr strength', 0, 1, 0.05, p.prop_decrease, (v) => setParam('prop_decrease', v));
  const atten = num('DFN atten dB', 0, 60, 1, p.atten_lim_db, (v) => setParam('atten_lim_db', v));
  const dmix = num('DFN mix', 0, 1, 0.05, p.dfn_mix, (v) => setParam('dfn_mix', v));
  const rmodel = sel('rnn model', RNN_MODELS, p.rnn_model, (v) => setParam('rnn_model', v));
  const rmix = num('rnn mix', 0, 1, 0.05, p.rnn_mix, (v) => setParam('rnn_mix', v));
  [method, prop, atten, dmix, rmodel, rmix].forEach((el) => box.appendChild(el));
}

function setEff(key, val) {
  const e = effective(state.focus);
  e[key] = val;
  state.overrides[state.focus] = e;
}
function setParam(key, val) {
  const e = effective(state.focus);
  e.params[key] = val;
  state.overrides[state.focus] = e;
}

function sel(label, opts, value, onchange) {
  const l = document.createElement('label'); l.textContent = label;
  const s = document.createElement('select');
  opts.forEach((o) => { const op = document.createElement('option');
    op.value = o; op.textContent = o; if (o === value) op.selected = true; s.appendChild(op); });
  s.onchange = () => onchange(s.value); l.appendChild(s); return l;
}
function num(label, min, max, step, value, onchange) {
  const l = document.createElement('label'); l.textContent = label;
  const i = document.createElement('input');
  i.type = 'range'; i.min = min; i.max = max; i.step = step; i.value = value;
  i.oninput = () => onchange(parseFloat(i.value)); l.appendChild(i); return l;
}

function currentTrim() {
  const r = Object.values(regions.getRegions())[0];
  return r ? [r.start, r.end] : [0, 0];
}

async function renderFocus() {
  const t = state.tracks[state.focus];
  const e = effective(state.focus);
  $('#status').textContent = 'rendering…';
  const res = await api('/api/render', {
    path: t.path, method: e.method, params: e.params,
    trim: currentTrim(), mode: 'normal',
  });
  state.cleaned[state.focus] = res.out_path;
  $('#status').textContent =
    `noise floor ${res.before.noise_floor_dbfs.toFixed(1)} → ${res.after.noise_floor_dbfs.toFixed(1)} dB`;
  toggleAB(true);
}

function toggleAB(showCleaned) {
  state.showCleaned = showCleaned;
  const t = state.tracks[state.focus];
  const cleaned = state.cleaned[state.focus];
  ws.load(audioUrl(showCleaned && cleaned ? cleaned : t.path));
}

function focusTrack(i) {
  state.focus = i;
  renderTrackList();
  ws.load(audioUrl(state.tracks[i].path));
  renderParams();
}

async function loadSession() {
  const paths = $('#paths').value.split(',').map((s) => s.trim()).filter(Boolean);
  const notice = $('#merge-notice');
  try {
    const res = await api('/api/session', { paths });
    state.tracks = res.tracks;
    state.mergeable = res.mergeable;
    state.focus = 0;
    $('#merge-btn').disabled = !res.mergeable;
    if (!res.mergeable) { notice.textContent = res.reason; notice.classList.remove('hidden'); }
    else notice.classList.add('hidden');
    renderTrackList();
    if (state.tracks.length) focusTrack(0);
  } catch (err) {
    notice.textContent = err.message;
    notice.classList.remove('hidden');
  }
}

$('#load-btn').onclick = loadSession;
$('#play').onclick = () => ws.playPause();
$('#render').onclick = renderFocus;
$('#ab').onclick = () => toggleAB(!state.showCleaned);

export { state, ws, focusTrack };  // for later tasks / debugging
