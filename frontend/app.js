import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js';

const state = { tracks: [], focus: 0, mergeable: false, mode: 'normal' };

const $ = (s) => document.querySelector(s);
const api = (path, body) =>
  fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body) }).then((r) => r.json());

const ws = WaveSurfer.create({
  container: '#waveform', waveColor: '#3b82f6', progressColor: '#1d4ed8',
  height: 160, normalize: true,
});

function audioUrl(path) {
  return `/api/audio?path=${encodeURIComponent(path)}`;
}

function renderTrackList() {
  const ul = $('#track-list');
  ul.innerHTML = '';
  state.tracks.forEach((t, i) => {
    const li = document.createElement('li');
    li.textContent = `${t.name}  ${t.duration.toFixed(1)}s`;
    if (i === state.focus) li.classList.add('active');
    li.onclick = () => focusTrack(i);
    ul.appendChild(li);
  });
}

function focusTrack(i) {
  state.focus = i;
  renderTrackList();
  ws.load(audioUrl(state.tracks[i].path));
}

async function loadSession() {
  const paths = $('#paths').value.split(',').map((s) => s.trim()).filter(Boolean);
  const res = await api('/api/session', { paths });
  state.tracks = res.tracks;
  state.mergeable = res.mergeable;
  state.focus = 0;
  const notice = $('#merge-notice');
  $('#merge-btn').disabled = !res.mergeable;
  if (!res.mergeable) { notice.textContent = res.reason; notice.classList.remove('hidden'); }
  else notice.classList.add('hidden');
  renderTrackList();
  if (state.tracks.length) focusTrack(0);
}

$('#load-btn').onclick = loadSession;
$('#play').onclick = () => ws.playPause();

export { state, ws, focusTrack };  // for later tasks / debugging
