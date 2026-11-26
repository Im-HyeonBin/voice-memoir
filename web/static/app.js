'use strict';
// 목소리 발굴 화면. 서버(web/server.py)와 주고받는 것 말고는 바깥으로 아무것도 보내지 않는다.

const MOSS = '#2f5d50', OTHER = '#cfcbbf', IDLE = '#9aa39d', SOFT = '#a9c2b8';
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const ORDER = ['home', 'scan', 'pick', 'result'];
const SHOWN = 6;

const S = {
  sid: null, limits: null, status: null, voices: null, picked: new Set(), lastPick: [],
  view: 'year', result: null, localFiles: [], polling: false, showAll: false,
};

// ── 도우미 ─────────────────────────────────────────────
function esc(t) {
  return String(t).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function mmss(sec) {
  return sec >= 60 ? `${Math.floor(sec / 60)}분 ${Math.floor(sec % 60)}초` : `${sec.toFixed(1)}초`;
}
function clock(sec) {
  const s = Math.round(sec);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}
function hasFinal(word) {
  const c = (word || '').slice(-1).charCodeAt(0);
  return c >= 0xAC00 && c <= 0xD7A3 && (c - 0xAC00) % 28 !== 0;
}
function josa(word, withFinal, without) { return word + (hasFinal(word) ? withFinal : without); }
function bytes(n) { return n >= 1048576 ? `${(n / 1048576).toFixed(1)}MB` : `${Math.max(1, Math.round(n / 1024))}KB`; }

let toastTimer;
function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 5200);
}

class ApiError extends Error {
  constructor(msg, status) { super(msg); this.status = status; }
}
async function api(method, path, body, extra = {}) {
  const opts = { method, headers: {} };
  if (body instanceof Blob) {
    opts.body = body;
    opts.headers['Content-Type'] = 'application/octet-stream';
  } else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers['Content-Type'] = 'application/json';
  }
  const url = path.startsWith('/api/') ? path : `/api/sessions/${S.sid}${path}`;
  let res;
  try {
    res = await fetch(url, { ...opts, ...extra });
  } catch (e) {
    throw new ApiError('서버에 연결하지 못했어요. 인터넷 연결을 확인해 주세요.', 0);
  }
  if (!res.ok) {
    let msg = '잠시 문제가 생겼어요. 다시 시도해 주세요.';
    try { msg = (await res.json()).error || msg; } catch (e) { /* 본문 없음 */ }
    if (res.status === 404 && S.sid) lost(msg);
    throw new ApiError(msg, res.status);
  }
  return res.json();
}
function apiUrl(path) { return `/api/sessions/${S.sid}${path}`; }

// ── 파형 ───────────────────────────────────────────────
function strip(env, colorAt, h, w = 10, bw = 6) {
  const n = env.length;
  let s = `<svg viewBox="0 0 ${n * w} ${h}" preserveAspectRatio="none" aria-hidden="true">`;
  env.forEach((v, i) => {
    const bh = Math.max(2, v * (h - 4));
    s += `<rect x="${i * w + (w - bw) / 2}" y="${((h - bh) / 2).toFixed(1)}" width="${bw}" height="${bh.toFixed(1)}" rx="${bw / 2}" fill="${colorAt(i)}"/>`;
  });
  return s + '</svg>';
}
function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
function heroBand() {
  // 여러 사람이 번갈아 말하는 녹음 모양. 회색으로 그린 뒤 찾는 사람 차례만 초록으로 바뀐다.
  const r = rng(11), n = 150, env = new Array(n).fill(0.05), turns = [];
  let i = 1, t = 0;
  while (i < n - 4) {
    const len = Math.min(n - 1 - i, 6 + Math.floor(r() * 14));
    for (let k = 0; k < len; k++) {
      const s = Math.pow(Math.sin(0.15 + (Math.PI - 0.3) * k / Math.max(1, len - 1)), 0.6);
      env[i + k] = Math.max(0.08, Math.min(1, s * (0.45 + 0.55 * r())));
    }
    turns.push([i, i + len, t++]);
    i += len + 1 + Math.floor(r() * 4);
  }
  const el = $('#heroBand');
  el.innerHTML = strip(env, () => OTHER, 84);
  const rects = $$('rect', el);
  turns.filter(t => [1, 4, 6, 9].includes(t[2])).forEach(([a, b], j) => setTimeout(() => {
    for (let k = a; k < b; k++) rects[k].setAttribute('fill', MOSS);
  }, 700 + j * 550));
}

// ── 재생 ───────────────────────────────────────────────
// 소리는 하나만 재생한다. 누른 버튼 옆 파형이 재생한 만큼 짙어진다.
const PLAY = '<svg width="14" height="16" viewBox="0 0 14 16" aria-hidden="true"><path d="M2 1.5v13l11-6.5z" fill="#fff"/></svg>';
const PAUSE = '<svg width="14" height="16" viewBox="0 0 14 16" aria-hidden="true"><rect x="2" y="1.5" width="3.5" height="13" rx="1" fill="#fff"/><rect x="8.5" y="1.5" width="3.5" height="13" rx="1" fill="#fff"/></svg>';
const player = new Audio();
player.preload = 'auto';
let cur = null;
function face(btn, label, playing) {
  if (label) btn.textContent = playing ? '멈추기' : label;
  else btn.innerHTML = playing ? PAUSE : PLAY;
  btn.setAttribute('aria-pressed', String(playing));
}
function stopPlaying() {
  if (!cur) return;
  player.pause();
  face(cur.btn, cur.label, false);
  cur.rects.forEach(r => r.setAttribute('fill', cur.base));
  cur = null;
}
// label을 주면 글자 버튼(재생 중에는 '멈추기'), 없으면 동그란 재생 버튼
function bindPlay(btn, waveEl, url, base, label = null) {
  face(btn, label, false);
  const toggle = () => {
    if (cur && cur.btn === btn) { stopPlaying(); return; }
    stopPlaying();
    cur = { btn, rects: waveEl ? $$('rect', waveEl) : [], base, label };
    face(btn, label, true);
    player.src = url;
    player.play().catch(() => { stopPlaying(); toast('소리를 재생하지 못했어요.'); });
  };
  btn.addEventListener('click', toggle);
  if (waveEl) waveEl.addEventListener('click', toggle);
}
player.addEventListener('timeupdate', () => {
  if (!cur || !player.duration) return;
  const upto = Math.floor(cur.rects.length * player.currentTime / player.duration);
  cur.rects.forEach((r, i) => r.setAttribute('fill', i < upto ? MOSS : cur.base));
});
player.addEventListener('ended', stopPlaying);

// ── 화면 전환 ──────────────────────────────────────────
function go(name) {
  document.body.dataset.screen = name;
  $$('.screen').forEach(s => s.classList.toggle('on', s.id === 's-' + name));
  const idx = ORDER.indexOf(name);
  $$('.steps span').forEach(sp => {
    const i = ORDER.indexOf(sp.dataset.s);
    sp.className = i === idx ? 'on' : i < idx ? 'done' : '';
  });
  stopPlaying();
  scrollTo({ top: 0 });
}
function setSession(sid) {
  S.sid = sid;
  document.body.classList.toggle('has-session', !!sid);
  try { sid ? sessionStorage.setItem('vm_sid', sid) : sessionStorage.removeItem('vm_sid'); } catch (e) { /* 저장 못 해도 동작 */ }
}
function lost(msg) {
  setSession(null);
  S.status = S.voices = S.result = null;
  go('home');
  toast(msg);
}
async function newSession() {
  if (S.sid) {
    try { await api('DELETE', ''); } catch (e) { /* 이미 지워졌으면 그대로 */ }
  }
  const r = await api('POST', '/api/sessions');
  setSession(r.id);
  S.limits = r.limits;
  S.picked = new Set();
  S.result = S.voices = null;
  S.view = 'year';
}

// ── 1. 올리기 ──────────────────────────────────────────
function acceptable(file) {
  const ext = file.name.includes('.') ? file.name.split('.').pop().toLowerCase() : '';
  const exts = S.limits ? S.limits.exts : ['m4a', 'mp3', 'wav', 'aac', 'flac', 'ogg', 'opus', 'amr', '3gp', 'mp4', 'mov', 'm4v', 'webm', 'mkv', 'avi'];
  return exts.includes(ext);
}
async function startUpload(fileList) {
  let files = [...fileList];
  try {
    await newSession();
  } catch (e) { toast(e.message); return; }
  const bad = files.filter(f => !acceptable(f));
  files = files.filter(acceptable);
  if (bad.length) toast(`${bad.length}개 파일은 녹음 형식이 아니라 뺐어요.`);
  if (!files.length) { go('home'); return; }
  if (files.length > S.limits.files) {
    toast(`한 번에 파일 ${S.limits.files}개까지 올릴 수 있어요. 앞의 ${S.limits.files}개만 올릴게요.`);
    files = files.slice(0, S.limits.files);
  }
  S.localFiles = files.map(f => ({ file: f, name: f.name, state: 'wait', pct: 0, msg: '' }));
  go('scan');
  $('#scanTitle').textContent = '녹음을 올리고 있어요';
  $('#scanNote').textContent = `파일 ${files.length}개 · 모두 ${bytes(files.reduce((a, f) => a + f.size, 0))}`;
  renderUploadRows();
  let ok = 0;
  for (const item of S.localFiles) {
    if (!S.sid) return;
    try {
      await uploadOne(item);
      item.state = 'done';
      ok++;
    } catch (e) {
      item.state = 'bad';
      item.msg = e.message;
      if (e.status === 404) return;
    }
    renderUploadRows();
  }
  if (!ok) {
    $('#meterNote').textContent = '올린 파일을 하나도 읽지 못했어요. 다른 파일로 다시 해 주세요.';
    return;
  }
  try {
    await api('POST', '/scan');
    $('#scanTitle').textContent = '녹음을 살펴보고 있어요';
    poll();
  } catch (e) { toast(e.message); }
}
async function uploadOne(item) {
  const f = item.file, chunk = Math.min(S.limits.chunk, 8 * 1024 * 1024);
  item.state = 'up';
  renderUploadRows();
  const u = await api('POST', '/uploads', { name: f.name, size: f.size, modified: f.lastModified || null });
  for (let off = 0; off < f.size; off += chunk) {
    await api('PUT', `/uploads/${u.id}?offset=${off}`, f.slice(off, off + chunk));
    item.pct = Math.min(100, Math.round((off + chunk) / f.size * 100));
    renderUploadRows();
  }
  item.state = 'check';
  renderUploadRows();
  const done = await api('POST', `/uploads/${u.id}/done`);
  item.seconds = done.seconds;
}
function renderUploadRows() {
  const items = S.localFiles;
  const finished = items.filter(i => i.state === 'done' || i.state === 'bad').length;
  $('#meter').style.width = `${items.length ? finished / items.length * 100 : 0}%`;
  $('#meterNote').textContent = `${items.length}개 중 ${finished}개 올림`;
  $('#files').innerHTML = items.map(i => {
    const st = { wait: '기다리는 중', up: `올리는 중 ${i.pct}%`, check: '확인하는 중…', done: '올림', bad: '올리지 못함' }[i.state];
    const cls = i.state === 'done' ? 'ok' : i.state === 'bad' ? 'bad' : '';
    return `<div class="frow"><span class="name" title="${esc(i.name)}">${esc(i.name)}</span>
      <span class="meta d">${i.state === 'bad' ? esc(i.msg) : ''}</span>
      <span class="meta l">${i.seconds ? clock(i.seconds) : bytes(i.file.size)}</span>
      <span class="st ${cls}">${st}</span></div>`;
  }).join('');
}
async function startDemo() {
  try {
    await newSession();
    S.localFiles = [];
    await api('POST', '/demo');
  } catch (e) { toast(e.message); return; }
  go('scan');
  $('#scanTitle').textContent = '예시 녹음을 살펴보고 있어요';
  $('#scanNote').textContent = '가족 모임, 통화 녹음, 카카오톡 영상처럼 꾸민 예시 녹음이에요. 공개 음성 자료(Zeroth-Korean, LibriSpeech, VOiCES, CC BY 4.0)로 만들었어요.';
  $('#files').innerHTML = '';
  poll();
}

// ── 상태 따라가기 ─────────────────────────────────────
async function poll() {
  if (S.polling) return;
  S.polling = true;
  try {
    for (;;) {
      if (!S.sid) return;
      const st = await api('GET', '');
      S.status = st;
      showProgress(st);
      if (!['queued', 'scanning', 'searching'].includes(st.state)) break;
      await new Promise(r => setTimeout(r, 800));
    }
    await settle(S.status);
  } catch (e) {
    if (e.status !== 404) toast(e.message);
  } finally {
    S.polling = false;
  }
}
function waitingText(st) {
  return st.queue ? `앞에 ${st.queue}명이 기다리고 있어요. 차례가 되면 바로 시작해요.` : '';
}
function showProgress(st) {
  const scanning = st.job === 'scan' && ['queued', 'scanning'].includes(st.state);
  if (scanning) {
    if (document.body.dataset.screen !== 'scan') go('scan');
    const p = st.progress || { done: 0, total: 0, names: [] };
    $('#scanTitle').textContent = S.status.demo ? '예시 녹음을 살펴보고 있어요' : '녹음을 살펴보고 있어요';
    $('#meter').style.width = p.total ? `${p.done / p.total * 100}%` : '4%';
    $('#meterNote').textContent = st.state === 'queued' ? waitingText(st)
      : p.total && p.done >= p.total ? '다 살펴봤어요. 목소리를 사람별로 나누는 중이에요…'
        : p.total ? `${p.total}개 중 ${p.done}개 살펴봄` : '살펴볼 준비를 하는 중이에요…';
    const done = new Set(p.names);
    if (S.localFiles.length) {
      $$('#files .frow').forEach((row, i) => {
        const it = S.localFiles[i];
        if (!it || it.state !== 'done') return;
        const st2 = $('.st', row);
        st2.textContent = done.has(it.name) ? '살펴봄' : '살펴보는 중…';
        st2.className = 'st' + (done.has(it.name) ? ' ok' : '');
      });
    } else {
      $('#files').innerHTML = p.names.map(n => `<div class="frow"><span class="name">${esc(n)}</span><span class="meta d"></span><span class="meta l"></span><span class="st ok">살펴봄</span></div>`).join('');
    }
  }
  const searching = st.job === 'search' && ['queued', 'searching'].includes(st.state);
  $('#working').hidden = !searching;
  $('#resultBody').classList.toggle('dim', searching && !!S.result);
  if (searching) {
    $('#workingText').textContent = st.state === 'queued' ? waitingText(st) : '녹음마다 목소리를 찾고 한 번 더 확인하는 중이에요…';
    if (document.body.dataset.screen !== 'result') { go('result'); if (!S.result) renderResultShell(); }
  }
}
async function settle(st) {
  if (st.message) toast(st.message);
  if (st.state === 'error') {
    if (!st.voices_ready) { $('#meterNote').textContent = st.message || '녹음을 살펴보지 못했어요.'; return; }
  }
  if (st.state === 'ready' || (st.state === 'error' && st.voices_ready)) {
    if (!S.voices || st.job === 'scan') await loadVoices();
    if (document.body.dataset.screen !== 'pick') go('pick');
    return;
  }
  if (st.state === 'done' && st.has_result) {
    if (!S.voices) await loadVoices();
    await loadResult();
  }
}

// ── 2. 목소리 고르기 ───────────────────────────────────
async function loadVoices() {
  const r = await api('GET', '/voices');
  S.voices = r.voices;
  if (r.picked && r.picked.length) S.picked = new Set(r.picked);
  S.showAll = S.voices.slice(SHOWN).some(v => S.picked.has(v.index));
  renderPick();
}
function renderPick() {
  const st = S.status, sum = st.summary;
  if (sum) {
    let t = `파일 ${sum.files}개, 모두 ${mmss(sum.seconds)} 분량을 살펴봤어요. 그중 사람이 말하는 부분은 ${mmss(sum.speech)}예요.`;
    if (sum.phone) t += ` 통화 녹음으로 보이는 파일 ${sum.phone}개는 통화 음질에 맞춰 따로 비교해요.`;
    if (sum.errors) t += ` 읽지 못한 파일이 ${sum.errors}개 있어요.`;
    $('#pickSummary').textContent = t;
    $('#fileTable').innerHTML = (st.files || []).map(f => `<div class="frow">
      <span class="name" title="${esc(f.name)}">${esc(f.name)}</span>
      <span class="meta d">${f.date ? esc(f.date) + ' · ' + esc(f.basis) : '날짜 모름'}</span>
      <span class="meta l">${clock(f.seconds)}</span>
      <span class="st ${f.error ? 'bad' : f.phone ? '' : 'ok'}">${f.error ? '읽지 못함' : f.phone ? '통화 녹음' : '살펴봄'}</span></div>`).join('');
  }
  $('#demoRefBtn').hidden = !st.has_demo_ref;
  if (!$('#who').value && st.person && st.person !== '찾는 사람') $('#who').value = st.person;
  const list = S.voices || [];
  if (!list.length) {
    $('#voices').innerHTML = '<p class="empty">말소리를 충분히 찾지 못했어요. 녹음이 너무 짧거나 조용하지 않은지 확인해 주세요. 참고 녹음으로 찾아볼 수도 있어요.</p>';
  } else {
    const shown = S.showAll ? list : list.slice(0, SHOWN);
    $('#voices').innerHTML = shown.map(v => `<div class="vrow${S.picked.has(v.index) ? ' sel' : ''}" data-i="${v.index}">
      <button class="play" aria-label="목소리 ${v.index} 들어 보기"></button>
      <div><h3>목소리 ${v.index}${v.phone ? '<span class="tag">통화</span>' : ''}</h3><div class="meta">파일 ${v.files}개 · 약 ${mmss(v.seconds)}</div></div>
      <div class="wave">${strip(v.env, () => IDLE, 40)}</div>
      <button class="pickbtn" aria-pressed="${S.picked.has(v.index)}"><span class="tick"><svg width="10" height="8" viewBox="0 0 10 8" aria-hidden="true"><path d="M1 4l3 3 5-6" stroke="#fff" stroke-width="1.8" fill="none"/></svg></span>이 사람이에요</button>
    </div>`).join('');
    $$('#voices .vrow').forEach(row => {
      const i = +row.dataset.i;
      bindPlay($('.play', row), $('.wave', row), apiUrl(`/voices/${i}.wav`), IDLE);
      $('.pickbtn', row).addEventListener('click', () => {
        S.picked.has(i) ? S.picked.delete(i) : S.picked.add(i);
        row.classList.toggle('sel', S.picked.has(i));
        $('.pickbtn', row).setAttribute('aria-pressed', S.picked.has(i));
        updateFind();
      });
    });
  }
  const rest = list.length - SHOWN;
  $('#moreBtn').hidden = S.showAll || rest <= 0;
  $('#moreBtn').textContent = `짧게 나온 목소리 ${rest}개 더 보기`;
  updateFind();
}
let refFile = null;
function activeTab() { return $('.tabs button.on').dataset.tab; }
function updateFind() {
  const b = $('#findBtn');
  if (activeTab() === 'ref') {
    b.disabled = !refFile;
    b.textContent = refFile ? '참고 녹음으로 찾기' : '참고 녹음을 올려 주세요';
  } else {
    const n = S.picked.size;
    b.disabled = n === 0;
    b.textContent = n === 0 ? '목소리를 골라 주세요' : `고른 목소리 ${n}개로 찾기`;
  }
}
function personName() { return ($('#who').value || '').trim() || '찾는 사람'; }
async function find() {
  const b = $('#findBtn');
  b.disabled = true;
  try {
    if (activeTab() === 'ref') {
      if (refFile.size > 50 * 1024 * 1024) throw new ApiError('참고 녹음은 50MB까지 올릴 수 있어요.', 400);
      await api('PUT', `/reference?name=${encodeURIComponent(refFile.name)}&person=${encodeURIComponent(personName())}`, refFile);
    } else {
      S.lastPick = [...S.picked];
      await api('POST', '/pick', { voices: S.lastPick, person: personName() });
    }
    S.result = null;
    go('result');
    renderResultShell();
    poll();
  } catch (e) {
    toast(e.message);
  } finally {
    updateFind();
  }
}

// ── 3. 결과 ────────────────────────────────────────────
function renderResultShell() {
  $('#rTitle').textContent = `${personName()}의 목소리, ${S.view === 'date' ? '날짜마다' : '해마다'}`;
  $('#rSum').textContent = '';
  ['#notices', '#ask', '#timeline', '#reviewed'].forEach(id => { $(id).innerHTML = ''; });
  $('#working').hidden = false;
}
async function loadResult() {
  const r = await api('GET', `/result?view=${S.view}`);
  S.result = r;
  if (document.body.dataset.screen !== 'result') go('result');
  renderResult();
}
function rowClip(c, label) {
  return `<div class="row" data-f="${c.file}" data-s="${c.start}" data-e="${c.end}"${c.overlap ? ' data-o="1"' : ''}>
    <button class="play small" aria-label="들어 보기"></button>
    <div><div class="t">${esc(c.name)}</div><div class="m">${esc(label)}</div></div>`;
}
function clipUrl(el) { return apiUrl(`/clip.wav?file=${el.dataset.f}&start=${el.dataset.s}&end=${el.dataset.e}`); }
function renderResult() {
  const r = S.result, person = r.person;
  $('#working').hidden = true;
  $('#resultBody').classList.remove('dim');
  $('#rTitle').textContent = `${person}의 목소리, ${r.view === 'date' ? '날짜마다' : '해마다'}`;
  $('#zipBtn').href = apiUrl('/download.zip');
  $('#zipBtn').setAttribute('aria-disabled', r.total > 0 ? 'false' : 'true');
  $('#rSum').textContent = r.total > 0
    ? `${r.years}개 해, 녹음 ${r.n_files}개에서 모두 ${mmss(r.total)}를 찾았어요.` : '';
  $$('#viewSeg button').forEach(b => b.classList.toggle('on', b.dataset.v === r.view));
  $('#levelSeg').innerHTML = r.levels.map((l, i) => `<button data-l="${i}" class="${i === r.level ? 'on' : ''}">${l}</button>`).join('');

  // 알림: 참고 녹음이 섞였음, 함께 따라간 목소리, 같은 사람인지 물어볼 목소리
  let notes = '';
  if (r.notice) notes += `<div class="box"><p>${esc(r.notice)}</p></div>`;
  if (r.followed.length) {
    const names = r.followed.slice(0, 8).map(i => `목소리 ${i}`).join(', ') + (r.followed.length > 8 ? ` 등 ${r.followed.length}개` : '');
    notes += `<div class="box"><h3>같은 사람으로 보이는 목소리도 함께 찾았어요</h3>
      <p>고른 목소리와 확실히 같은 사람으로 보이는 ${esc(names)}를 함께 썼어요. 멀리서 녹음했거나 녹음이 길면 한 사람이 여러 목소리로 나뉘어 보이기 때문이에요. 다른 사람이 섞여 보이면 고른 목소리만으로 다시 찾을 수 있어요.</p>
      <div class="acts"><button class="btn ghost small" id="onlyPicked">고른 목소리만으로 찾기</button></div></div>`;
  }
  if (r.unsure.length) {
    notes += `<div class="box" id="unsureBox"><h3>이 목소리도 같은 사람인가요?</h3>
      <p>참고 녹음과 비슷하지만 같은 사람인지 확실하지 않아 지금은 빼 두었어요. 같은 사람이면 알려 주세요. 시끄러운 곳이나 멀리서 녹음된 목소리를 더 찾을 수 있어요.</p>
      ${r.unsure.map(v => `<div class="row" data-i="${v.index}"><button class="play small" aria-label="목소리 ${v.index} 들어 보기"></button>
        <div><div class="t">목소리 ${v.index}</div><div class="m">파일 ${v.files}개 · 약 ${mmss(v.seconds)}</div></div>
        <div class="yn"><button class="yes">같은 사람</button><button class="no">다른 사람</button></div></div>`).join('')}</div>`;
  }
  $('#notices').innerHTML = notes;
  const only = $('#onlyPicked');
  if (only) only.addEventListener('click', () => research(() => api('POST', '/pick', { voices: S.lastPick.length ? S.lastPick : [...S.picked], follow: false })));
  $$('#unsureBox .row').forEach(row => {
    const i = +row.dataset.i;
    bindPlay($('.play', row), null, apiUrl(`/voices/${i}.wav`), IDLE);
    $('.yes', row).addEventListener('click', () => research(() => api('POST', '/same', { index: i, yes: true })));
    $('.no', row).addEventListener('click', async () => {
      try { await api('POST', '/same', { index: i, yes: false }); row.remove(); } catch (e) { toast(e.message); }
    });
  });

  // 확인해 주세요
  if (r.maybe.length) {
    $('#ask').innerHTML = `<div class="box"><h3>확인해 주세요 ${r.maybe.length + r.maybe_more}</h3>
      <p>기준에 조금 못 미치지만, 녹음에 나온 다른 누구보다 ${esc(josa(person, '과', '와'))} 더 닮은 구간이에요. 들어 보고 알려 주시면 다음 찾기에도 반영해요.</p>
      ${r.maybe.map(m => rowClip(m, `${m.date} · ${m.start.toFixed(1)}초부터 ${m.end.toFixed(1)}초까지${m.overlap ? ' · 다른 목소리와 겹친 구간' : ''}`)
        + '<div class="yn"><button class="yes">맞아요</button><button class="no">아니에요</button></div></div>').join('')}
      ${r.maybe_more ? `<p class="m">나머지 ${r.maybe_more}개는 위 구간을 확인하면 이어서 보여 드려요.</p>` : ''}</div>`;
    $$('#ask .row').forEach(row => {
      bindPlay($('.play', row), null, clipUrl(row), IDLE);
      $('.yes', row).addEventListener('click', () => review(row, 'yes'));
      $('.no', row).addEventListener('click', () => review(row, 'no'));
    });
  } else {
    $('#ask').innerHTML = '';
  }

  // 해마다 / 날짜마다
  if (!r.groups.length) {
    $('#timeline').innerHTML = `<p class="empty">이 기준으로는 목소리를 찾지 못했어요. 기준을 넉넉하게 바꾸거나, 다른 목소리를 함께 골라 보세요.</p>`;
  } else {
    $('#timeline').innerHTML = r.groups.map((g, k) => {
      const files = g.files.map(f => {
        const spans = f.spans, n = f.env.length;
        const on = i => { const c = (i + 0.5) / n; return spans.some(([a, b]) => a <= c && c <= b); };
        return `<div class="row"><div class="fn">${esc(f.name)}${f.phone ? '<span class="tag">통화</span>' : ''}</div>
          <div class="fd">${esc(f.date)}</div><div class="strip">${strip(f.env, i => on(i) ? MOSS : OTHER, 26)}</div></div>`;
      }).join('');
      const fname = `${person}_${g.title}_목소리.wav`.replace(/\s+/g, '_');
      const full = g.highlight && g.full_seconds > g.main.seconds + 1;
      return `<div class="yr" data-k="${esc(g.key)}"><div class="y${g.highlight ? '' : ' small'}">${esc(g.title)}</div>
        <div class="ysum">파일 ${g.files.length}개에서 ${mmss(g.seconds)}를 찾았어요${g.highlight && full ? ` · 가장 또렷한 ${mmss(g.main.seconds)}를 모았어요` : ''}</div>
        <div class="hl"><button class="play" aria-label="${esc(g.title)} 목소리 듣기"></button>
          <div class="wave">${strip(g.main.env, () => SOFT, 40)}</div><div class="time">${clock(g.main.seconds)}</div></div>
        <div class="hlnote">
          <a href="${apiUrl(`/groups/${r.view}/${encodeURIComponent(g.key)}.wav`)}" download="${esc(fname)}">음성 파일 받기</a>
          ${full ? `<button class="linkbtn fullbtn">이 해에 찾은 목소리 전부 듣기 (${mmss(g.full_seconds)})</button>` : ''}
        </div>
        <div class="src">${files}</div>
        <details class="clips"><summary>찾은 구간 ${g.clips.length}개 하나씩 듣기</summary>
          ${g.clips.map(c => `<div class="row" data-f="${c.file}" data-s="${c.start}" data-e="${c.end}">
            <button class="play small" aria-label="들어 보기"></button>
            <div>${esc(c.name)}<div class="m">${c.start.toFixed(1)}초부터 ${c.end.toFixed(1)}초까지${c.confirmed ? ' · 직접 확인함' : ''}</div></div>
            <button class="nobtn">이 사람 아니에요</button></div>`).join('')}
        </details></div>`;
    }).join('');
    $$('#timeline .yr').forEach((el, k) => {
      const g = r.groups[k];
      const url = apiUrl(`/groups/${r.view}/${encodeURIComponent(g.key)}.wav`);
      bindPlay($('.hl .play', el), $('.hl .wave', el), url, SOFT);
      const fb = $('.fullbtn', el);
      if (fb) bindPlay(fb, null, url + '?full=1', SOFT, `이 해에 찾은 목소리 전부 듣기 (${mmss(g.full_seconds)})`);
      $$('.clips .row', el).forEach(row => {
        bindPlay($('.play', row), null, clipUrl(row), IDLE);
        $('.nobtn', row).addEventListener('click', () => review(row, 'no'));
      });
    });
  }

  // 직접 확인한 구간
  if (r.reviewed.length) {
    const yes = r.reviewed.filter(x => x.answer === 'yes').length, no = r.reviewed.length - yes;
    $('#reviewed').innerHTML = `<details class="box clips"><summary>직접 확인한 구간 ${r.reviewed.length}개 (맞아요 ${yes}, 아니에요 ${no}) 보기와 되돌리기</summary>
      ${r.reviewed.map(x => `<div class="row" data-f="${x.file}" data-s="${x.start}" data-e="${x.end}">
        <button class="play small" aria-label="들어 보기"></button>
        <div>${esc(x.name)}<div class="m">${x.start.toFixed(1)}초부터 ${x.end.toFixed(1)}초까지 · ‘${x.answer === 'yes' ? '맞아요' : '아니에요'}’로 확인함</div></div>
        <button class="nobtn">되돌리기</button></div>`).join('')}</details>`;
    $$('#reviewed .row').forEach(row => {
      bindPlay($('.play', row), null, clipUrl(row), IDLE);
      $('.nobtn', row).addEventListener('click', () => review(row, 'undo'));
    });
  } else {
    $('#reviewed').innerHTML = '';
  }
}
async function research(call) {
  try {
    await call();
    $('#working').hidden = false;
    $('#resultBody').classList.add('dim');
    poll();
  } catch (e) { toast(e.message); }
}
function review(row, answer) {
  research(() => api('POST', '/review', {
    file: +row.dataset.f, start: +row.dataset.s, end: +row.dataset.e, answer, overlap: row.dataset.o === '1',
  }));
}

// ── 시작 ───────────────────────────────────────────────
function wireDrop(el, onFiles) {
  ['dragenter', 'dragover'].forEach(e => el.addEventListener(e, ev => { ev.preventDefault(); el.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(e => el.addEventListener(e, ev => { ev.preventDefault(); el.classList.remove('over'); }));
  el.addEventListener('drop', ev => { if (ev.dataTransfer.files.length) onFiles(ev.dataTransfer.files); });
}
function setRef(file) {
  if (!file) return;
  refFile = file;
  $('#refName').textContent = `${file.name} (${bytes(file.size)}) ·`;
  $('#refDrop').classList.add('has');
  updateFind();
}

async function boot() {
  heroBand();
  try {
    const info = await api('GET', '/api/info');
    const L = info.limits;
    $('#limitsNote').textContent = `영상과 통화 녹음이 섞여 있어도 괜찮아요. 한 번에 ${L.files}개, 녹음 길이 모두 ${L.minutes}분까지 살펴봐요.`;
    $('#ttlNote').textContent = `마지막으로 쓰고 ${L.ttl_minutes >= 60 ? L.ttl_minutes / 60 + '시간' : L.ttl_minutes + '분'}이 지나면 올린 녹음과 결과를 모두 지워요.`;
    $('#demoBtn').hidden = !info.demo;
  } catch (e) { /* 안내 문구만 기본값으로 둔다 */ }
  addEventListener('scroll', () => $('#top').classList.toggle('scrolled', scrollY > 8), { passive: true });
  wireDrop($('#drop'), files => startUpload(files));
  $('#fileInput').addEventListener('change', e => { if (e.target.files.length) startUpload(e.target.files); e.target.value = ''; });
  $('#demoBtn').addEventListener('click', startDemo);
  $('#homeBtn').addEventListener('click', () => go('home'));
  $('#wipeBtn').addEventListener('click', async () => {
    if (!confirm('올린 녹음과 찾은 결과를 지금 모두 지울까요? 되돌릴 수 없어요.')) return;
    try { await api('DELETE', ''); } catch (e) { /* 이미 지워졌어도 화면은 처음으로 */ }
    setSession(null);
    S.status = S.voices = S.result = null;
    go('home');
    toast('올린 녹음과 결과를 모두 지웠어요.');
  });
  $$('.tabs button').forEach(b => b.addEventListener('click', () => {
    $$('.tabs button').forEach(x => { x.classList.toggle('on', x === b); x.setAttribute('aria-selected', x === b); });
    $('#tab-voices').hidden = b.dataset.tab !== 'voices';
    $('#tab-ref').hidden = b.dataset.tab !== 'ref';
    updateFind();
  }));
  $('#moreBtn').addEventListener('click', () => { S.showAll = true; renderPick(); });
  $('#refInput').addEventListener('change', e => setRef(e.target.files[0]));
  wireDrop($('#refDrop'), files => setRef(files[0]));
  $('#demoRefBtn').addEventListener('click', async () => {
    try {
      await api('POST', `/demo-reference?person=${encodeURIComponent(personName())}`);
      S.result = null; go('result'); renderResultShell(); poll();
    } catch (e) { toast(e.message); }
  });
  $('#findBtn').addEventListener('click', find);
  $('#backPick').addEventListener('click', () => { go('pick'); renderPick(); });
  $$('#viewSeg button').forEach(b => b.addEventListener('click', async () => {
    S.view = b.dataset.v;
    try { await loadResult(); } catch (e) { toast(e.message); }
  }));
  $('#levelSeg').addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b || b.classList.contains('on')) return;
    research(() => api('POST', '/settings', { level: +b.dataset.l }));
  });

  // 새로 고침해도 하던 곳에서 이어 간다
  let sid = null;
  try { sid = sessionStorage.getItem('vm_sid'); } catch (e) { /* 없음 */ }
  if (sid) {
    setSession(sid);
    try {
      const st = await api('GET', '');
      S.status = st;
      if (['queued', 'scanning', 'searching'].includes(st.state)) { showProgress(st); poll(); }
      else if (st.voices_ready) await settle(st);
      else go('home');
    } catch (e) { /* 지워진 세션이면 처음 화면 */ }
  }
}
document.addEventListener('DOMContentLoaded', boot);
