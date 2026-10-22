'use strict';

const $ = (id) => document.getElementById(id);
const statusNames = { queued: 'Queued', running: 'Writing & checking', pause_requested: 'Pausing', paused: 'Paused', awaiting_review: 'Ready for review', completed: 'Complete', failed: 'Stopped' };
const state = { token: sessionStorage.getItem('book-token'), user: null, jobs: [], job: null, selected: null, view: 'review', busy: false, refreshing: null, generation: 0, listKey: '', readingKey: '', register: false, feedback: new Map() };
const date = (value) => new Date(value).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
const bookTitle = (job) => job.brief?.match(/^Title:\s*(.+)$/mi)?.[1].trim() || job.brief?.trim().split('\n')[0].slice(0, 200) || `Book ${job.id.slice(0, 6)}`;

function message(id, text) {
  $(id).textContent = text;
  $(id).hidden = !text;
}

async function request(path, options = {}, authenticated = true) {
  const token = state.token;
  const headers = new Headers(options.headers);
  if (authenticated && token) headers.set('Authorization', `Bearer ${token}`);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(path, { ...options, headers, signal: controller.signal, cache: 'no-store' });
    if (authenticated && token !== state.token) throw new Error('Your session changed. Sign in to continue.');
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      if (response.status === 401 && authenticated && token === state.token) {
        signOut('Your session has expired. Sign in to continue; your books are saved.');
      }
      const detail = Array.isArray(body.detail) ? body.detail.map((item) => `${item.loc.at(-1)}: ${item.msg}`).join('; ') : body.detail;
      throw new Error(detail || `The request failed (${response.status}). Try again.`);
    }
    return response;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The request timed out. Refresh to check whether it completed before trying again.');
    if (error instanceof TypeError) throw new Error('Cannot reach the studio. Check your connection and try Refresh.');
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function accountMode(register) {
  state.register = register;
  $('login-mode').setAttribute('aria-pressed', String(!register));
  $('register-mode').setAttribute('aria-pressed', String(register));
  $('auth-title').textContent = register ? 'Make room for your book.' : 'Welcome back.';
  $('auth-description').textContent = register ? 'Create an account with your registration key.' : 'Sign in to pick up where you left off.';
  $('auth-submit').textContent = register ? 'Create account' : 'Sign in';
  $('password').autocomplete = register ? 'new-password' : 'current-password';
  $('password').minLength = register ? 12 : 1;
  $('username').minLength = register ? 3 : 1;
  if (register) $('username').pattern = '[A-Za-z0-9_-]+';
  else $('username').removeAttribute('pattern');
  for (const id of ['invite-field', 'username-help', 'password-help']) $(id).hidden = !register;
  $('invite').required = register;
  message('auth-error', '');
  for (const id of ['username', 'password', 'invite']) $(id).removeAttribute('aria-invalid');
}

function signOut(reason = '') {
  state.generation += 1;
  state.token = null;
  state.user = null;
  state.jobs = [];
  state.job = null;
  state.selected = null;
  state.feedback.clear();
  state.listKey = '';
  state.readingKey = '';
  sessionStorage.removeItem('book-token');
  $('auth-form').reset();
  $('feedback-form').reset();
  $('book-form').reset();
  $('custom-book').open = false;
  $('manuscript').replaceChildren();
  $('job-list').replaceChildren();
  $('workspace').hidden = true;
  $('auth-screen').hidden = false;
  $('logout').hidden = true;
  $('account-name').hidden = true;
  accountMode(false);
  message('auth-error', reason);
  message('workspace-error', '');
  message('notice', '');
  $('username').focus();
}

async function enterWorkspace() {
  const token = state.token;
  const user = await (await request('/auth/me')).json();
  if (token !== state.token) return;
  state.user = user;
  $('account-name').textContent = user.username;
  $('account-name').hidden = false;
  $('logout').hidden = false;
  $('auth-screen').hidden = true;
  $('workspace').hidden = false;
  $('auth-form').reset();
  await refresh();
  $('books-title').tabIndex = -1;
  $('books-title').focus();
}

async function serviceHealth() {
  try {
    await request('/health', {}, false);
    $('service-status').textContent = 'Service available';
    $('service-status').className = 'service-status available';
  } catch {
    $('service-status').textContent = 'Service unavailable';
    $('service-status').className = 'service-status unavailable';
  }
}

function renderJobs() {
  const key = JSON.stringify([state.jobs, state.selected]);
  if (key === state.listKey) return;
  state.listKey = key;
  const focused = document.activeElement?.dataset.job;
  const fragment = document.createDocumentFragment();
  for (const job of state.jobs) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'job-button';
    button.dataset.job = job.id;
    button.setAttribute('aria-current', String(job.id === state.selected));
    const title = document.createElement('strong');
    title.textContent = bookTitle(job);
    const created = document.createElement('span');
    created.textContent = date(job.created_at);
    const status = document.createElement('span');
    status.className = 'job-state';
    status.textContent = `${statusNames[job.status] || job.status} · ${job.chapter_index}/3 approved`;
    button.append(title, created, status);
    button.addEventListener('click', () => selectJob(job.id));
    fragment.append(button);
  }
  $('job-list').replaceChildren(fragment);
  if (focused) [...$('job-list').children].find((button) => button.dataset.job === focused)?.focus();
  $('list-note').textContent = state.jobs.length ? (state.jobs.length === 100 ? 'Showing your latest 100 books.' : `${state.jobs.length} ${state.jobs.length === 1 ? 'book' : 'books'}`) : 'No books yet. Start one above.';
}

async function refresh() {
  if (!state.token || state.refreshing === state.token) return;
  const token = state.token;
  const generation = state.generation;
  state.refreshing = token;
  $('refresh').disabled = true;
  try {
    const jobs = await (await request('/jobs?limit=100')).json();
    if (generation !== state.generation || token !== state.token) return;
    state.jobs = jobs;
    if (!state.selected && jobs.length) state.selected = jobs[0].id;
    renderJobs();
    if (state.selected) {
      const id = state.selected;
      const job = await (await request(`/jobs/${id}`)).json();
      if (generation !== state.generation || id !== state.selected || token !== state.token) return;
      if (job.status === 'awaiting_review' && state.job?.status !== 'awaiting_review') state.view = 'review';
      state.job = job;
      renderJob();
    } else {
      $('empty-desk').hidden = false;
      $('book-desk').hidden = true;
    }
    message('workspace-error', '');
  } catch (error) {
    if (token === state.token && generation === state.generation) message('workspace-error', error.message);
  } finally {
    if (state.refreshing === token) state.refreshing = null;
    $('refresh').disabled = false;
  }
}

async function selectJob(id) {
  if (id === state.selected) return;
  state.generation += 1;
  state.selected = id;
  state.job = null;
  state.view = 'review';
  state.readingKey = '';
  $('feedback').value = state.feedback.get(id) || '';
  $('book-desk').hidden = true;
  message('feedback-error', '');
  message('notice', 'Loading book…');
  $('manuscript').replaceChildren();
  $('review-controls').hidden = true;
  renderJobs();
  // A previous refresh may still finish; its generation cannot update this selection.
  state.refreshing = null;
  await refresh();
  if (id === state.selected) message('notice', '');
}

function renderJob() {
  const job = state.job;
  if (!job) return;
  $('empty-desk').hidden = true;
  $('book-desk').hidden = false;
  $('book-title').textContent = bookTitle(job);
  $('book-date').textContent = `Book ${job.id.slice(0, 6)} · Started ${date(job.created_at)}`;
  $('job-status').textContent = statusNames[job.status] || job.status;
  $('job-status').className = `badge ${job.status}`;
  $('book-progress').value = job.chapter_index;
  $('progress-label').textContent = `${job.chapter_index} of 3 chapters approved`;
  const descriptions = {
    queued: 'Your book is in the queue. You can pause it before the next step.',
    running: `Chapter ${Math.min(job.chapter_index + 1, 3)} is being researched, written, and checked. It will stop for your review.`,
    pause_requested: 'Finishing the current step, then pausing. Your progress will be saved.',
    paused: 'Progress saved. Resume whenever you are ready.',
    awaiting_review: `Chapter ${job.review_chapter + 1} has passed its checks. Read it, then approve or request changes.`,
    completed: 'All three chapters are approved. Your final book is ready to download.',
    failed: 'This book stopped before completion. Approved chapters remain available. Start a new book after resolving the reported issue.',
  };
  if ($('status-description').textContent !== descriptions[job.status]) $('status-description').textContent = descriptions[job.status] || 'Refresh to check this book.';
  message('job-error', job.error || '');
  const pausable = ['queued', 'running'].includes(job.status);
  $('pause-resume').hidden = !pausable && job.status !== 'paused' && job.status !== 'pause_requested';
  $('pause-resume').textContent = job.status === 'paused' ? 'Resume book' : job.status === 'pause_requested' ? 'Pausing…' : 'Pause book';
  $('pause-resume').disabled = state.busy || job.status === 'pause_requested';
  $('final-downloads').hidden = job.status !== 'completed';
  const nav = $('chapter-nav');
  const navKey = JSON.stringify([job.chapter_index, job.status, state.view]);
  if (nav.dataset.key !== navKey) {
    nav.dataset.key = navKey;
    const focused = document.activeElement?.dataset.view;
    nav.replaceChildren();
    const views = [];
    for (let n = 1; n <= 3; n++) views.push({ key: String(n), label: `Chapter ${n}`, number: n, available: n <= job.chapter_index, approved: n <= job.chapter_index });
    if (job.status === 'awaiting_review') views.push({ key: 'review', label: 'Review draft', number: job.review_chapter + 1, available: true });
    if (job.status === 'completed') views.push({ key: 'book', label: 'Full book', number: '✓', available: true });
    if (state.view === 'review' && job.status === 'completed') state.view = 'book';
    for (const view of views) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = `chapter-button${view.approved ? ' approved' : ''}`;
      button.dataset.view = view.key; button.disabled = !view.available;
      button.setAttribute('aria-pressed', String(view.key === state.view));
      const number = document.createElement('span'); number.className = 'chapter-number'; number.textContent = view.number;
      const label = document.createElement('span'); label.textContent = view.label;
      button.append(number, label);
      button.addEventListener('click', () => { state.view = view.key; renderJob(); });
      nav.append(button);
    }
    if (focused) [...nav.children].find((button) => button.dataset.view === focused && !button.disabled)?.focus();
  }
  const reviewing = state.view === 'review' && job.status === 'awaiting_review';
  $('review-controls').hidden = !reviewing;
  $('approve').disabled = state.busy;
  $('request-revision').disabled = state.busy;
  $('chapter-download').hidden = !/^[1-3]$/.test(state.view) || Number(state.view) > job.chapter_index;
  $('reading-label').textContent = reviewing ? `Chapter ${job.review_chapter + 1} · Review draft` : state.view === 'book' ? 'Complete manuscript' : /^[1-3]$/.test(state.view) ? `Chapter ${state.view} · Approved` : 'Next chapter';
  loadReading();
}

// Build only text nodes and HTTPS links: retrieved Markdown is never trusted HTML.
function renderMarkdown(markdown) {
  const fragment = document.createDocumentFragment();
  for (const block of markdown.trim().split(/\n\s*\n/)) {
    const heading = block.match(/^(#{1,3})\s+(.*)$/s);
    const lines = /^\[\d+\]/.test(block) ? block.split('\n') : [block];
    for (const line of lines) {
      const node = document.createElement(heading ? (heading[1].length < 3 ? 'h2' : 'h3') : 'p');
      if (line.startsWith('Takeaway:')) node.className = 'takeaway';
      if (/^\[\d+\]/.test(line)) node.className = 'reference';
      const text = heading ? heading[2] : line;
      for (const part of text.split(/(<https:\/\/[^>]+>)/g)) {
        if (part.startsWith('<https://') && part.endsWith('>')) {
          let url;
          try { url = new URL(part.slice(1, -1)); }
          catch { node.append(document.createTextNode(part)); continue; }
          if (url.protocol === 'https:' && !url.username && !url.password) {
            const link = document.createElement('a'); link.href = url.href; link.textContent = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer'; node.append(link); continue;
          }
        }
        node.append(document.createTextNode(part));
      }
      fragment.append(node);
    }
  }
  return fragment;
}

async function loadReading() {
  const job = state.job;
  const id = state.selected;
  const view = state.view;
  const key = JSON.stringify([id, view, job.status, job.review_preview]);
  if (key === state.readingKey) return;
  state.readingKey = key;
  $('manuscript').setAttribute('aria-busy', 'true');
  const placeholder = document.createElement('p'); placeholder.className = 'placeholder'; placeholder.textContent = 'Loading manuscript…';
  $('manuscript').replaceChildren(placeholder);
  try {
    let markdown;
    if (view === 'review' && job.status === 'awaiting_review') markdown = job.review_preview;
    else if (view === 'book' && job.status === 'completed') markdown = await (await request(`/jobs/${id}/artifacts/book.md`)).text();
    else if (/^[1-3]$/.test(view) && Number(view) <= job.chapter_index) markdown = await (await request(`/jobs/${id}/artifacts/chapter-${view}.md`)).text();
    if (state.readingKey !== key || id !== state.selected || !state.token) return;
    if (markdown) $('manuscript').replaceChildren(renderMarkdown(markdown));
    else placeholder.textContent = job.status === 'paused' ? 'Your place is saved. Resume the book to continue.' : job.status === 'failed' ? 'Choose an approved chapter, or start a new book.' : 'The next chapter will appear here when it is ready for your review.';
  } catch (error) {
    if (state.readingKey === key && state.token) { placeholder.textContent = error.message; state.readingKey = ''; }
  } finally {
    if (state.readingKey === key || state.readingKey === '') $('manuscript').setAttribute('aria-busy', 'false');
  }
}

async function mutate(action, successMessage) {
  if (state.busy) return;
  state.busy = true;
  for (const id of ['new-book', 'empty-new-book', 'custom-start', 'approve', 'request-revision', 'pause-resume']) $(id).disabled = true;
  message('workspace-error', '');
  try { await action(); message('notice', successMessage); }
  catch (error) { if (state.token) message('workspace-error', error.message); }
  finally {
    state.busy = false;
    $('new-book').disabled = false; $('empty-new-book').disabled = false;
    $('custom-start').disabled = false;
    await refresh();
    if (state.job) renderJob();
  }
}

async function newBook(brief = null) {
  await mutate(async () => {
    const options = { method: 'POST' };
    if (brief !== null) { options.headers = { 'Content-Type': 'application/json' }; options.body = JSON.stringify({ brief }); }
    const job = await (await request('/jobs', options)).json();
    $('book-form').reset();
    $('custom-book').open = false;
    await selectJob(job.id);
  }, 'Book started. Research and checks will run before your first chapter review.');
}

async function download(name, button) {
  if (!state.selected || button.disabled) return;
  const id = state.selected;
  const token = state.token;
  button.disabled = true;
  try {
    const blob = await (await request(`/jobs/${id}/artifacts/${name}`)).blob();
    if (token !== state.token) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a'); link.href = url; link.download = `book-${id.slice(0, 6)}-${name}`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    message('notice', `Downloaded ${name}.`);
  } catch (error) { if (state.token) message('workspace-error', error.message); }
  finally { button.disabled = false; }
}

$('login-mode').addEventListener('click', () => accountMode(false));
$('register-mode').addEventListener('click', () => accountMode(true));
$('auth-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = $('auth-submit'); button.disabled = true;
  message('auth-error', '');
  for (const id of ['username', 'password', 'invite']) $(id).removeAttribute('aria-invalid');
  try {
    const credentials = { username: $('username').value.trim(), password: $('password').value };
    if (state.register) {
      await request('/auth/register', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Registration-Key': $('invite').value }, body: JSON.stringify(credentials) }, false);
    }
    const result = await (await request('/auth/token', { method: 'POST', body: new URLSearchParams(credentials) }, false)).json();
    state.token = result.access_token; state.generation += 1;
    sessionStorage.setItem('book-token', state.token);
    await enterWorkspace();
  } catch (error) {
    message('auth-error', error.message);
    $('password').setAttribute('aria-invalid', 'true');
  } finally { button.disabled = false; }
});
$('logout').addEventListener('click', () => signOut());
$('refresh').addEventListener('click', () => { refresh(); serviceHealth(); });
$('new-book').addEventListener('click', () => newBook());
$('empty-new-book').addEventListener('click', () => newBook());
$('book-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const brief = $('book-brief').value.trim();
  if (!brief) { $('book-brief').setCustomValidity('Describe the book you want to write.'); $('book-brief').reportValidity(); return; }
  newBook(brief);
});
$('book-brief').addEventListener('input', () => $('book-brief').setCustomValidity(''));
$('pause-resume').addEventListener('click', () => {
  const id = state.selected;
  const resume = state.job.status === 'paused';
  mutate(() => request(`/jobs/${id}/${resume ? 'resume' : 'pause'}`, { method: 'POST' }), resume ? 'Book resumed from its saved progress.' : 'Pause requested. The current step will finish first.');
});
$('approve').addEventListener('click', () => {
  const id = state.selected;
  mutate(async () => {
    await request(`/jobs/${id}/review`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ approved: true }) });
    state.feedback.delete(id); $('feedback').value = '';
  }, 'Chapter approved. The book will continue.');
});
$('feedback').addEventListener('input', () => { state.feedback.set(state.selected, $('feedback').value); $('feedback').removeAttribute('aria-invalid'); message('feedback-error', ''); });
$('feedback-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const id = state.selected; const feedback = $('feedback').value.trim();
  if (!feedback) { $('feedback').setAttribute('aria-invalid', 'true'); message('feedback-error', 'Describe what you want changed before requesting a revision.'); $('feedback').focus(); return; }
  mutate(async () => {
    await request(`/jobs/${id}/review`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ approved: false, feedback }) });
    state.feedback.delete(id); $('feedback').value = '';
  }, 'Changes requested. The writer will revise the chapter and run its checks again.');
});
$('chapter-download').addEventListener('click', () => download(`chapter-${state.view}.md`, $('chapter-download')));
document.querySelectorAll('[data-download]').forEach((button) => button.addEventListener('click', () => download(button.dataset.download, button)));
document.addEventListener('visibilitychange', () => { if (!document.hidden) { refresh(); serviceHealth(); } });
setInterval(() => { if (!document.hidden && state.user && !state.busy) refresh(); }, 4000);
accountMode(false);
serviceHealth();
if (state.token) enterWorkspace().catch((error) => { if (state.token) { signOut(); message('auth-error', error.message); } });
