/* Switchboard Admin SPA - hash-based router, vanilla JS */

'use strict';

// ── Token management ─────────────────────────────────────────────────────────

const TOKEN_KEY = 'switchboard_admin_token';

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || '';
}

function setToken(t) {
  localStorage.setItem(TOKEN_KEY, t);
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

// ── API helpers ──────────────────────────────────────────────────────────────

async function apiFetch(path, options) {
  options = options || {};
  const token = getToken();
  const headers = Object.assign({ 'Content-Type': 'application/json' }, options.headers || {});
  if (token) {
    headers['Authorization'] = 'Bearer ' + token;
  }
  const resp = await fetch(path, Object.assign({}, options, { headers }));
  if (resp.status === 401) {
    clearToken();
    showLogin();
    throw new Error('Unauthorized — please log in again.');
  }
  return resp;
}

async function apiGet(path) {
  const resp = await apiFetch(path);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error('GET ' + path + ' failed (' + resp.status + '): ' + text);
  }
  return resp.json();
}

async function apiPost(path, body) {
  const resp = await apiFetch(path, {
    method: 'POST',
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error('POST ' + path + ' failed (' + resp.status + '): ' + text);
  }
  return resp.json();
}

async function apiPut(path, body) {
  const resp = await apiFetch(path, {
    method: 'PUT',
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error('PUT ' + path + ' failed (' + resp.status + '): ' + text);
  }
  return resp.json();
}

// ── HTML helpers ─────────────────────────────────────────────────────────────

// Escape user-supplied values before insertion into HTML to prevent XSS.
function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function badge(status) {
  const safe = (status || 'disabled').toLowerCase().replace(/\s+/g, '_');
  const cls = 'badge badge-' + esc(safe);
  return '<span class="' + cls + '">' + esc(status || 'unknown') + '</span>';
}

function errorHtml(msg) {
  return '<div class="alert alert-error">' + esc(msg) + '</div>';
}

function successHtml(msg) {
  return '<div class="alert alert-success">' + esc(msg) + '</div>';
}

// Safe wrapper around innerHTML — all values must be pre-escaped via esc().
function setContent(el, html) {
  el.innerHTML = html;
}

// ── Login overlay ────────────────────────────────────────────────────────────

function showLogin() {
  let overlay = document.getElementById('login-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'login-overlay';
    document.body.appendChild(overlay);
  }
  setContent(overlay,
    '<div class="login-box">' +
    '<h2>Switchboard Admin</h2>' +
    '<p>Enter your admin token to continue.</p>' +
    '<div class="form-group">' +
    '<label for="token-input">Admin Token</label>' +
    '<input type="text" id="token-input" placeholder="Bearer token..." autocomplete="off">' +
    '</div>' +
    '<button class="btn btn-primary" id="login-btn" style="width:100%">Sign In</button>' +
    '<div id="login-error" style="margin-top:12px"></div>' +
    '</div>'
  );

  overlay.style.display = 'flex';

  document.getElementById('login-btn').addEventListener('click', async function() {
    const val = (document.getElementById('token-input').value || '').trim();
    if (!val) { return; }
    setToken(val);
    try {
      await apiGet('/admin/api/v1/health');
      overlay.style.display = 'none';
      renderPage(currentPage());
    } catch (e) {
      setContent(document.getElementById('login-error'), errorHtml('Invalid token — ' + e.message));
    }
  });
}

function hideLogin() {
  const overlay = document.getElementById('login-overlay');
  if (overlay) { overlay.style.display = 'none'; }
}

// ── Router ───────────────────────────────────────────────────────────────────

const PAGES = ['dashboard', 'key-pools', 'guardrails', 'routing', 'rate-limits', 'config'];

function currentPage() {
  const hash = window.location.hash.replace('#', '') || 'dashboard';
  return PAGES.indexOf(hash) !== -1 ? hash : 'dashboard';
}

function renderPage(page) {
  document.querySelectorAll('.nav-link').forEach(function(a) {
    a.classList.toggle('active', a.dataset.page === page);
  });

  const container = document.getElementById('page-content');
  setContent(container, '<div class="loading">Loading...</div>');

  const renderers = {
    'dashboard': renderDashboard,
    'key-pools': renderKeyPools,
    'guardrails': renderGuardrails,
    'routing': renderRouting,
    'rate-limits': renderRateLimits,
    'config': renderConfig,
  };

  const fn = renderers[page] || renderDashboard;
  fn(container).catch(function(err) {
    setContent(container, errorHtml(err.message));
  });
}

// ── Dashboard page ───────────────────────────────────────────────────────────

async function renderDashboard(container) {
  let health;
  try {
    health = await apiGet('/admin/api/v1/health');
  } catch (e) {
    setContent(container, errorHtml(e.message));
    return;
  }

  const providerRows = Object.entries(health.providers || {}).map(function(entry) {
    const id = entry[0];
    const info = entry[1];
    const pct = info.total_keys > 0
      ? Math.round((info.eligible_keys / info.total_keys) * 100)
      : 0;
    const statusText = info.eligible_keys > 0 ? 'healthy' : 'degraded';
    return '<tr>' +
      '<td>' + esc(id) + '</td>' +
      '<td>' + badge(statusText) + '</td>' +
      '<td>' + esc(info.eligible_keys) + ' / ' + esc(info.total_keys) + '</td>' +
      '<td>' + esc(pct) + '%</td>' +
      '</tr>';
  }).join('');

  const noProviders = Object.keys(health.providers || {}).length === 0
    ? '<tr><td colspan="4" style="text-align:center;color:var(--text-muted)">No providers configured</td></tr>'
    : providerRows;

  setContent(container,
    '<h1>Dashboard</h1>' +
    '<div class="stats-row">' +
    '<div class="stat-card">' +
    '<div class="stat-label">System Status</div>' +
    '<div class="stat-value" style="color:var(--success)">' + esc(health.status || 'ok') + '</div>' +
    '</div>' +
    '<div class="stat-card">' +
    '<div class="stat-label">Total Requests</div>' +
    '<div class="stat-value">\u2014</div>' +
    '</div>' +
    '<div class="stat-card">' +
    '<div class="stat-label">Error Rate</div>' +
    '<div class="stat-value">\u2014</div>' +
    '</div>' +
    '<div class="stat-card">' +
    '<div class="stat-label">Est. Cost (30d)</div>' +
    '<div class="stat-value">\u2014</div>' +
    '</div>' +
    '</div>' +
    '<div class="card">' +
    '<h2>Provider Health</h2>' +
    '<table>' +
    '<thead><tr><th>Provider</th><th>Status</th><th>Keys (eligible/total)</th><th>Availability</th></tr></thead>' +
    '<tbody>' + noProviders + '</tbody>' +
    '</table>' +
    '</div>'
  );
}

// ── Key Pools page ───────────────────────────────────────────────────────────

async function renderKeyPools(container) {
  let providers;
  try {
    providers = await apiGet('/admin/api/v1/providers');
  } catch (e) {
    setContent(container, errorHtml(e.message));
    return;
  }

  if (!Array.isArray(providers) || providers.length === 0) {
    setContent(container,
      '<h1>Key Pools</h1>' +
      '<div class="alert alert-info">No providers configured.</div>'
    );
    return;
  }

  let html = '<h1>Key Pools</h1>';

  for (const provider of providers) {
    let keys = [];
    try {
      keys = await apiGet('/admin/api/v1/providers/' + encodeURIComponent(provider.id) + '/keys');
    } catch (_) {
      // Show empty table for this provider on error.
    }

    const keyRows = Array.isArray(keys) && keys.length > 0
      ? keys.map(function(k) {
          return '<tr>' +
            '<td>' + esc(k.id) + '</td>' +
            '<td>' + badge(k.status) + '</td>' +
            '<td>' + esc(k.source) + '</td>' +
            '<td>' + esc(k.weight) + '</td>' +
            '<td>' + esc(k.total_requests) + '</td>' +
            '<td>' + esc(k.errors_last_5m) + '</td>' +
            '<td>' +
              '<button class="btn btn-danger btn-small" ' +
              'data-provider="' + esc(provider.id) + '" ' +
              'data-kid="' + esc(k.id) + '" ' +
              'data-action="disable">Disable</button>' +
            '</td>' +
            '</tr>';
        }).join('')
      : '<tr><td colspan="7" style="text-align:center;color:var(--text-muted)">No keys in pool</td></tr>';

    html +=
      '<div class="card">' +
      '<h2>' + esc(provider.id) + ' <span style="color:var(--text-muted);font-weight:400;font-size:13px">(' + esc(provider.api_format) + ')</span></h2>' +
      '<p style="color:var(--text-muted);margin-bottom:12px;font-size:12px">Models: ' + esc((provider.models || []).join(', ') || '\u2014') + '</p>' +
      '<table>' +
      '<thead><tr><th>Key ID</th><th>Status</th><th>Source</th><th>Weight</th><th>Total Req</th><th>Errors (5m)</th><th>Actions</th></tr></thead>' +
      '<tbody>' + keyRows + '</tbody>' +
      '</table>' +
      '</div>';
  }

  setContent(container, html);

  // Wire up disable buttons after setting content.
  container.querySelectorAll('button[data-action="disable"]').forEach(function(btn) {
    btn.addEventListener('click', async function() {
      const providerId = btn.dataset.provider;
      const keyId = btn.dataset.kid;
      try {
        await apiPut(
          '/admin/api/v1/providers/' + encodeURIComponent(providerId) + '/keys/' + encodeURIComponent(keyId),
          { status: 'disabled' }
        );
        await renderKeyPools(container);
      } catch (e) {
        container.insertAdjacentHTML('afterbegin', errorHtml(e.message));
      }
    });
  });
}

// ── Guardrails page ──────────────────────────────────────────────────────────

async function renderGuardrails(container) {
  let guardrails;
  try {
    guardrails = await apiGet('/admin/api/v1/guardrails');
  } catch (e) {
    setContent(container, errorHtml(e.message));
    return;
  }

  const engines = guardrails.engines || [];
  const engineRows = engines.length > 0
    ? engines.map(function(eng, i) {
        const ruleCount = (eng.rules || []).length + (eng.keywords || []).length;
        const statusLabel = eng.enabled !== false ? 'healthy' : 'disabled';
        return '<tr>' +
          '<td>' + esc(i) + '</td>' +
          '<td>' + esc(eng.engine_type || eng.type || '\u2014') + '</td>' +
          '<td>' + badge(statusLabel) + '</td>' +
          '<td>' + esc(ruleCount) + '</td>' +
          '<td>' + esc(eng.action || '\u2014') + '</td>' +
          '</tr>';
      }).join('')
    : '<tr><td colspan="5" style="text-align:center;color:var(--text-muted)">No engines configured</td></tr>';

  setContent(container,
    '<h1>Guardrails</h1>' +
    '<div class="card">' +
    '<h2>Configuration</h2>' +
    '<p style="margin-bottom:12px">Guardrails enabled: ' +
    '<strong>' + esc(guardrails.enabled ? 'Yes' : 'No') + '</strong></p>' +
    '<table>' +
    '<thead><tr><th>#</th><th>Engine</th><th>Status</th><th>Rules</th><th>Default Action</th></tr></thead>' +
    '<tbody>' + engineRows + '</tbody>' +
    '</table>' +
    '</div>' +
    '<div class="card">' +
    '<h2>Test Input</h2>' +
    '<div class="form-group">' +
    '<label for="guardrail-input">Input text to evaluate</label>' +
    '<textarea id="guardrail-input" placeholder="Enter text to test against guardrail rules..."></textarea>' +
    '</div>' +
    '<button class="btn btn-primary" id="guardrail-test-btn">Test</button>' +
    '<div id="guardrail-verdict" class="verdict-box"></div>' +
    '</div>'
  );

  document.getElementById('guardrail-test-btn').addEventListener('click', async function() {
    const input = document.getElementById('guardrail-input').value;
    const verdictBox = document.getElementById('guardrail-verdict');
    setContent(verdictBox, '<div class="loading">Evaluating...</div>');
    try {
      const result = await apiPost('/admin/api/v1/guardrails/test', { input: input });
      const cls = result.matched ? 'alert-error' : 'alert-success';
      const icon = result.matched ? 'BLOCKED' : 'ALLOWED';
      setContent(verdictBox,
        '<div class="alert ' + cls + '">' +
        '<strong>' + icon + '</strong>' +
        (result.engine ? ' &mdash; engine: <code>' + esc(result.engine) + '</code>' : '') +
        (result.action ? ', action: <code>' + esc(result.action) + '</code>' : '') +
        '<br><span style="font-size:12px;opacity:.8">' + esc(result.details) + '</span>' +
        '</div>'
      );
    } catch (e) {
      setContent(verdictBox, errorHtml(e.message));
    }
  });
}

// ── Routing page ─────────────────────────────────────────────────────────────

async function renderRouting(container) {
  let routing;
  try {
    routing = await apiGet('/admin/api/v1/routing/semantic');
  } catch (e) {
    setContent(container, errorHtml(e.message));
    return;
  }

  const rules = routing.rules || [];
  const ruleRows = rules.length > 0
    ? rules.map(function(r) {
        return '<tr>' +
          '<td>' + esc(r.name || '\u2014') + '</td>' +
          '<td>' + esc(r.pattern || r.classifier || '\u2014') + '</td>' +
          '<td>' + esc(r.provider || '\u2014') + '</td>' +
          '<td>' + esc(r.model || '\u2014') + '</td>' +
          '<td>' + esc(r.priority !== undefined ? r.priority : '\u2014') + '</td>' +
          '</tr>';
      }).join('')
    : '<tr><td colspan="5" style="text-align:center;color:var(--text-muted)">No routing rules configured</td></tr>';

  setContent(container,
    '<h1>Semantic Routing</h1>' +
    '<div class="card">' +
    '<p style="margin-bottom:12px;color:var(--text-muted)">Semantic routing enabled: ' +
    '<strong style="color:var(--text)">' + esc(routing.enabled ? 'Yes' : 'No') + '</strong>' +
    (routing.default_provider
      ? ' &nbsp;|&nbsp; Default provider: <strong style="color:var(--text)">' + esc(routing.default_provider) + '</strong>'
      : '') +
    '</p>' +
    '<table>' +
    '<thead><tr><th>Name</th><th>Pattern / Classifier</th><th>Provider</th><th>Model</th><th>Priority</th></tr></thead>' +
    '<tbody>' + ruleRows + '</tbody>' +
    '</table>' +
    '</div>'
  );
}

// ── Rate Limits page ─────────────────────────────────────────────────────────

async function renderRateLimits(container) {
  let limits;
  try {
    limits = await apiGet('/admin/api/v1/rate-limits');
  } catch (e) {
    setContent(container, errorHtml(e.message));
    return;
  }

  setContent(container,
    '<h1>Rate Limits</h1>' +
    '<div class="card">' +
    '<h2>Global Limits</h2>' +
    '<form id="rate-limit-form">' +
    '<div class="form-group">' +
    '<label for="rl-enabled">Enabled</label>' +
    '<select id="rl-enabled">' +
    '<option value="true"' + (limits.enabled ? ' selected' : '') + '>Yes</option>' +
    '<option value="false"' + (!limits.enabled ? ' selected' : '') + '>No</option>' +
    '</select>' +
    '</div>' +
    '<div class="form-group">' +
    '<label for="rl-rpm">Requests per Minute (global)</label>' +
    '<input type="number" id="rl-rpm" value="' + esc(limits.requests_per_minute || '') + '" min="0">' +
    '</div>' +
    '<div class="form-group">' +
    '<label for="rl-tpm">Tokens per Minute (global)</label>' +
    '<input type="number" id="rl-tpm" value="' + esc(limits.tokens_per_minute || '') + '" min="0">' +
    '</div>' +
    '<div class="form-group">' +
    '<label for="rl-user-rpm">Requests per Minute (per user)</label>' +
    '<input type="number" id="rl-user-rpm" value="' + esc(limits.per_user_requests_per_minute || '') + '" min="0">' +
    '</div>' +
    '<button type="submit" class="btn btn-primary">Save</button>' +
    '<div id="rl-status" style="margin-top:12px"></div>' +
    '</form>' +
    '</div>'
  );

  document.getElementById('rate-limit-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const statusEl = document.getElementById('rl-status');
    const enabled = document.getElementById('rl-enabled').value === 'true';
    const rpm = parseInt(document.getElementById('rl-rpm').value, 10) || null;
    const tpm = parseInt(document.getElementById('rl-tpm').value, 10) || null;
    const userRpm = parseInt(document.getElementById('rl-user-rpm').value, 10) || null;

    const payload = Object.assign({}, limits, {
      enabled: enabled,
      requests_per_minute: rpm,
      tokens_per_minute: tpm,
      per_user_requests_per_minute: userRpm,
    });

    try {
      await apiPut('/admin/api/v1/rate-limits', payload);
      setContent(statusEl, successHtml('Rate limits updated.'));
    } catch (err) {
      setContent(statusEl, errorHtml(err.message));
    }
  });
}

// ── Config page ──────────────────────────────────────────────────────────────

async function renderConfig(container) {
  let config;
  try {
    config = await apiGet('/admin/api/v1/config');
  } catch (e) {
    setContent(container, errorHtml(e.message));
    return;
  }

  setContent(container,
    '<h1>Configuration</h1>' +
    '<div class="card">' +
    '<p class="alert alert-info" style="margin-bottom:16px">Secrets are redacted server-side before display.</p>' +
    '<button class="btn btn-secondary" id="reload-btn" style="margin-bottom:16px">Reload Config from Disk</button>' +
    '<div id="reload-status" style="margin-bottom:12px"></div>' +
    '<pre id="config-pre"></pre>' +
    '</div>'
  );

  // Set config content via textContent (no escaping needed, no HTML injection risk).
  document.getElementById('config-pre').textContent = JSON.stringify(config, null, 2);

  document.getElementById('reload-btn').addEventListener('click', async function() {
    const statusEl = document.getElementById('reload-status');
    try {
      const result = await apiPost('/admin/api/v1/config/reload');
      setContent(statusEl, successHtml('Config reloaded: ' + esc(result.status || 'ok')));
      const updated = await apiGet('/admin/api/v1/config');
      document.getElementById('config-pre').textContent = JSON.stringify(updated, null, 2);
    } catch (err) {
      setContent(statusEl, errorHtml(err.message));
    }
  });
}

// ── Bootstrap ────────────────────────────────────────────────────────────────

async function init() {
  if (!getToken()) {
    showLogin();
    return;
  }

  try {
    await apiGet('/admin/api/v1/health');
  } catch (_) {
    showLogin();
    return;
  }

  hideLogin();
  renderPage(currentPage());
}

window.addEventListener('hashchange', function() {
  if (getToken()) {
    renderPage(currentPage());
  }
});

window.addEventListener('DOMContentLoaded', init);
