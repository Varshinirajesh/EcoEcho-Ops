// Shared across every dashboard page (not the sign-in page itself).

// Demo-only client-side gate — see index.html's login form. No real
// backend auth exists in this project; this just prevents landing on any
// dashboard page directly without going through the sign-in screen first.
if (!sessionStorage.getItem("ecoecho_auth")) {
  window.location.href = "index.html";
}

// Same-origin always — nginx proxies /api/ to dashboard-api (see nginx.conf),
// so this works identically whether it's opened on localhost:3001 in dev or
// a real domain in production.
const API = "";

const NAV_ITEMS = [
  { href: "overview.html", icon: "📊", label: "Overview" },
  { href: "live.html", icon: "🎙️", label: "Live Monitoring" },
  { href: "drift.html", icon: "🌊", label: "Drift Detection" },
  { href: "registry.html", icon: "🧬", label: "Model Registry" },
  { href: "health.html", icon: "🩺", label: "System Health" },
];

function renderSidebar(activeHref) {
  const el = document.getElementById("sidebar");
  if (!el) return;
  el.innerHTML = `
    <div class="brand">
      <div class="brand-icon">🐋</div>
      <div>
        <div class="brand-title">EcoEcho-Ops</div>
        <div class="brand-sub">Autonomous Acoustic Telemetry</div>
      </div>
    </div>
    ${NAV_ITEMS.map(
      (n) => `<a class="nav-item${n.href === activeHref ? " active" : ""}" href="${n.href}">${n.icon} ${n.label}</a>`
    ).join("")}
    <div class="nav-note" id="signout-link" style="cursor:pointer;margin-top:16px;">🔓 Sign out</div>
  `;
  document.getElementById("signout-link").addEventListener("click", () => {
    sessionStorage.removeItem("ecoecho_auth");
    window.location.href = "index.html";
  });
}

const fmtPct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
const fmtTime = (t) => (t == null ? "—" : new Date(t * 1000).toLocaleTimeString());
const fmtAgo = (t) => {
  if (t == null) return "—";
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return Math.floor(s) + "s ago";
  if (s < 3600) return Math.floor(s / 60) + "m ago";
  return Math.floor(s / 3600) + "h ago";
};

async function fetchOverview() {
  const res = await fetch(`${API}/api/overview`);
  return res.json();
}

function setHealthBadge(d) {
  const badge = document.getElementById("health-badge");
  const text = document.getElementById("health-text");
  if (!badge) return;
  badge.className = "health-badge" + (d.system_healthy ? "" : " bad");
  text.textContent = d.system_healthy ? "System Healthy" : "Degraded";
  const updatedEl = document.getElementById("updated");
  if (updatedEl) updatedEl.textContent = "Updated " + fmtAgo(d.updated_at);
}

function setHealthUnreachable() {
  const badge = document.getElementById("health-badge");
  const text = document.getElementById("health-text");
  if (!badge) return;
  badge.className = "health-badge bad";
  text.textContent = "Unreachable";
}

function drawDriftChart(elId, history, threshold) {
  const el = document.getElementById(elId);
  if (!el) return;
  if (!history || history.length < 2) {
    el.innerHTML = '<div class="empty">Not enough data yet.</div>';
    return;
  }
  const w = 620, h = 220, pad = 8;
  const values = history.map((p) => p[1]);
  const maxV = Math.max(1, threshold, ...values);
  const xs = history.map((_, i) => pad + (i / (history.length - 1)) * (w - pad * 2));
  const ys = values.map((v) => h - pad - (v / maxV) * (h - pad * 2));
  const path = xs.map((x, i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(" ");
  const threshY = h - pad - (threshold / maxV) * (h - pad * 2);
  el.innerHTML = `
    <svg width="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <line x1="${pad}" y1="${threshY}" x2="${w - pad}" y2="${threshY}" stroke="var(--critical)" stroke-width="1.5" stroke-dasharray="5 4" />
      <path d="${path}" fill="none" stroke="var(--blue)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
    <div style="font-size:11px;color:var(--ink-muted);display:flex;justify-content:space-between;margin-top:4px;">
      <span>— drift score</span><span style="color:var(--critical)">- - threshold ${threshold}</span>
    </div>`;
}

function drawMiniChart(elId, history, threshold) {
  const el = document.getElementById(elId);
  if (!el) return;
  if (!history || history.length < 2) {
    el.innerHTML = '<div class="empty" style="padding:4px 0;">No data yet.</div>';
    return;
  }
  const w = 260, h = 60, pad = 4;
  const values = history.map((p) => p[1]);
  const maxV = Math.max(0.6, threshold, ...values);
  const xs = history.map((_, i) => pad + (i / (history.length - 1)) * (w - pad * 2));
  const ys = values.map((v) => h - pad - (v / maxV) * (h - pad * 2));
  const path = xs.map((x, i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(" ");
  const threshY = h - pad - (threshold / maxV) * (h - pad * 2);
  el.innerHTML = `
    <svg width="100%" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <line x1="${pad}" y1="${threshY}" x2="${w - pad}" y2="${threshY}" stroke="var(--critical)" stroke-width="1" stroke-dasharray="3 3" opacity="0.7" />
      <path d="${path}" fill="none" stroke="var(--blue)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
    </svg>`;
}

function renderLogRows(elId, events, filterFn) {
  const el = document.getElementById(elId);
  if (!el) return;
  const list = filterFn ? events.filter(filterFn) : events;
  if (!list || list.length === 0) {
    el.innerHTML = '<div class="empty">Nothing yet.</div>';
    return;
  }
  el.innerHTML = list
    .map(
      (e) => `
    <div class="log-row">
      <span class="log-time">${new Date(e.time).toLocaleTimeString()}</span>
      <span class="log-level ${e.level}">${e.level.toUpperCase()}</span>
      <span>${e.message}</span>
    </div>`
    )
    .join("");
}
