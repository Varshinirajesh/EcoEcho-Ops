# Static pages, no build step — plain HTML/CSS/JS polling dashboard-api.
# index.html is the sign-in gate (demo-only, client-side — see its own
# comment); the rest are dashboard pages behind it, sharing style.css
# and common.js (auth guard, sidebar nav, fetch/format helpers).
FROM nginx:1.27-alpine
COPY services/dashboard-frontend/index.html /usr/share/nginx/html/index.html
COPY services/dashboard-frontend/overview.html /usr/share/nginx/html/overview.html
COPY services/dashboard-frontend/live.html /usr/share/nginx/html/live.html
COPY services/dashboard-frontend/drift.html /usr/share/nginx/html/drift.html
COPY services/dashboard-frontend/registry.html /usr/share/nginx/html/registry.html
COPY services/dashboard-frontend/health.html /usr/share/nginx/html/health.html
COPY services/dashboard-frontend/style.css /usr/share/nginx/html/style.css
COPY services/dashboard-frontend/common.js /usr/share/nginx/html/common.js
COPY services/dashboard-frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
