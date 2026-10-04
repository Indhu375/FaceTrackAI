"""
src/dashboard.py
----------------
Live Analytics Web Dashboard Server for FaceTrackAI — Phase 3.

Features:
- Pure Python zero-dependency HTTP server (uses standard library http.server).
- Real-time JSON API endpoints (/api/stats, /api/visitors, /api/events, /api/export).
- Static image serving for face crop snapshots (logs/entries and logs/exits).
- Ultra-modern dark-glass UI with live auto-refresh polling, interactive charts,
  and instant report downloads.
"""

import os
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional

from src.database import DatabaseManager
from src.analytics import AnalyticsEngine
from src.logger import get_logger


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FaceTrackAI — Live Intelligence & Visitor Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #080c14;
            --bg-card: rgba(18, 26, 43, 0.75);
            --bg-card-hover: rgba(26, 38, 64, 0.85);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(56, 189, 248, 0.35);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --cyan: #38bdf8;
            --emerald: #10b981;
            --rose: #f43f5e;
            --violet: #8b5cf6;
            --amber: #f59e0b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: radial-gradient(circle at top right, #1e1b4b 0%, #080c14 60%, #030712 100%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Top Navigation Header */
        header {
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(8, 12, 20, 0.8);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-subtle);
            padding: 16px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--cyan), var(--violet));
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-family: 'Outfit', sans-serif;
            color: #fff;
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
        }

        .brand-text h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #fff, #93c5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-text span {
            font-size: 11px;
            color: var(--cyan);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .live-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--emerald);
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background: var(--emerald);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        .btn-export {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border-subtle);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn-export:hover {
            background: rgba(56, 189, 248, 0.15);
            border-color: var(--cyan);
            color: var(--cyan);
            transform: translateY(-1px);
        }

        /* Main Container */
        main {
            max-width: 1380px;
            width: 100%;
            margin: 0 auto;
            padding: 32px 24px;
            flex: 1;
        }

        /* KPI Cards Grid */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }

        .kpi-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 24px;
            backdrop-filter: blur(12px);
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
        }

        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: transparent;
            transition: all 0.3s ease;
        }

        .kpi-card:hover {
            transform: translateY(-3px);
            border-color: var(--border-glow);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
        }

        .kpi-card.cyan::before { background: var(--cyan); }
        .kpi-card.emerald::before { background: var(--emerald); }
        .kpi-card.violet::before { background: var(--violet); }
        .kpi-card.amber::before { background: var(--amber); }

        .kpi-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .kpi-title {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }

        .kpi-icon {
            font-size: 18px;
        }

        .kpi-value {
            font-family: 'Outfit', sans-serif;
            font-size: 38px;
            font-weight: 700;
            color: #fff;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }

        .kpi-desc {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 8px;
        }

        /* Two-Column Layout */
        .content-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 32px;
        }

        @media (max-width: 992px) {
            .content-grid { grid-template-columns: 1fr; }
        }

        .panel {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 24px;
            backdrop-filter: blur(12px);
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .panel-header h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 18px;
            font-weight: 600;
        }

        /* Chart Canvas */
        .chart-box {
            position: relative;
            height: 220px;
            width: 100%;
        }

        canvas {
            width: 100% !important;
            height: 100% !important;
        }

        /* Custom Modern Tables */
        .table-wrap {
            overflow-x: auto;
            max-height: 380px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        th {
            position: sticky;
            top: 0;
            background: #111a2e;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 11px;
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-subtle);
        }

        td {
            padding: 12px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-primary);
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.03);
        }

        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .badge-entry {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-exit {
            background: rgba(244, 63, 94, 0.15);
            color: #fb7185;
            border: 1px solid rgba(244, 63, 94, 0.3);
        }

        .thumb {
            width: 38px;
            height: 38px;
            border-radius: 8px;
            object-fit: cover;
            border: 1px solid var(--border-subtle);
            vertical-align: middle;
            transition: transform 0.2s;
            cursor: pointer;
        }

        .thumb:hover {
            transform: scale(2.2);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
            z-index: 50;
            position: relative;
        }

        /* Footer */
        footer {
            border-top: 1px solid var(--border-subtle);
            padding: 20px 32px;
            text-align: center;
            font-size: 13px;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="logo-icon">FT</div>
            <div class="brand-text">
                <h1>FaceTrackAI</h1>
                <span>Intelligence Suite &bull; Phase 3</span>
            </div>
        </div>
        <div class="header-controls">
            <div class="live-indicator">
                <div class="pulse-dot"></div>
                <span>LIVE SYNC</span>
            </div>
            <a href="/api/export/csv-visitors" class="btn-export" download="visitors_summary.csv">
                <span>&darr;</span> Visitors CSV
            </a>
            <a href="/api/export/csv-events" class="btn-export" download="events_log.csv">
                <span>&darr;</span> Events CSV
            </a>
            <a href="/api/export/json" class="btn-export" download="analytics_report.json">
                <span>&darr;</span> Report JSON
            </a>
        </div>
    </header>

    <main>
        <!-- KPI Metrics Row -->
        <div class="kpi-row">
            <div class="kpi-card cyan">
                <div class="kpi-header">
                    <div class="kpi-title">Total Unique Visitors</div>
                    <div class="kpi-icon">&bull;</div>
                </div>
                <div class="kpi-value" id="kpi-unique">--</div>
                <div class="kpi-desc">Persistent ArcFace identities</div>
            </div>

            <div class="kpi-card emerald">
                <div class="kpi-header">
                    <div class="kpi-title">Avg Dwell Time</div>
                    <div class="kpi-icon">&bull;</div>
                </div>
                <div class="kpi-value" id="kpi-dwell">--</div>
                <div class="kpi-desc" id="kpi-dwell-sub">Duration between Entry & Exit</div>
            </div>

            <div class="kpi-card violet">
                <div class="kpi-header">
                    <div class="kpi-title">Total Activity Events</div>
                    <div class="kpi-icon">&bull;</div>
                </div>
                <div class="kpi-value" id="kpi-events">--</div>
                <div class="kpi-desc" id="kpi-events-sub">Entries & Exits recorded</div>
            </div>

            <div class="kpi-card amber">
                <div class="kpi-header">
                    <div class="kpi-title">Returning Visitor Rate</div>
                    <div class="kpi-icon">&bull;</div>
                </div>
                <div class="kpi-value" id="kpi-retention">--</div>
                <div class="kpi-desc" id="kpi-retention-sub">Multi-session retention</div>
            </div>
        </div>

        <!-- Charts and Activity Grid -->
        <div class="content-grid">
            <!-- Hourly Traffic Chart Panel -->
            <div class="panel">
                <div class="panel-header">
                    <h2>Traffic Activity Timeline</h2>
                    <span style="font-size:12px; color:var(--text-muted);" id="traffic-updated">Live</span>
                </div>
                <div class="chart-box">
                    <canvas id="trafficChart"></canvas>
                </div>
            </div>

            <!-- Registered Visitors Registry -->
            <div class="panel">
                <div class="panel-header">
                    <h2>Registered Visitor Directory</h2>
                    <span style="font-size:12px; color:var(--cyan);" id="visitors-count">0 visitors</span>
                </div>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>Face ID</th>
                                <th>First Seen</th>
                                <th>Last Seen</th>
                                <th>Visits</th>
                            </tr>
                        </thead>
                        <tbody id="visitors-tbody">
                            <tr><td colspan="4" style="text-align:center; color:var(--text-muted);">Loading registry...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Live Events Activity Log -->
        <div class="panel">
            <div class="panel-header">
                <h2>Real-Time Entry & Exit Event Stream</h2>
                <span style="font-size:12px; color:var(--emerald);" id="events-count">0 events</span>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Visitor</th>
                            <th>Event</th>
                            <th>Timestamp</th>
                            <th>Face Snapshot</th>
                        </tr>
                    </thead>
                    <tbody id="events-tbody">
                        <tr><td colspan="4" style="text-align:center; color:var(--text-muted);">Awaiting events...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </main>

    <footer>
        FaceTrackAI &bull; Phase 3 Analytics &amp; Real-Time Tracking Engine &bull; Automated SQLite Sync
    </footer>

    <script>
        // Simple Canvas Bar Chart Renderer (Zero External JS Dependencies)
        function renderTrafficChart(canvasId, dataPoints) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;

            // Set display size vs internal size
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);

            const w = rect.width;
            const h = rect.height;
            const padding = { top: 20, right: 15, bottom: 35, left: 35 };
            const chartW = w - padding.left - padding.right;
            const chartH = h - padding.top - padding.bottom;

            ctx.clearRect(0, 0, w, h);

            const keys = Object.keys(dataPoints);
            const values = Object.values(dataPoints);
            const maxVal = Math.max(1, ...values);

            // Draw grid lines
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = padding.top + (chartH / 4) * i;
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(w - padding.right, y);
                ctx.stroke();

                const gridVal = Math.round(maxVal - (maxVal / 4) * i);
                ctx.fillStyle = '#64748b';
                ctx.font = '10px Inter';
                ctx.textAlign = 'right';
                ctx.fillText(gridVal, padding.left - 8, y + 3);
            }

            // Draw bars
            const barW = Math.max(8, (chartW / keys.length) * 0.65);
            const gap = chartW / keys.length;

            keys.forEach((k, idx) => {
                const val = values[idx];
                const barH = (val / maxVal) * chartH;
                const x = padding.left + idx * gap + (gap - barW) / 2;
                const y = padding.top + (chartH - barH);

                // Bar gradient
                const grad = ctx.createLinearGradient(0, y, 0, y + barH);
                grad.addColorStop(0, '#38bdf8');
                grad.addColorStop(1, '#818cf8');

                ctx.fillStyle = val > 0 ? grad : 'rgba(255,255,255,0.04)';
                ctx.beginPath();
                ctx.roundRect(x, val > 0 ? y : y + barH - 4, barW, val > 0 ? barH : 4, [4, 4, 0, 0]);
                ctx.fill();

                // X label every 3 hours
                if (idx % 3 === 0 || idx === keys.length - 1) {
                    ctx.fillStyle = '#94a3b8';
                    ctx.font = '10px Inter';
                    ctx.textAlign = 'center';
                    ctx.fillText(k, x + barW / 2, h - 10);
                }
            });
        }

        async function fetchDashboardData() {
            try {
                const res = await fetch('/api/stats');
                if (!res.ok) return;
                const data = await res.json();

                // Update KPIs
                const summary = data.summary || {};
                const ret = data.retention || {};
                document.getElementById('kpi-unique').textContent = summary.total_unique_visitors || 0;
                document.getElementById('kpi-dwell').textContent = summary.avg_dwell_time_formatted || '0s';
                document.getElementById('kpi-dwell-sub').textContent = `Max: ${summary.max_dwell_time_seconds || 0}s | Min: ${summary.min_dwell_time_seconds || 0}s`;
                document.getElementById('kpi-events').textContent = summary.total_events || 0;
                document.getElementById('kpi-events-sub').textContent = `${summary.total_entries || 0} In &bull; ${summary.total_exits || 0} Out`;
                document.getElementById('kpi-retention').textContent = `${ret.return_rate_percentage || 0}%`;
                document.getElementById('kpi-retention-sub').textContent = `${ret.returning_visitors || 0} returning visitor(s)`;

                // Update chart
                if (data.hourly_traffic) {
                    renderTrafficChart('trafficChart', data.hourly_traffic);
                }

                // Update Visitors
                const vRes = await fetch('/api/visitors');
                if (vRes.ok) {
                    const visitors = await vRes.json();
                    document.getElementById('visitors-count').textContent = `${visitors.length} visitors`;
                    const vTbody = document.getElementById('visitors-tbody');
                    if (visitors.length === 0) {
                        vTbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:#64748b;">No visitors yet</td></tr>';
                    } else {
                        vTbody.innerHTML = visitors.map(v => `
                            <tr>
                                <td><strong style="color:#38bdf8;">${v.face_id}</strong></td>
                                <td>${v.first_seen}</td>
                                <td>${v.last_seen}</td>
                                <td><span style="background:rgba(255,255,255,0.06); padding:2px 8px; border-radius:4px;">${v.visits || 1}</span></td>
                            </tr>
                        `).join('');
                    }
                }

                // Update Events
                const eRes = await fetch('/api/events');
                if (eRes.ok) {
                    const events = await eRes.json();
                    document.getElementById('events-count').textContent = `${events.length} events`;
                    const eTbody = document.getElementById('events-tbody');
                    if (events.length === 0) {
                        eTbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:#64748b;">No events recorded yet</td></tr>';
                    } else {
                        eTbody.innerHTML = events.slice(-50).reverse().map(e => `
                            <tr>
                                <td><strong>${e.face_id}</strong></td>
                                <td><span class="badge ${e.event_type === 'ENTRY' ? 'badge-entry' : 'badge-exit'}">${e.event_type}</span></td>
                                <td>${e.timestamp}</td>
                                <td>${e.image_url ? `<img src="${e.image_url}" class="thumb" alt="Face"/>` : '<span style="color:#64748b;">-</span>'}</td>
                            </tr>
                        `).join('');
                    }
                }
            } catch (err) {
                console.error('Fetch error:', err);
            }
        }

        // Initial fetch + 3-second auto-polling
        fetchDashboardData();
        setInterval(fetchDashboardData, 3000);
        window.addEventListener('resize', () => fetchDashboardData());
    </script>
</body>
</html>
"""


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving dashboard pages, API endpoints, and images."""

    db: Optional[DatabaseManager] = None
    analytics: Optional[AnalyticsEngine] = None

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stdout logging for clean console."""
        pass

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            self._respond_html(DASHBOARD_HTML)
        elif path == "/api/stats":
            self._handle_api_stats()
        elif path == "/api/visitors":
            self._handle_api_visitors()
        elif path == "/api/events":
            self._handle_api_events()
        elif path.startswith("/api/export/"):
            self._handle_api_export(path)
        elif path.startswith("/images/"):
            self._handle_serve_image(path)
        else:
            self.send_error(404, "Endpoint not found")

    def _respond_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _respond_json(self, data: Any) -> None:
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_api_stats(self) -> None:
        if self.analytics:
            metrics = self.analytics.compute_metrics()
            self._respond_json(metrics)
        else:
            self._respond_json({})

    def _handle_api_visitors(self) -> None:
        if not self.db:
            self._respond_json([])
            return

        visitors = self.db.get_all_visitors()
        events = self.db.get_all_events()
        entries_count: Dict[str, int] = {}
        for ev in events:
            if ev.get("event_type") == "ENTRY":
                fid = ev["face_id"]
                entries_count[fid] = entries_count.get(fid, 0) + 1

        res = [
            {
                "face_id": v["face_id"],
                "first_seen": v.get("first_seen", ""),
                "last_seen": v.get("last_seen", ""),
                "visits": entries_count.get(v["face_id"], 1),
            }
            for v in visitors
        ]
        self._respond_json(res)

    def _handle_api_events(self) -> None:
        if not self.db:
            self._respond_json([])
            return

        events = self.db.get_all_events()
        res = []
        for ev in events:
            img_path = ev.get("image_path")
            img_url = None
            if img_path and os.path.exists(img_path):
                # normalize path to URL format: /images/<filename>
                fname = os.path.basename(img_path)
                parent = os.path.basename(os.path.dirname(img_path))
                img_url = f"/images/{parent}/{fname}"

            res.append(
                {
                    "face_id": ev.get("face_id"),
                    "event_type": ev.get("event_type"),
                    "timestamp": ev.get("timestamp"),
                    "image_url": img_url,
                }
            )
        self._respond_json(res)

    def _handle_api_export(self, path: str) -> None:
        if not self.analytics:
            self.send_error(500, "Analytics engine not configured")
            return

        fmt = path.replace("/api/export/", "")
        if fmt == "csv-visitors":
            fp = self.analytics.export_visitors_csv("visitors_export.csv")
            self._send_file_download(fp, "text/csv", "visitors_summary.csv")
        elif fmt == "csv-events":
            fp = self.analytics.export_events_csv("events_export.csv")
            self._send_file_download(fp, "text/csv", "events_log.csv")
        elif fmt == "json":
            fp = self.analytics.export_json("analytics_export.json")
            self._send_file_download(fp, "application/json", "analytics_report.json")
        else:
            self.send_error(400, "Unknown export format")

    def _send_file_download(self, filepath: str, mime: str, dl_name: str) -> None:
        if not os.path.exists(filepath):
            self.send_error(404, "Export file not found")
            return

        with open(filepath, "rb") as f:
            data = f.read()

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Disposition", f'attachment; filename="{dl_name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_serve_image(self, path: str) -> None:
        # e.g. /images/entries/visitor_001.jpg -> logs/entries/visitor_001.jpg
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "images":
            subfolder = parts[1]  # 'entries' or 'exits'
            fname = parts[2]
            real_path = os.path.join("logs", subfolder, fname)
            if os.path.exists(real_path):
                with open(real_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_error(404, "Image not found")


class DashboardServer:
    """
    Manages the lifecycle of the Analytics Dashboard Web Server.
    """

    def __init__(
        self,
        db: DatabaseManager,
        port: int = 8000,
        host: str = "127.0.0.1",
    ) -> None:
        self.db = db
        self.port = port
        self.host = host
        self.analytics = AnalyticsEngine(db)
        self.logger = get_logger()
        self._server: Optional[HTTPServer] = None

    def start(self) -> None:
        """Start the HTTP server and serve indefinitely."""
        DashboardRequestHandler.db = self.db
        DashboardRequestHandler.analytics = self.analytics

        self._server = HTTPServer((self.host, self.port), DashboardRequestHandler)
        url = f"http://{self.host}:{self.port}"
        self.logger.info(f"DASHBOARD | Serving Live Analytics Dashboard at {url}")
        print(f"\n=======================================================")
        print(f"  FaceTrackAI Live Analytics Dashboard is running!")
        print(f"  Open in your browser: {url}")
        print(f"  Press Ctrl+C to stop.")
        print(f"=======================================================\n")

        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            self.logger.info("DASHBOARD | Stopped by user")
        finally:
            self.stop()

    def stop(self) -> None:
        """Close HTTP server cleanly."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self.logger.info("DASHBOARD | Server stopped")
