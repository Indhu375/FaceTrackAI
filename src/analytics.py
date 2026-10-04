"""
src/analytics.py
----------------
Analytics and Reporting Engine for FaceTrackAI — Phase 3.

Provides rich business intelligence and foot-traffic analytics:
1. Total & Distinct Visitor metrics.
2. Dwell Time calculation (duration between ENTRY and EXIT pairs per visitor).
3. Hourly Traffic & Peak Occupancy analysis.
4. Returning vs One-time Visitor retention rates.
5. Export engines:
   - CSV export (visitors and events)
   - JSON structured analytics report
   - Self-contained interactive HTML executive report with CSS visual styling
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.database import DatabaseManager
from src.logger import get_logger


class AnalyticsEngine:
    """
    Computes analytics and exports reports from visitor tracking databases.

    Parameters
    ----------
    db          : DatabaseManager Data source.
    reports_dir : str             Directory to save exported reports.
    """

    def __init__(
        self,
        db: DatabaseManager,
        reports_dir: str = "outputs/reports",
    ) -> None:
        self.db = db
        self.reports_dir = reports_dir
        self.logger = get_logger()

        os.makedirs(self.reports_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Core Analytics Computations
    # ------------------------------------------------------------------

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute high-level visitor metrics, dwell times, and hourly traffic.

        Returns
        -------
        Dictionary containing all computed KPIs and statistical summaries.
        """
        visitors = self.db.get_all_visitors()
        events = self.db.get_all_events()

        total_unique = len(visitors)
        total_events = len(events)
        entry_events = [e for e in events if e.get("event_type") == "ENTRY"]
        exit_events = [e for e in events if e.get("event_type") == "EXIT"]

        # Compute dwell times (ENTRY -> EXIT sessions per visitor)
        dwell_times_sec, visitor_dwells = self._calculate_dwell_times(events)

        avg_dwell_sec = float(np.mean(dwell_times_sec)) if dwell_times_sec else 0.0
        max_dwell_sec = float(np.max(dwell_times_sec)) if dwell_times_sec else 0.0
        min_dwell_sec = float(np.min(dwell_times_sec)) if dwell_times_sec else 0.0

        # Hourly traffic distribution
        hourly_traffic = self._calculate_hourly_traffic(events)

        # Retention / frequency
        retention = self._calculate_retention(events)

        return {
            "summary": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_unique_visitors": total_unique,
                "total_events": total_events,
                "total_entries": len(entry_events),
                "total_exits": len(exit_events),
                "avg_dwell_time_seconds": round(avg_dwell_sec, 1),
                "min_dwell_time_seconds": round(min_dwell_sec, 1),
                "max_dwell_time_seconds": round(max_dwell_sec, 1),
                "avg_dwell_time_formatted": self._format_duration(avg_dwell_sec),
            },
            "hourly_traffic": hourly_traffic,
            "retention": retention,
            "visitor_dwells": visitor_dwells,
        }

    def _calculate_dwell_times(
        self,
        events: List[Dict[str, Any]],
    ) -> Tuple[List[float], Dict[str, List[Dict[str, Any]]]]:
        """Match ENTRY and EXIT events sequentially to compute dwell times."""
        visitor_events: Dict[str, List[Dict[str, Any]]] = {}
        for ev in events:
            fid = ev["face_id"]
            if fid not in visitor_events:
                visitor_events[fid] = []
            visitor_events[fid].append(ev)

        all_dwell_seconds: List[float] = []
        visitor_dwells: Dict[str, List[Dict[str, Any]]] = {}

        for fid, ev_list in visitor_events.items():
            visitor_dwells[fid] = []
            pending_entry = None

            for ev in ev_list:
                ev_type = ev.get("event_type")
                ts_str = ev.get("timestamp")

                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue

                if ev_type == "ENTRY":
                    pending_entry = (ts, ev)
                elif ev_type == "EXIT" and pending_entry is not None:
                    entry_ts, entry_ev = pending_entry
                    duration = max(0.0, (ts - entry_ts).total_seconds())
                    all_dwell_seconds.append(duration)

                    visitor_dwells[fid].append(
                        {
                            "entry_time": entry_ev["timestamp"],
                            "exit_time": ev["timestamp"],
                            "dwell_seconds": round(duration, 1),
                            "dwell_formatted": self._format_duration(duration),
                            "entry_image": entry_ev.get("image_path"),
                            "exit_image": ev.get("image_path"),
                        }
                    )
                    pending_entry = None

        return all_dwell_seconds, visitor_dwells

    def _calculate_hourly_traffic(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate traffic counts by hour of day (00 to 23)."""
        counts = {f"{h:02d}:00": 0 for h in range(24)}
        for ev in events:
            ts_str = ev.get("timestamp", "")
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                hour_key = f"{dt.hour:02d}:00"
                counts[hour_key] += 1
            except Exception:
                continue
        # Filter to only hours with activity or return full dictionary
        return counts

    def _calculate_retention(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify visitors into single-visit vs multi-visit."""
        visit_counts: Dict[str, int] = {}
        for ev in events:
            if ev.get("event_type") == "ENTRY":
                fid = ev["face_id"]
                visit_counts[fid] = visit_counts.get(fid, 0) + 1

        single = sum(1 for c in visit_counts.values() if c == 1)
        multiple = sum(1 for c in visit_counts.values() if c > 1)
        total = max(1, len(visit_counts))

        return {
            "single_visit_visitors": single,
            "returning_visitors": multiple,
            "return_rate_percentage": round((multiple / total) * 100.0, 1),
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into readable Xm Ys format."""
        mins = int(seconds // 60)
        secs = int(round(seconds % 60))
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    # ------------------------------------------------------------------
    # Report Exporters
    # ------------------------------------------------------------------

    def export_all(self, prefix: Optional[str] = None) -> Dict[str, str]:
        """Generate and export CSV, JSON, and HTML reports."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"{prefix}_{ts}" if prefix else ts

        csv_visitors = self.export_visitors_csv(f"visitors_summary_{tag}.csv")
        csv_events = self.export_events_csv(f"events_log_{tag}.csv")
        json_report = self.export_json(f"analytics_report_{tag}.json")
        html_report = self.export_html(f"visitor_report_{tag}.html")

        self.logger.info(f"ANALYTICS | Reports exported successfully to {self.reports_dir}")
        return {
            "visitors_csv": csv_visitors,
            "events_csv": csv_events,
            "json_report": json_report,
            "html_report": html_report,
        }

    def export_visitors_csv(self, filename: str = "visitors_summary.csv") -> str:
        """Export all registered visitors with timestamps and visit metrics to CSV."""
        filepath = os.path.join(self.reports_dir, filename)
        visitors = self.db.get_all_visitors()
        events = self.db.get_all_events()

        # Count total entries per visitor
        entries_count: Dict[str, int] = {}
        for ev in events:
            if ev.get("event_type") == "ENTRY":
                fid = ev["face_id"]
                entries_count[fid] = entries_count.get(fid, 0) + 1

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Face ID", "First Seen", "Last Seen", "Total Visits"])
            for v in visitors:
                fid = v["face_id"]
                writer.writerow([
                    fid,
                    v.get("first_seen", ""),
                    v.get("last_seen", ""),
                    entries_count.get(fid, 1),
                ])
        return filepath

    def export_events_csv(self, filename: str = "events_log.csv") -> str:
        """Export all raw ENTRY and EXIT events to CSV."""
        filepath = os.path.join(self.reports_dir, filename)
        events = self.db.get_all_events()

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Event ID", "Face ID", "Event Type", "Timestamp", "Image Snapshot"])
            for i, ev in enumerate(events, 1):
                writer.writerow([
                    i,
                    ev.get("face_id", ""),
                    ev.get("event_type", ""),
                    ev.get("timestamp", ""),
                    ev.get("image_path", ""),
                ])
        return filepath

    def export_json(self, filename: str = "analytics_report.json") -> str:
        """Export full structured metrics and event breakdowns to JSON."""
        filepath = os.path.join(self.reports_dir, filename)
        data = self.compute_metrics()
        data["events"] = [
            {
                "face_id": e.get("face_id"),
                "event_type": e.get("event_type"),
                "timestamp": e.get("timestamp"),
                "image_path": e.get("image_path"),
            }
            for e in self.db.get_all_events()
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath

    def export_html(self, filename: str = "visitor_report.html") -> str:
        """Generate a sleek, standalone HTML report with KPIs and visual charts."""
        filepath = os.path.join(self.reports_dir, filename)
        metrics = self.compute_metrics()
        summary = metrics["summary"]
        retention = metrics["retention"]
        events = self.db.get_all_events()
        recent_events = events[-50:]  # last 50 events for clean display

        # Build table rows for recent events
        event_rows = ""
        for ev in reversed(recent_events):
            badge_class = "badge-entry" if ev["event_type"] == "ENTRY" else "badge-exit"
            img_html = ""
            if ev.get("image_path") and os.path.exists(ev["image_path"]):
                # Use relative path so browser loads image
                rel_path = os.path.relpath(ev["image_path"], os.path.dirname(filepath)).replace("\\", "/")
                img_html = f'<img src="{rel_path}" class="snapshot-thumb" alt="Face"/>'
            else:
                img_html = '<span class="no-img">-</span>'

            event_rows += f"""
            <tr>
                <td><strong>{ev.get('face_id')}</strong></td>
                <td><span class="badge {badge_class}">{ev.get('event_type')}</span></td>
                <td>{ev.get('timestamp')}</td>
                <td>{img_html}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FaceTrackAI — Visitor Analytics Executive Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #34d399;
            --accent-red: #f87171;
            --accent-purple: #c084fc;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #1e1e38 100%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 40px 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 32px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--card-border);
        }}
        .header h1 {{ font-size: 28px; font-weight: 700; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .header .meta {{ color: var(--text-muted); font-size: 14px; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 36px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        }}
        .card .title {{ font-size: 13px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 8px; }}
        .card .value {{ font-size: 34px; font-weight: 700; color: #fff; }}
        .card .sub {{ font-size: 13px; color: var(--text-muted); margin-top: 6px; }}
        .highlight-green {{ color: var(--accent-green) !important; }}
        .highlight-blue {{ color: var(--accent-blue) !important; }}
        .highlight-purple {{ color: var(--accent-purple) !important; }}
        .section-title {{ font-size: 20px; font-weight: 600; margin-bottom: 18px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 14px 16px;
            text-align: left;
            border-bottom: 1px solid var(--card-border);
        }}
        th {{ color: var(--text-muted); font-weight: 500; text-transform: uppercase; font-size: 12px; }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-entry {{ background: rgba(52, 211, 153, 0.15); color: var(--accent-green); border: 1px solid rgba(52, 211, 153, 0.3); }}
        .badge-exit {{ background: rgba(248, 113, 113, 0.15); color: var(--accent-red); border: 1px solid rgba(248, 113, 113, 0.3); }}
        .snapshot-thumb {{ width: 44px; height: 44px; object-fit: cover; border-radius: 8px; border: 1px solid var(--card-border); vertical-align: middle; }}
        .no-img {{ color: var(--text-muted); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>FaceTrackAI — Analytics Report</h1>
                <div class="meta">Automated Business Intelligence & Foot Traffic Report</div>
            </div>
            <div class="meta">Generated: {summary['generated_at']}</div>
        </div>

        <div class="kpi-grid">
            <div class="card">
                <div class="title">Unique Visitors</div>
                <div class="value highlight-blue">{summary['total_unique_visitors']}</div>
                <div class="sub">Distinct identities tracked</div>
            </div>
            <div class="card">
                <div class="title">Average Dwell Time</div>
                <div class="value highlight-green">{summary['avg_dwell_time_formatted']}</div>
                <div class="sub">Max dwell: {self._format_duration(summary['max_dwell_time_seconds'])}</div>
            </div>
            <div class="card">
                <div class="title">Total Activity Events</div>
                <div class="value highlight-purple">{summary['total_events']}</div>
                <div class="sub">{summary['total_entries']} Entries | {summary['total_exits']} Exits</div>
            </div>
            <div class="card">
                <div class="title">Visitor Return Rate</div>
                <div class="value">{retention['return_rate_percentage']}%</div>
                <div class="sub">{retention['returning_visitors']} returning of {summary['total_unique_visitors']} total</div>
            </div>
        </div>

        <div class="card">
            <div class="section-title">Recent Event Activity Log</div>
            <table>
                <thead>
                    <tr>
                        <th>Visitor ID</th>
                        <th>Event</th>
                        <th>Timestamp</th>
                        <th>Snapshot</th>
                    </tr>
                </thead>
                <tbody>
                    {event_rows if event_rows else '<tr><td colspan="4" style="text-align:center; color:#94a3b8;">No events recorded yet.</td></tr>'}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        return filepath
