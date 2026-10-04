"""
tests/test_analytics.py
-----------------------
Unit tests for the AnalyticsEngine and reporting systems (Phase 3).
"""

import os
import json
import shutil
import tempfile
import pytest
import numpy as np

from src.database import DatabaseManager
from src.analytics import AnalyticsEngine


class TestAnalyticsEngine:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "analytics_test.db")
        self.reports_dir = os.path.join(self.tmpdir, "reports")

        self.db = DatabaseManager(self.db_path)
        self.engine = AnalyticsEngine(self.db, reports_dir=self.reports_dir)

    def teardown_method(self):
        self.db.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_compute_metrics_empty_db(self):
        metrics = self.engine.compute_metrics()
        assert metrics["summary"]["total_unique_visitors"] == 0
        assert metrics["summary"]["total_events"] == 0
        assert metrics["summary"]["avg_dwell_time_seconds"] == 0.0

    def test_calculate_dwell_time(self):
        # Visitor 1: Entry at 10:00:00, Exit at 10:02:30 -> 150 seconds (2m 30s)
        self.db.insert_visitor("visitor_001", np.zeros(512), first_seen="2026-10-04 10:00:00")
        self.db.insert_event("visitor_001", "ENTRY", timestamp="2026-10-04 10:00:00")
        self.db.insert_event("visitor_001", "EXIT", timestamp="2026-10-04 10:02:30")

        metrics = self.engine.compute_metrics()
        summary = metrics["summary"]
        assert summary["total_unique_visitors"] == 1
        assert summary["total_events"] == 2
        assert summary["avg_dwell_time_seconds"] == 150.0
        assert summary["avg_dwell_time_formatted"] == "2m 30s"

    def test_retention_and_frequency(self):
        # Visitor 1: 1 visit
        self.db.insert_visitor("visitor_001", np.zeros(512))
        self.db.insert_event("visitor_001", "ENTRY", timestamp="2026-10-04 10:00:00")

        # Visitor 2: 2 visits (returning)
        self.db.insert_visitor("visitor_002", np.zeros(512))
        self.db.insert_event("visitor_002", "ENTRY", timestamp="2026-10-04 11:00:00")
        self.db.insert_event("visitor_002", "EXIT", timestamp="2026-10-04 11:05:00")
        self.db.insert_event("visitor_002", "ENTRY", timestamp="2026-10-04 14:00:00")

        metrics = self.engine.compute_metrics()
        ret = metrics["retention"]
        assert ret["single_visit_visitors"] == 1
        assert ret["returning_visitors"] == 1
        assert ret["return_rate_percentage"] == 50.0

    def test_hourly_traffic_distribution(self):
        self.db.insert_visitor("visitor_001", np.zeros(512))
        self.db.insert_event("visitor_001", "ENTRY", timestamp="2026-10-04 09:15:00")
        self.db.insert_event("visitor_001", "EXIT", timestamp="2026-10-04 09:45:00")
        self.db.insert_event("visitor_001", "ENTRY", timestamp="2026-10-04 15:30:00")

        metrics = self.engine.compute_metrics()
        traffic = metrics["hourly_traffic"]
        assert traffic["09:00"] == 2
        assert traffic["15:00"] == 1
        assert traffic["12:00"] == 0

    def test_export_all_reports(self):
        self.db.insert_visitor("visitor_001", np.zeros(512))
        self.db.insert_event("visitor_001", "ENTRY", timestamp="2026-10-04 10:00:00")
        self.db.insert_event("visitor_001", "EXIT", timestamp="2026-10-04 10:03:00")

        reports = self.engine.export_all(prefix="test")
        assert os.path.exists(reports["visitors_csv"])
        assert os.path.exists(reports["events_csv"])
        assert os.path.exists(reports["json_report"])
        assert os.path.exists(reports["html_report"])

        # Check JSON report content
        with open(reports["json_report"], "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "summary" in data
            assert data["summary"]["total_unique_visitors"] == 1
            assert len(data["events"]) == 2
