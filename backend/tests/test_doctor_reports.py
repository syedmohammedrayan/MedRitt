"""Authorization and behaviour coverage for the doctor report index.

Uses an isolated temporary SQLite database only. No production data,
fixtures, or seed records are created.
"""

import json
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import crud
from db.database import Base, get_db
from db.models import Appointment, DiagnosticOrder, Report, Result, Scan
from routers import doctor_reports
from routers.auth import create_access_token


class DoctorReportIndexTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "doctor-reports.db"
        self.engine = create_engine(
            f"sqlite:///{database_path}", connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()

        self.doctor = crud.create_user(self.db, "doc.one", "hash", role="doctor", full_name="Doctor One")
        self.other_doctor = crud.create_user(self.db, "doc.two", "hash", role="doctor", full_name="Doctor Two")
        self.admin = crud.create_user(self.db, "admin.one", "hash", role="admin", full_name="Admin One")
        self.lab = crud.create_user(self.db, "lab.one", "hash", role="lab_tech", full_name="Lab One")
        self.alice = crud.create_user(self.db, "alice.p", "hash", role="patient", full_name="Alice Sharma")
        self.bob = crud.create_user(self.db, "bob.p", "hash", role="patient", full_name="Bob Verma")
        self.carol = crud.create_user(self.db, "carol.p", "hash", role="patient", full_name="Carol Iyer")

        self.base_time = datetime(2026, 1, 10, 9, 30, 15)
        # Alice: two reports with Doctor One (follow-up), plus a detection study.
        self.alice_first = self._report(self.alice, self.doctor, "pneumonia", self.base_time, label="PNEUMONIA", confidence=0.91)
        self.alice_second = self._report(self.alice, self.doctor, "pneumonia", self.base_time + timedelta(days=5, hours=2), label="NORMAL", confidence=0.88, approved=True)
        self.alice_fracture = self._report(
            self.alice, self.doctor, "bone_fracture", self.base_time + timedelta(days=9),
            task_type="detection", boxes=[{"class": "fracture", "confidence": 0.7}, {"class": "fracture", "confidence": 0.6}],
            label="fracture", confidence=0.7,
        )
        # Bob: belongs to Doctor Two only (unrelated to Doctor One).
        self.bob_report = self._report(self.bob, self.other_doctor, "skin_cancer", self.base_time + timedelta(days=2), label="nv", confidence=0.8)
        # Carol: belongs to Doctor Two but forwarded to Doctor One.
        self.carol_report = self._report(self.carol, self.other_doctor, "brain_tumor", self.base_time + timedelta(days=3), task_type="detection", boxes=[], label="No Detection", confidence=0.0)
        self.carol_report.forwarded_to_doctor_id = self.doctor.id
        self.db.commit()

        app = FastAPI()
        app.include_router(doctor_reports.router, prefix="/api/v1/reports")

        def test_db():
            yield self.db

        app.dependency_overrides[get_db] = test_db
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    def _headers(self, user):
        return {"Authorization": f"Bearer {create_access_token({'sub': user.username})}"}

    def _report(self, patient, doctor, scan_type, tested_at, label=None, confidence=None,
                task_type="classification", boxes=None, approved=False):
        appointment = Appointment(patient_id=patient.id, doctor_id=doctor.id, status="in_progress")
        self.db.add(appointment)
        self.db.flush()
        scan = Scan(
            id=str(uuid.uuid4()), user_id=patient.id, filename="study.png", scan_type=scan_type,
            modality="X-ray", file_path="unused", status="analyzed", uploaded_at=tested_at, lab_tech_id=self.lab.id,
        )
        self.db.add(scan)
        self.db.flush()
        order = DiagnosticOrder(
            appointment_id=appointment.id, ordering_doctor_id=doctor.id, scan_type=scan_type,
            organ="chest", status="completed", scan_id=scan.id,
        )
        result = Result(
            scan_id=scan.id, task_type=task_type, top_label=label, confidence=confidence,
            severity=None, bounding_boxes=json.dumps(boxes or []),
            localization_type="bbox" if task_type == "detection" else "heatmap",
        )
        report = Report(
            scan_id=scan.id, report_json=json.dumps({"findings": "x"}),
            doctor_approved_at=tested_at + timedelta(hours=1) if approved else None,
            reviewed_by_doctor_id=doctor.id if approved else None,
        )
        self.db.add_all([order, result, report])
        self.db.commit()
        return report

    async def asyncTearDown(self):
        await self.client.aclose()
        self.db.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    async def _get(self, user, **params):
        return await self.client.get("/api/v1/reports/doctor", headers=self._headers(user), params=params)

    async def test_requires_authentication(self):
        response = await self.client.get("/api/v1/reports/doctor")
        self.assertIn(response.status_code, [401, 403])

    async def test_non_doctor_roles_are_forbidden(self):
        for user in (self.alice, self.lab):
            response = await self._get(user)
            self.assertEqual(response.status_code, 403, user.role)

    async def test_doctor_sees_own_and_forwarded_but_not_unrelated(self):
        body = (await self._get(self.doctor)).json()
        ids = {item["report_id"] for item in body["reports"]}
        self.assertEqual(ids, {self.alice_first.id, self.alice_second.id, self.alice_fracture.id, self.carol_report.id})
        self.assertNotIn(self.bob_report.id, ids)
        carol = next(item for item in body["reports"] if item["report_id"] == self.carol_report.id)
        self.assertTrue(carol["forwarded_to_me"])

    async def test_client_supplied_scope_parameters_are_ignored(self):
        body = (await self._get(self.doctor, doctor_id=self.other_doctor.id, patient_id=self.bob.id, role="admin")).json()
        self.assertNotIn(self.bob_report.id, {item["report_id"] for item in body["reports"]})

    async def test_search_is_case_insensitive_on_name_and_username(self):
        by_name = (await self._get(self.doctor, search="aLiCe sHaR")).json()
        self.assertEqual({item["patient"]["id"] for item in by_name["reports"]}, {self.alice.id})
        self.assertEqual(by_name["total"], 3)
        by_username = (await self._get(self.doctor, search="CAROL.P")).json()
        self.assertEqual([item["report_id"] for item in by_username["reports"]], [self.carol_report.id])

    async def test_search_cannot_reach_unrelated_patient(self):
        body = (await self._get(self.doctor, search="bob")).json()
        self.assertEqual(body, {"reports": [], "total": 0})

    async def test_ordering_and_exact_timestamps(self):
        newest = (await self._get(self.doctor, search="alice")).json()["reports"]
        self.assertEqual([item["report_id"] for item in newest], [self.alice_fracture.id, self.alice_second.id, self.alice_first.id])
        oldest = (await self._get(self.doctor, search="alice", sort="oldest")).json()["reports"]
        self.assertEqual([item["report_id"] for item in oldest], [self.alice_first.id, self.alice_second.id, self.alice_fracture.id])
        first = oldest[0]
        self.assertEqual(first["test_date"], "2026-01-10")
        self.assertEqual(first["test_time"], "09:30:15")

    async def test_status_and_type_filters(self):
        reviewed = (await self._get(self.doctor, status="reviewed")).json()["reports"]
        self.assertEqual([item["report_id"] for item in reviewed], [self.alice_second.id])
        self.assertTrue(reviewed[0]["doctor_approved"])
        fractures = (await self._get(self.doctor, scan_type="bone_fracture")).json()["reports"]
        self.assertEqual([item["report_id"] for item in fractures], [self.alice_fracture.id])

    async def test_detection_is_not_converted_to_classification(self):
        reports = {item["report_id"]: item for item in (await self._get(self.doctor)).json()["reports"]}
        fracture = reports[self.alice_fracture.id]
        self.assertEqual(fracture["task_type"], "detection")
        self.assertIsNone(fracture["severity"])
        self.assertEqual(fracture["detection"], {"count": 2, "classes": ["fracture"]})
        carol = reports[self.carol_report.id]
        self.assertIsNone(carol["top_label"])
        self.assertIsNone(carol["confidence"])
        self.assertEqual(carol["detection"]["count"], 0)
        pneumonia = reports[self.alice_first.id]
        self.assertIsNone(pneumonia["detection"])
        self.assertEqual(pneumonia["top_label"], "PNEUMONIA")

    async def test_empty_result_for_doctor_without_patients(self):
        lonely = crud.create_user(self.db, "doc.three", "hash", role="doctor", full_name="Doctor Three")
        body = (await self._get(lonely)).json()
        self.assertEqual(body, {"reports": [], "total": 0})

    async def test_admin_sees_all_patient_reports(self):
        body = (await self._get(self.admin)).json()
        self.assertEqual(body["total"], 5)

    async def test_no_secrets_exposed(self):
        raw = (await self._get(self.doctor)).text.lower()
        for forbidden in ("hashed_password", "password", "token", "secret"):
            self.assertNotIn(forbidden, raw)


if __name__ == "__main__":
    unittest.main()
