import pytest
import time
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.database import Base, get_db
from main import app
from db import crud
from routers.auth import SECURITY_QUESTIONS_BANK, hash_answer

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    import routers.auth
    routers.auth._reset_attempts.clear()

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    # Create test user A (patient)
    userA = crud.create_user(
        db,
        username="patient_a",
        email="patient_a@example.com",
        hashed_password="initial_hash_a",
        role="patient"
    )
    # Create test user B (doctor)
    userB = crud.create_user(
        db,
        username="doctor_b",
        email="doctor_b@example.com",
        hashed_password="initial_hash_b",
        role="doctor"
    )
    db.close()
    yield


# A. User configures 1 question → Forgot Password shows exactly 1 question
def test_a_user_configures_1_question_shows_exactly_1():
    db = TestingSessionLocal()
    user = crud.get_user_by_username(db, "patient_a")
    crud.set_user_security_questions(db, user.id, [
        {"question_id": "DOB", "answer_hash": hash_answer("1995-01-01")}
    ])
    db.close()

    res = client.post("/api/v1/auth/forgot-password/identify", json={"identifier": "patient_a"})
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["question_id"] == "DOB"
    assert data[0]["question"] == SECURITY_QUESTIONS_BANK["DOB"]


# B. User configures 2 questions → Forgot Password shows exactly those 2
def test_b_user_configures_2_questions_shows_exactly_those_2():
    db = TestingSessionLocal()
    user = crud.get_user_by_username(db, "patient_a")
    crud.set_user_security_questions(db, user.id, [
        {"question_id": "BIRTH_PLACE", "answer_hash": hash_answer("Chicago")},
        {"question_id": "FAVORITE_SPORT", "answer_hash": hash_answer("Tennis")}
    ])
    db.close()

    res = client.post("/api/v1/auth/forgot-password/identify", json={"identifier": "patient_a"})
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 2
    assert {q["question_id"] for q in data} == {"BIRTH_PLACE", "FAVORITE_SPORT"}


# C. User configures 3 questions → Forgot Password shows exactly those 3
def test_c_user_configures_3_questions_shows_exactly_those_3():
    db = TestingSessionLocal()
    user = crud.get_user_by_username(db, "patient_a")
    crud.set_user_security_questions(db, user.id, [
        {"question_id": "FIRST_SCHOOL", "answer_hash": hash_answer("Lincoln High")},
        {"question_id": "FAVORITE_FOOD", "answer_hash": hash_answer("Pizza")},
        {"question_id": "CHILDHOOD_FRIEND", "answer_hash": hash_answer("Sam")}
    ])
    db.close()

    res = client.post("/api/v1/auth/forgot-password/identify", json={"identifier": "patient_a"})
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 3
    assert {q["question_id"] for q in data} == {"FIRST_SCHOOL", "FAVORITE_FOOD", "CHILDHOOD_FRIEND"}


# D. Question bank contains 8 questions → Forgot Password does NOT show all 8
def test_d_question_bank_contains_8_questions_forgot_password_does_not_show_all_8():
    assert len(SECURITY_QUESTIONS_BANK) == 8

    db = TestingSessionLocal()
    user = crud.get_user_by_username(db, "patient_a")
    crud.set_user_security_questions(db, user.id, [
        {"question_id": "BIRTH_PLACE", "answer_hash": hash_answer("Paris")},
        {"question_id": "FAVORITE_SPORT", "answer_hash": hash_answer("Football")}
    ])
    db.close()

    res = client.post("/api/v1/auth/forgot-password/identify", json={"identifier": "patient_a"})
    data = res.json()
    assert len(data) == 2
    assert len(data) != len(SECURITY_QUESTIONS_BANK)
    returned_ids = {q["question_id"] for q in data}
    assert "FIRST_SCHOOL" not in returned_ids
    assert "DOB" not in returned_ids
    assert "CHILDHOOD_FRIEND" not in returned_ids
    assert "FAVORITE_SUBJECT" not in returned_ids
    assert "FAVORITE_FOOD" not in returned_ids
    assert "FIRST_TEACHER" not in returned_ids


# E. User A cannot receive User B's questions
def test_e_user_a_cannot_receive_user_b_questions():
    db = TestingSessionLocal()
    user_a = crud.get_user_by_username(db, "patient_a")
    user_b = crud.get_user_by_username(db, "doctor_b")
    crud.set_user_security_questions(db, user_a.id, [
        {"question_id": "DOB", "answer_hash": hash_answer("1990-01-01")}
    ])
    crud.set_user_security_questions(db, user_b.id, [
        {"question_id": "FAVORITE_SPORT", "answer_hash": hash_answer("Cricket")},
        {"question_id": "FIRST_TEACHER", "answer_hash": hash_answer("Mr. Smith")}
    ])
    db.close()

    res_a = client.post("/api/v1/auth/forgot-password/identify", json={"identifier": "patient_a"})
    data_a = res_a.json()
    assert len(data_a) == 1
    assert data_a[0]["question_id"] == "DOB"

    res_b = client.post("/api/v1/auth/forgot-password/identify", json={"identifier": "doctor_b"})
    data_b = res_b.json()
    assert len(data_b) == 2
    assert {q["question_id"] for q in data_b} == {"FAVORITE_SPORT", "FIRST_TEACHER"}
    assert "DOB" not in {q["question_id"] for q in data_b}


# F. User cannot choose questions through frontend manipulation during reset
def test_f_user_cannot_choose_questions_through_frontend_manipulation():
    db = TestingSessionLocal()
    user = crud.get_user_by_username(db, "patient_a")
    crud.set_user_security_questions(db, user.id, [
        {"question_id": "BIRTH_PLACE", "answer_hash": hash_answer("Dallas")}
    ])
    db.close()

    # Even if client sends a different question_id in verify request, backend evaluates against configured questions
    res = client.post("/api/v1/auth/forgot-password/verify", json={
        "identifier": "patient_a",
        "answers": [{"question_id": "FAVORITE_SPORT", "answer": "Soccer"}]
    })
    assert res.status_code == 400
    assert "The verification answers could not be confirmed." in res.json()["detail"]


# G. No configured questions → recovery unavailable message
def test_g_no_configured_questions_shows_recovery_unavailable_message():
    res = client.post("/api/v1/auth/forgot-password/identify", json={"identifier": "patient_a"})
    assert res.status_code == 400
    assert "Password recovery has not been configured for this account. Please contact your administrator." in res.json()["detail"]


# H. Correct configured answers → reset token issued
def test_h_correct_configured_answers_issues_reset_token():
    db = TestingSessionLocal()
    user = crud.get_user_by_username(db, "patient_a")
    crud.set_user_security_questions(db, user.id, [
        {"question_id": "BIRTH_PLACE", "answer_hash": hash_answer("Dallas")},
        {"question_id": "FAVORITE_SPORT", "answer_hash": hash_answer("Basketball")}
    ])
    db.close()

    res = client.post("/api/v1/auth/forgot-password/verify", json={
        "identifier": "patient_a",
        "answers": [
            {"question_id": "BIRTH_PLACE", "answer": "dallas"},
            {"question_id": "FAVORITE_SPORT", "answer": "  basketball  "}
        ]
    })
    assert res.status_code == 200
    data = res.json()
    assert "reset_token" in data
    assert isinstance(data["reset_token"], str) and len(data["reset_token"]) > 20


# I. Wrong configured answers → reset denied
def test_i_wrong_configured_answers_denied():
    db = TestingSessionLocal()
    user = crud.get_user_by_username(db, "patient_a")
    crud.set_user_security_questions(db, user.id, [
        {"question_id": "BIRTH_PLACE", "answer_hash": hash_answer("Dallas")},
        {"question_id": "FAVORITE_SPORT", "answer_hash": hash_answer("Basketball")}
    ])
    db.close()

    res = client.post("/api/v1/auth/forgot-password/verify", json={
        "identifier": "patient_a",
        "answers": [
            {"question_id": "BIRTH_PLACE", "answer": "Dallas"},
            {"question_id": "FAVORITE_SPORT", "answer": "WrongSport"}
        ]
    })
    assert res.status_code == 400
    assert "The verification answers could not be confirmed." in res.json()["detail"]


# J. Answers never returned by API
def test_j_answers_never_returned_by_api():
    db = TestingSessionLocal()
    user = crud.get_user_by_username(db, "patient_a")
    crud.set_user_security_questions(db, user.id, [
        {"question_id": "BIRTH_PLACE", "answer_hash": hash_answer("SecretCity")}
    ])
    db.close()

    res = client.post("/api/v1/auth/forgot-password/identify", json={"identifier": "patient_a"})
    assert res.status_code == 200
    for item in res.json():
        assert "answer" not in item
        assert "SecretCity" not in str(item)


# K. Answer hashes never returned by API
def test_k_answer_hashes_never_returned_by_api():
    db = TestingSessionLocal()
    user = crud.get_user_by_username(db, "patient_a")
    secret_hash = hash_answer("SecretCity")
    crud.set_user_security_questions(db, user.id, [
        {"question_id": "BIRTH_PLACE", "answer_hash": secret_hash}
    ])
    db.close()

    res = client.post("/api/v1/auth/forgot-password/identify", json={"identifier": "patient_a"})
    assert res.status_code == 200
    for item in res.json():
        assert "answer_hash" not in item
        assert secret_hash not in str(item)


# L. Reset token remains short-lived and single-use
def test_l_reset_token_is_single_use_and_resets_password():
    db = TestingSessionLocal()
    user = crud.get_user_by_username(db, "patient_a")
    crud.set_user_security_questions(db, user.id, [
        {"question_id": "BIRTH_PLACE", "answer_hash": hash_answer("Dallas")}
    ])
    db.close()

    verify_res = client.post("/api/v1/auth/forgot-password/verify", json={
        "identifier": "patient_a",
        "answers": [{"question_id": "BIRTH_PLACE", "answer": "Dallas"}]
    })
    token = verify_res.json()["reset_token"]

    # First reset usage succeeds
    reset_res1 = client.post("/api/v1/auth/forgot-password/reset", json={
        "reset_token": token,
        "new_password": "brand_new_secure_password_123"
    })
    assert reset_res1.status_code == 200
    assert "successfully reset" in reset_res1.json()["detail"]

    # Verify login with new password works
    login_res = client.post("/api/v1/auth/login", json={
        "username": "patient_a",
        "password": "brand_new_secure_password_123"
    })
    assert login_res.status_code == 200
    assert "access_token" in login_res.json()

    # Re-using the same reset token fails immediately
    reset_res2 = client.post("/api/v1/auth/forgot-password/reset", json={
        "reset_token": token,
        "new_password": "yet_another_password"
    })
    assert reset_res2.status_code == 400
    assert "already been used" in reset_res2.json()["detail"]
