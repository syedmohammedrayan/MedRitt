"""
MedRittAI — Authentication Router
JWT-based login with bcrypt password hashing.
"""

import logging
import os
import re
import time as _time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, BackgroundTasks
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings
from db.database import get_db
from db import crud
from models.schemas import (
    LoginRequest, LoginResponse, ProfileUpdate, RegisterRequest, UserSummary,
    SecurityQuestionSetupRequest, SecurityQuestionResponse,
    ForgotPasswordIdentifyRequest, ForgotPasswordVerifyRequest,
    ForgotPasswordVerifyResponse, ForgotPasswordResetRequest
)

logger = logging.getLogger(__name__)

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


def serialize_user(user) -> dict:
    """Return the safe, role-aware identity shape used across the API."""
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "full_name": user.full_name or user.username,
        "email": user.email or "",
        "phone": user.phone or "",
        "avatar_url": user.avatar_url or "",
        "specialization": user.specialization or "",
        "qualification": user.qualification or "",
        "department_id": user.department_id,
        "department_name": user.department.name if user.department else None,
        "is_active": bool(user.is_active),
        "is_available": bool(user.is_available),
        "availability_note": user.availability_note or "",
    }


def create_access_token(data: dict) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=settings.JWT_EXPIRY_HOURS)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

# --- Security Questions Utilities ---

SECURITY_QUESTIONS_BANK = {
    "DOB": "What is your date of birth?",
    "BIRTH_PLACE": "What city or place were you born in?",
    "FIRST_SCHOOL": "What was the name of your first school?",
    "FAVORITE_SPORT": "What is your favorite sport?",
    "CHILDHOOD_FRIEND": "What was the name of your childhood best friend?",
    "FAVORITE_SUBJECT": "What is your favorite subject?",
    "FAVORITE_FOOD": "What is your favorite food?",
    "FIRST_TEACHER": "What was the name of your first teacher?",
}

def normalize_answer(answer: str) -> str:
    """Normalize security question answer: lowercase, strip, remove extra spaces/punctuation."""
    s = str(answer).lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def hash_answer(answer: str) -> str:
    return pwd_context.hash(normalize_answer(answer))

def verify_answer(plain_answer: str, hashed_answer: str) -> bool:
    return pwd_context.verify(normalize_answer(plain_answer), hashed_answer)


# --- Rate Limiting for Forgot Password ---
_reset_attempts: dict[str, list[float]] = {}
_consumed_reset_jtis: set[str] = set()
_RESET_MAX_ATTEMPTS = 5
_RESET_WINDOW_SECONDS = 900  # 15 minutes

def _check_rate_limit(identifier: str):
    """Raise 429 if too many forgot-password attempts for this identifier."""
    now = _time.time()
    key = identifier.strip().lower()
    attempts = _reset_attempts.get(key, [])
    attempts = [t for t in attempts if now - t < _RESET_WINDOW_SECONDS]
    if len(attempts) >= _RESET_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many password reset attempts. Please try again later."
        )
    attempts.append(now)
    _reset_attempts[key] = attempts


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    """
    FastAPI dependency: extracts and validates JWT from Authorization header.
    Returns the authenticated User object.
    """
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = crud.get_user_by_username(db, username)
    if user is None or not user.is_active:
        raise credentials_exception

    return user


def require_roles(*allowed_roles: str):
    """Create a FastAPI dependency that authorizes one or more user roles."""
    async def role_guard(current_user=Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires one of these roles: {', '.join(allowed_roles)}",
            )
        return current_user

    return role_guard


# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate user and return JWT token.
    """
    user = crud.get_user_by_username(db, request.username)

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id, "role": user.role}
    )

    logger.info(f"User '{user.username}' logged in successfully.")

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.JWT_EXPIRY_HOURS * 3600,
        user=UserSummary(**serialize_user(user)),
    )


@router.post("/register", response_model=LoginResponse, status_code=201)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Self-register a patient and return an authenticated session."""
    username = request.username.strip().lower()
    if crud.get_user_by_username(db, username):
        raise HTTPException(status_code=409, detail="Username is already registered")

    user = crud.create_user(
        db,
        username=username,
        hashed_password=pwd_context.hash(request.password),
        role="patient",
        full_name=request.full_name.strip(),
        email=request.email.strip(),
        phone=request.phone.strip(),
    )
    token = create_access_token(
        {"sub": user.username, "user_id": user.id, "role": user.role}
    )
    return LoginResponse(
        access_token=token,
        expires_in=settings.JWT_EXPIRY_HOURS * 3600,
        user=UserSummary(**serialize_user(user)),
    )


@router.get("/me", response_model=UserSummary)
async def me(current_user=Depends(get_current_user)):
    return UserSummary(**serialize_user(current_user))


@router.patch("/me", response_model=UserSummary)
async def update_me(request: ProfileUpdate, current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    current_user.full_name = request.full_name.strip()
    current_user.email = request.email.strip()
    current_user.phone = request.phone.strip()
    db.commit()
    db.refresh(current_user)
    return UserSummary(**serialize_user(current_user))


@router.post("/me/avatar", response_model=UserSummary)
async def upload_avatar(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    allowed_types = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    extension = allowed_types.get(file.content_type or "")
    if not extension:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP image")
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Profile image must be 5 MB or smaller")
    try:
        from services.cloudinary_storage import upload_avatar as upload_cloud_avatar, delete_asset

        old_pub_id = getattr(current_user, "avatar_public_id", None)
        avatar_url, pub_id = upload_cloud_avatar(content)
        current_user.avatar_url = avatar_url
        current_user.avatar_public_id = pub_id

        if old_pub_id:
            background_tasks.add_task(delete_asset, old_pub_id)

    except Exception as e:
        logger.error(f"Avatar upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload avatar to cloud storage.")
    db.commit()
    db.refresh(current_user)
    return UserSummary(**serialize_user(current_user))


# ============================================================
# SECURITY QUESTIONS & PASSWORD RECOVERY
# ============================================================

@router.get("/security-questions/bank", response_model=list[SecurityQuestionResponse])
async def get_security_questions_bank():
    """Get the fixed bank of security questions."""
    return [
        SecurityQuestionResponse(question_id=qid, question=text)
        for qid, text in SECURITY_QUESTIONS_BANK.items()
    ]


@router.get("/security-questions", response_model=list[SecurityQuestionResponse])
async def get_my_security_questions(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Get the current user's configured security questions (no answers)."""
    sqs = crud.get_user_security_questions(db, current_user.id)
    return [
        SecurityQuestionResponse(
            question_id=sq.question_id,
            question=SECURITY_QUESTIONS_BANK.get(sq.question_id, "Unknown Question")
        )
        for sq in sqs
    ]


@router.put("/security-questions")
async def setup_security_questions(
    request: SecurityQuestionSetupRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Configure or update security questions."""
    # Ensure distinct questions
    if not request.questions:
        raise HTTPException(status_code=400, detail="At least one security question is required.")

    q_ids = [q.question_id for q in request.questions]
    if len(set(q_ids)) != len(q_ids):
        raise HTTPException(status_code=400, detail="Duplicate questions selected.")

    for q in request.questions:
        if q.question_id not in SECURITY_QUESTIONS_BANK:
            raise HTTPException(status_code=400, detail=f"Invalid question_id: {q.question_id}")
        if not q.answer or not q.answer.strip():
            raise HTTPException(status_code=400, detail="All selected questions must be answered.")

    questions_data = [
        {
            "question_id": q.question_id,
            "answer_hash": hash_answer(q.answer)
        }
        for q in request.questions
    ]

    crud.set_user_security_questions(db, current_user.id, questions_data)
    return {"detail": "Security questions updated successfully."}


@router.post("/forgot-password/identify", response_model=list[SecurityQuestionResponse])
async def forgot_password_identify(request: ForgotPasswordIdentifyRequest, db: Session = Depends(get_db)):
    """Identify a user and return their security questions."""
    _check_rate_limit(request.identifier)

    user = crud.get_user_by_identifier(db, request.identifier.strip().lower())
    if not user or not user.is_active:
        raise HTTPException(status_code=404, detail="User not found.")

    sqs = crud.get_user_security_questions(db, user.id)
    if not sqs:
        raise HTTPException(status_code=400, detail="Password recovery has not been configured for this account. Please contact your administrator.")

    return [
        SecurityQuestionResponse(
            question_id=sq.question_id,
            question=SECURITY_QUESTIONS_BANK.get(sq.question_id, "Unknown Question")
        )
        for sq in sqs
    ]


@router.post("/forgot-password/verify", response_model=ForgotPasswordVerifyResponse)
async def forgot_password_verify(request: ForgotPasswordVerifyRequest, db: Session = Depends(get_db)):
    """Verify answers and return a short-lived reset token."""
    _check_rate_limit(request.identifier)

    user = crud.get_user_by_identifier(db, request.identifier.strip().lower())
    if not user or not user.is_active:
        raise HTTPException(status_code=404, detail="User not found.")

    sqs = crud.get_user_security_questions(db, user.id)
    if not sqs:
        raise HTTPException(status_code=400, detail="Password recovery has not been configured for this account. Please contact your administrator.")

    # Check all answers provided match what is saved
    saved_answers = {sq.question_id: sq.answer_hash for sq in sqs}
    provided_answers = {q.question_id: q.answer for q in request.answers}

    if len(provided_answers) != len(saved_answers):
        raise HTTPException(status_code=400, detail="The verification answers could not be confirmed.")

    for q_id, hash_val in saved_answers.items():
        if q_id not in provided_answers:
            raise HTTPException(status_code=400, detail="The verification answers could not be confirmed.")
        if not verify_answer(provided_answers[q_id], hash_val):
            raise HTTPException(status_code=400, detail="The verification answers could not be confirmed.")


    # Success, generate reset token (15 mins)
    expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    jti = str(uuid.uuid4())
    to_encode = {
        "sub": str(user.id),
        "scope": "password_reset",
        "hash_prefix": user.hashed_password[:10],
        "jti": jti,
        "exp": expire,
        "iat": datetime.now(timezone.utc)
    }
    reset_token = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    return ForgotPasswordVerifyResponse(reset_token=reset_token)


@router.post("/forgot-password/reset")
async def forgot_password_reset(request: ForgotPasswordResetRequest, db: Session = Depends(get_db)):
    """Reset password using the single-use reset token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired reset token",
    )

    try:
        payload = jwt.decode(request.reset_token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id_str: str = payload.get("sub")
        scope: str = payload.get("scope")
        hash_prefix: str = payload.get("hash_prefix")
        jti: str | None = payload.get("jti")

        if scope != "password_reset" or not user_id_str or not hash_prefix:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    user = crud.get_user(db, int(user_id_str))
    if not user or not user.is_active:
        raise credentials_exception

    # Check single-use constraint (both jti and hash_prefix)
    if jti and jti in _consumed_reset_jtis:
        raise HTTPException(status_code=400, detail="Token has already been used.")

    if user.hashed_password[:10] != hash_prefix:
        raise HTTPException(status_code=400, detail="Token has already been used.")

    if jti:
        _consumed_reset_jtis.add(jti)

    # Update password using crud operation
    crud.update_user_password(db, user.id, pwd_context.hash(request.new_password))

    return {"detail": "Password has been successfully reset."}
