import os
import uuid
from typing import Tuple, Optional
import cloudinary
import cloudinary.uploader
import cloudinary.api
from io import BytesIO
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))
load_dotenv(".env")
load_dotenv("../.env")

# Configure Cloudinary using environment variables
CLOUDINARY_ENABLED = os.environ.get("CLOUDINARY_ENABLED", "false").lower() == "true"
if CLOUDINARY_ENABLED:
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
        secure=True
    )

CLOUDINARY_FOLDER = os.environ.get("CLOUDINARY_FOLDER", "medritt")



def _upload_to_cloudinary(file_data, folder_path: str) -> Tuple[str, str]:
    if not CLOUDINARY_ENABLED:
        raise RuntimeError("Cloudinary is not enabled or configured.")

    unique_id = str(uuid.uuid4())
    public_id = f"{CLOUDINARY_FOLDER}/{folder_path}/{unique_id}"

    try:
        response = cloudinary.uploader.upload(
            file_data,
            public_id=public_id,
            resource_type="image",
            overwrite=True
        )
        return response.get("secure_url"), response.get("public_id")
    except Exception as e:
        logger.error(f"Cloudinary upload failed: {str(e)}")
        raise RuntimeError("Failed to upload image to Cloudinary.")


def upload_original_scan(file_data) -> Tuple[str, str]:
    return _upload_to_cloudinary(file_data, "scans/originals")


def upload_gradcam(image_array: np.ndarray) -> Tuple[str, str]:
    img = Image.fromarray(image_array)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return _upload_to_cloudinary(buffer, "scans/gradcam")


def upload_yolo_overlay(image_array: np.ndarray) -> Tuple[str, str]:
    img = Image.fromarray(image_array)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return _upload_to_cloudinary(buffer, "scans/detection-overlays")


def upload_avatar(file_path: str) -> Tuple[str, str]:
    return _upload_to_cloudinary(file_path, "avatars")


def delete_asset(public_id: str) -> bool:
    if not CLOUDINARY_ENABLED or not public_id:
        return False
    try:
        cloudinary.uploader.destroy(public_id)
        return True
    except Exception as e:
        logger.error(f"Failed to delete Cloudinary asset {public_id}: {str(e)}")
        return False


def upload_report_pdf(file_path: str) -> Tuple[str, str]:
    if not CLOUDINARY_ENABLED:
        raise RuntimeError("Cloudinary is not enabled or configured.")
    import uuid
    unique_id = str(uuid.uuid4())
    public_id = f"{CLOUDINARY_FOLDER}/reports/{unique_id}"
    try:
        response = cloudinary.uploader.upload(
            file_path,
            public_id=public_id,
            resource_type="raw",
            overwrite=True
        )
        return response.get("secure_url"), response.get("public_id")
    except Exception as e:
        logger.error(f"Cloudinary PDF upload failed: {str(e)}")
        raise RuntimeError("Failed to upload PDF to Cloudinary.")
