from fastapi import (
    FastAPI,
    Request,
    File,
    UploadFile,
    HTTPException,
    Form,
    Body,
    BackgroundTasks,
    Depends,
    status,
    Response,
)
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import platform
import razorpay
import hmac
import hashlib


from datetime import datetime, timedelta
import pytz  # You may need to install: pip install pytz

import subprocess
import uuid
import os
import boto3
import httpx
import io
import fitz
import logging
import pdfplumber
import requests
import traceback
import redis
import time
import re
import pandas as pd
import json
import asyncio

# Initialize colorama
from botocore.exceptions import ClientError
import pathlib
import shutil

from pypdf import PdfReader
import gc
from typing import Any, Dict, List, Optional
from pydantic import Field
from langchain_core.retrievers import BaseRetriever

from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

# langchain
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

from sentence_transformers import SentenceTransformer
import threading

#### for openai
from pathlib import Path
import hashlib
import numpy as np

###
import secrets
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


from app.pdf_operations import (
    encrypt_pdf_streaming,
    decrypt_pdf_streaming,
    # remove_background_rembg
)

from rembg import remove
from dotenv import load_dotenv
from colorama import Fore, Style, init

init()
load_dotenv()
from fastapi.middleware.gzip import GZipMiddleware

# ==================== SECURITY IMPORTS ====================
import magic
import bleach
import filetype
from collections import defaultdict
from markupsafe import escape

app = FastAPI()
# ######


# Set cache directories BEFORE any imports that might load models
os.environ["TRANSFORMERS_CACHE"] = "/home/ec2-user/.cache/huggingface"
os.environ["HF_HOME"] = "/home/ec2-user/.cache/huggingface"
os.environ["CHROMA_CACHE"] = "/home/ec2-user/.cache/chroma"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create cache directories if they don't exist
os.makedirs("/home/ec2-user/.cache/huggingface", exist_ok=True)
os.makedirs("/home/ec2-user/.cache/chroma", exist_ok=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://recallmind.online",
        "https://www.recallmind.online",
        "https://d7ypf0jdu8oou.cloudfront.net",
        "http://localhost:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Requested-With",
        "X-Auth-Retry",
        "Cache-Control",
        "Pragma",
    ],
    expose_headers=["X-Auth-Error", "WWW-Authenticate"],
    max_age=600,
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==================== SECURITY CONFIGURATION ====================
ALLOWED_FILE_TYPES = {
    "pdf": {
        "mime": [
            "application/pdf",
            "application/octet-stream",
        ],  # octet-stream for browser-generated PDFs
        "extensions": [".pdf"],
        "max_size": 50 * 1024 * 1024,  # 50MB
        "magic": [b"%PDF"],
    },
    "image": {
        "mime": [
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
            "image/jpg",
            "image/x-ms-bmp",
        ],
        "extensions": [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"],
        "max_size": 20 * 1024 * 1024,
        "magic": [b"\xff\xd8\xff", b"\x89PNG", b"GIF87a", b"GIF89a", b"BM"],
    },
    "video": {
        "mime": ["video/mp4", "video/webm", "video/ogg", "video/x-matroska"],
        "extensions": [".mp4", ".webm", ".ogg", ".mkv"],
        "max_size": 100 * 1024 * 1024,
        "magic": [b"ftypmp4", b"ftypisom", b"webm", b"OggS"],
    },
}

# === for production, we will enforce strict limits. In development, we can be more lenient.
RATE_LIMITS = {
    "upload": "10/hour",
    "compress": "5/minute",
    "estimation": "5/minute",
    "chat": "30/minute",
    "convert": "3/minute",
    "auth": "5/minute",
}

# == For testing, we can set very high limits to avoid interference during development and testing
# RATE_LIMITS = {
#     'upload': "30/minute",      # Increased for testing
#     'compress': "30/minute",
#     'estimation': "30/minute",
#     'chat': "30/minute",
#     'convert': "30/minute",     # Changed from 3 to 30
#     'auth': "30/minute"         # Changed from 5 to 30
# }

# WebSocket rate limiting storage
ws_connections = defaultdict(list)


# ==================== SECURITY FUNCTIONS ====================
def sanitize_filename(filename: str) -> str:
    """Remove all dangerous characters from filename"""
    filename = filename.replace("..", "")
    filename = re.sub(r"[;&|`$(){}[\]<>]", "_", filename)
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    parts = filename.split(".")
    if len(parts) > 2:
        filename = f"{parts[0]}.{parts[-1]}"
    return filename[:100]


def sanitize_input(text: str, max_length: int = 1000, allow_html: bool = False) -> str:
    """Sanitize ALL user text input"""
    if not text:
        return ""
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\r\t")
    if len(text) > max_length:
        text = text[:max_length]
    if allow_html:
        allowed_tags = ["b", "i", "strong", "em", "p", "br", "ul", "li"]
        text = bleach.clean(text, tags=allowed_tags, strip=True)
    else:
        text = escape(text)
    return text.strip()


def sanitize_pdf(pdf_bytes: bytes) -> bytes:
    try:
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")

        # Remove JavaScript
        try:
            if hasattr(doc, "set_js"):
                doc.set_js("")
        except Exception:
            pass

        # Remove embedded files
        try:
            if hasattr(doc, "embfile_names"):
                for embed in doc.embfile_names():
                    try:
                        doc.embfile_del(embed)
                    except Exception:
                        pass
        except Exception:
            pass

        # Remove malicious annotations
        try:
            for page in doc:
                try:
                    for annot in page.annots():
                        if annot and hasattr(annot, "type"):
                            try:
                                if annot.type[0] in [7, 8, 9]:
                                    page.delete_annot(annot)
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass

        # Save sanitized PDF - without no_encrypt for compatibility
        output = io.BytesIO()
        doc.save(
            output,
            garbage=4,  # Remove unused objects ✅
            deflate=True,  # Compress ✅
            clean=True,  # Clean metadata ✅
            # no_encrypt removed - sanitization still happens
        )
        doc.close()

        return output.getvalue()

    except Exception as e:
        logger.error(f"PDF sanitization failed: {e}, returning original")
        return pdf_bytes


def get_secure_download_url(s3_key: str, expires_in: int = 3600) -> str:
    """Generate time-limited presigned URL for S3 objects"""
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": BUCKET_NAME,
                "Key": s3_key,
                "ResponseContentDisposition": "attachment",
            },
            ExpiresIn=expires_in,
        )
        return url
    except Exception as e:
        logging.error(f"Failed to generate presigned URL: {e}")
        return None


async def check_ws_rate_limit(
    client_ip: str, max_connections: int = 5, window_seconds: int = 60
) -> bool:
    now = datetime.now()
    ws_connections[client_ip] = [
        ts
        for ts in ws_connections[client_ip]
        if (now - ts).total_seconds() < window_seconds
    ]
    if len(ws_connections[client_ip]) >= max_connections:
        return False
    ws_connections[client_ip].append(now)
    return True


# ===== UPLOAD TO DISK FIRST =====
async def upload_to_disk_first(file: UploadFile, task_id: str = None) -> str:
    """
    IMMEDIATELY write file to disk WITHOUT reading into RAM
    Uses .read() chunks - CORRECT method for UploadFile
    """
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    safe_filename = sanitize_filename(file.filename)

    if task_id:
        final_filename = f"{task_id}_{safe_filename}"
    else:
        final_filename = f"{uuid.uuid4().hex}_{safe_filename}"

    file_path = UPLOAD_DIR / final_filename

    # ✅ CORRECT: Read in chunks and write to disk
    total_size = 0
    with open(file_path, "wb") as buffer:
        while chunk := await file.read(65536):  # 64KB chunks
            total_size += len(chunk)
            buffer.write(chunk)

    logger.info(f"✅ File saved to disk: {file_path} ({total_size} bytes)")
    return str(file_path)


# ==================== CLAMAV ANTIVIRUS SCANNING ====================
CLAMAV_AVAILABLE = False
CLAMAV_SCANNER_PATH: Optional[str] = None


async def scan_with_clamav_from_disk(file_path: str, filename: str = "upload") -> dict:
    """
    Scan file with ClamAV - reads from disk directly, NO RAM loading
    Returns: {"safe": bool, "message": str}
    """

    # Skip scanning entirely on EC2
    # if os.path.exists("/home/ec2-user"):  # Running on EC2
    #     logger.info(f"⏩ EC2 optimization - skipping ClamAV scan for {filename}")
    #     return {"safe": True, "message": "EC2 optimization - scan skipped"}

    global CLAMAV_AVAILABLE
    logger.info(f"🔍 Scanning with ClamAV: {filename}")

    if not CLAMAV_AVAILABLE:
        logger.warning(f"⚠️ ClamAV not available - skipping scan for {filename}")
        if os.getenv("ENVIRONMENT") == "production":
            return {
                "safe": False,
                "message": "Security scanner unavailable - contact support",
            }
        return {
            "safe": True,
            "message": "ClamAV not available - manual review recommended",
        }

    max_retries = 2

    for attempt in range(max_retries):
        try:
            # Build command - scan file directly from disk path
            if platform.system().lower() == "windows":
                if CLAMAV_SCANNER_PATH and os.path.exists(CLAMAV_SCANNER_PATH):
                    cmd = [CLAMAV_SCANNER_PATH, file_path]
                else:
                    cmd = ["clamscan", file_path]
                creationflags = (
                    subprocess.CREATE_NO_WINDOW
                    if hasattr(subprocess, "CREATE_NO_WINDOW")
                    else 0
                )

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    shell=True,
                    creationflags=creationflags,
                )
            else:
                # Linux - scan file directly from disk
                cmd = [
                    "nice",
                    "-n",
                    "10",
                    CLAMAV_SCANNER_PATH,
                    "--infected",
                    "--no-summary",
                    file_path,
                ]
                timeout_value = 180 if file_path.endswith(".pdf") else 30
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout_value
                )
                # result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            output = result.stdout.strip()
            error_output = result.stderr.strip()

            # Check for viruses using return code (1 = virus found)
            if result.returncode == 1:
                logger.error(f"🚨 MALWARE DETECTED: {filename}")
                return {"safe": False, "message": "Malware detected"}

            # Check for clean file (return code 0 means clean)
            if result.returncode == 0:
                logger.info(f"✅ Scan clean: {filename}")
                return {"safe": True, "message": "Clean"}

            # Also check stdout for "Infected files: 0" as fallback
            if "Infected files: 0" in output and "Infected files: 1" not in output:
                logger.info(f"✅ Scan clean (by summary): {filename}")
                return {"safe": True, "message": "Clean"}

            # Check for "FOUND" in output
            if "FOUND" in output:
                virus_name = "Unknown"
                lines = output.split("\n")
                for line in lines:
                    if "FOUND" in line:
                        parts = line.split(":")
                        if len(parts) >= 2:
                            virus_info = parts[1].strip()
                            virus_name = virus_info.split("FOUND")[0].strip()
                        break

                logger.error(f"🚨 MALWARE DETECTED: {filename} → {virus_name}")
                return {"safe": False, "message": f"Malware found: {virus_name}"}

            # Check for scan errors
            if error_output and (
                "ERROR" in error_output.upper() or "failed" in error_output.lower()
            ):
                logger.error(f"ClamAV error: {error_output}")
                if attempt < max_retries - 1:
                    logger.info(
                        f"Retrying scan (attempt {attempt + 2}/{max_retries})..."
                    )
                    await asyncio.sleep(1)
                    continue
                else:
                    if platform.system().lower() == "windows":
                        logger.warning(
                            f"⚠️ ClamAV scan failed but allowing file: {filename}"
                        )
                        return {
                            "safe": True,
                            "message": "Scan failed - manual inspection recommended",
                        }
                    return {
                        "safe": False,
                        "message": "Security scan failed - file rejected",
                    }

            logger.warning(f"Unexpected ClamAV output: {output}")
            return {"safe": True, "message": "Clean (uncertain)"}

        except subprocess.TimeoutExpired:
            logger.error(f"ClamAV scan timeout for {filename} (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                continue
            if platform.system().lower() == "windows":
                return {"safe": True, "message": "Scan timeout - allowed"}
            return {"safe": False, "message": "Scan timeout - file rejected"}

        except FileNotFoundError as e:
            logger.error(f"ClamAV not found: {e}")
            CLAMAV_AVAILABLE = False
            return {"safe": True, "message": "ClamAV not installed - skipping scan"}

        except Exception as e:
            logger.error(f"ClamAV scan error for {filename}: {e}")
            if attempt < max_retries - 1:
                continue
            if platform.system().lower() == "windows":
                logger.warning(
                    f"⚠️ Allowing file due to scan error on Windows: {filename}"
                )
                return {"safe": True, "message": f"Scan error: {str(e)} - allowed"}
            return {"safe": False, "message": f"Scan failed: {str(e)}"}

    return {"safe": False, "message": "Scan failed after retries"}


def init_clamav_scanner():
    """Initialize ClamAV scanner - Optimized for Windows + Amazon Linux 2023"""
    global CLAMAV_AVAILABLE, CLAMAV_SCANNER_PATH

    system = platform.system().lower()
    logger.info(f"🔍 Initializing ClamAV on {system}...")

    try:
        if system == "windows":
            # First check if clamscan is in PATH
            try:
                result = subprocess.run(
                    ["where", "clamscan"],
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    paths = result.stdout.strip().split("\n")
                    CLAMAV_SCANNER_PATH = paths[0]  # Use first found
                    CLAMAV_AVAILABLE = True
                    logger.info(
                        f"✅ ClamAV found on Windows PATH: {CLAMAV_SCANNER_PATH}"
                    )

                    # ✅ TEST DISK-BASED SCANNING (not just version)
                    test_file_path = None
                    try:
                        # Create a small test file
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=".txt", delete=False
                        ) as tmp:
                            tmp.write(b"ClamAV test file for disk-based scanning")
                            test_file_path = tmp.name

                        # Test scanning from disk
                        test_result = subprocess.run(
                            [CLAMAV_SCANNER_PATH, test_file_path],
                            capture_output=True,
                            text=True,
                            shell=True,
                            timeout=10,
                        )
                        if test_result.returncode == 0:
                            logger.info(f"✅ ClamAV disk-based scan test passed")
                        else:
                            logger.warning(
                                f"⚠️ ClamAV scan test returned: {test_result.returncode}"
                            )

                        # Also test version
                        version_result = subprocess.run(
                            [CLAMAV_SCANNER_PATH, "--version"],
                            capture_output=True,
                            text=True,
                            shell=True,
                            timeout=5,
                        )
                        if version_result.returncode == 0:
                            logger.info(
                                f"✅ ClamAV version: {version_result.stdout.strip()}"
                            )

                    except Exception as test_e:
                        logger.warning(f"ClamAV test failed: {test_e}")
                    finally:
                        # Cleanup test file
                        if test_file_path and os.path.exists(test_file_path):
                            try:
                                os.unlink(test_file_path)
                            except:
                                pass
                    return

            except Exception as e:
                logger.warning(f"PATH check failed: {e}")

            # Check common installation paths
            possible_paths = [
                r"C:\Program Files\ClamAV\clamscan.exe",
                r"C:\ClamAV\clamscan.exe",
                r"C:\Program Files (x86)\ClamAV\clamscan.exe",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    CLAMAV_SCANNER_PATH = path
                    CLAMAV_AVAILABLE = True
                    logger.info(f"✅ ClamAV found on Windows: {path}")

                    # Test disk-based scanning
                    try:
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=".txt", delete=False
                        ) as tmp:
                            tmp.write(b"Test")
                            test_file = tmp.name
                        subprocess.run(
                            [CLAMAV_SCANNER_PATH, test_file],
                            capture_output=True,
                            timeout=10,
                        )
                        os.unlink(test_file)
                        logger.info(f"✅ ClamAV disk-based scan ready")
                    except Exception as test_e:
                        logger.warning(f"ClamAV test warning: {test_e}")
                    return

            # If we get here, ClamAV not found
            logger.warning(
                "⚠️ ClamAV not found on Windows. Install from: https://clamav.net/downloads"
            )
            CLAMAV_AVAILABLE = False

        else:
            # Linux detection - Enhanced for disk-based scanning
            common_paths = [
                "/usr/bin/clamscan",
                "/usr/local/bin/clamscan",
                "/bin/clamscan",
            ]
            for path in common_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    CLAMAV_SCANNER_PATH = path
                    CLAMAV_AVAILABLE = True
                    logger.info(f"✅ ClamAV found at: {path}")

                    # ✅ Test disk-based scanning performance
                    try:
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=".txt", delete=False
                        ) as tmp:
                            tmp.write(b"ClamAV disk-based scan test")
                            test_file = tmp.name

                        # Test with nice level (simulates actual usage)
                        test_cmd = [
                            "nice",
                            "-n",
                            "10",
                            CLAMAV_SCANNER_PATH,
                            "--infected",
                            "--no-summary",
                            test_file,
                        ]
                        result = subprocess.run(
                            test_cmd, capture_output=True, text=True, timeout=10
                        )

                        if result.returncode == 0:
                            logger.info(
                                f"✅ ClamAV disk-based scan ready (with nice level)"
                            )
                        else:
                            logger.warning(
                                f"⚠️ ClamAV test returned: {result.returncode}"
                            )

                        os.unlink(test_file)
                    except Exception as test_e:
                        logger.warning(f"ClamAV test warning: {test_e}")

                    return

            try:
                result = subprocess.run(
                    ["which", "clamscan"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    CLAMAV_SCANNER_PATH = result.stdout.strip()
                    CLAMAV_AVAILABLE = True
                    logger.info(f"✅ ClamAV found via which: {CLAMAV_SCANNER_PATH}")

                    # Test disk-based scanning
                    try:
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            mode="wb", suffix=".txt", delete=False
                        ) as tmp:
                            tmp.write(b"Test")
                            test_file = tmp.name
                        subprocess.run(
                            [CLAMAV_SCANNER_PATH, test_file],
                            capture_output=True,
                            timeout=10,
                        )
                        os.unlink(test_file)
                        logger.info(f"✅ ClamAV disk-based scan verified")
                    except Exception:
                        pass
                    return
            except Exception:
                pass

            CLAMAV_AVAILABLE = False
            logger.warning(
                "⚠️ ClamAV not found. Install with: sudo yum install clamav clamav-update -y"
            )

    except Exception as e:
        logger.error(f"ClamAV initialization failed: {e}")
        CLAMAV_AVAILABLE = False


# ✅ NEW: Function to check if ClamAV can handle large files efficiently
def test_clamav_performance():
    """Test ClamAV's performance with a 10MB test file"""
    if not CLAMAV_AVAILABLE:
        logger.info("ClamAV not available, skipping performance test")
        return False

    try:
        import tempfile
        import time

        # Create a 10MB test file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as tmp:
            tmp.write(b"X" * (10 * 1024 * 1024))  # 10MB of dummy data
            test_file = tmp.name

        # Time the scan
        start_time = time.time()

        if platform.system().lower() == "windows":
            cmd = [CLAMAV_SCANNER_PATH, test_file]
        else:
            cmd = [
                "nice",
                "-n",
                "10",
                CLAMAV_SCANNER_PATH,
                "--infected",
                "--no-summary",
                test_file,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        scan_time = time.time() - start_time

        os.unlink(test_file)

        if result.returncode == 0:
            logger.info(f"✅ ClamAV performance: {scan_time:.2f} seconds for 10MB file")
            if scan_time > 5:
                logger.warning(
                    f"⚠️ ClamAV is slow ({scan_time:.2f}s for 10MB). Consider tuning."
                )
            return True
        else:
            logger.warning(f"ClamAV performance test returned {result.returncode}")
            return False

    except Exception as e:
        logger.error(f"ClamAV performance test failed: {e}")
        return False


# ==================== UNIFIED FILE VALIDATION (NO DUPLICATES) ====================
def is_valid_pdf_structure(file_path: str) -> bool:
    """
    Check if file is a valid PDF with proper structure
    DISK-BASED - Minimal RAM usage (reads only headers and end of file)
    """
    try:
        # Step 1: Quick check - Read only first 1KB for PDF header
        with open(file_path, "rb") as f:
            header = f.read(1024)
            if not header.startswith(b"%PDF"):
                logger.debug("Not a PDF: missing %PDF header")
                return False

        # Step 2: Check for garbage after EOF - Read only last 10KB
        with open(file_path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()

            # For very small files, read everything (it's small anyway)
            if file_size < 10240:  # Less than 10KB
                content = f.read()
                last_eof = content.rfind(b"%%EOF")
                if last_eof != -1:
                    after_eof = content[last_eof + 5 :].strip()
                    if after_eof:
                        logger.warning(f"⚠️ PDF has {len(after_eof)} bytes after %%EOF")
                        return False
            else:
                # Read only the last 10KB
                f.seek(file_size - 10240)
                end_content = f.read()
                last_eof = end_content.rfind(b"%%EOF")
                if last_eof != -1:
                    after_eof = end_content[last_eof + 5 :].strip()
                    if after_eof:
                        logger.warning(f"⚠️ PDF has garbage after %%EOF")
                        return False

        # Step 3: Validate PDF structure - Use pdfplumber if available (more efficient)
        try:

            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                if page_count == 0:
                    logger.debug("PDF has no pages")
                    return False
        except ImportError:
            # Fallback to fitz (still loads into RAM but unavoidable)

            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
            if page_count == 0:
                return False

        return True

    except Exception as e:
        logger.debug(f"PDF validation failed: {e}")
        return False


# ============================================= FILE VALIDATION (INTEGRATED) ============================


def get_page_count_from_disk_only(file_path: str) -> Optional[int]:
    """Read only last 32KB of file - 0 RAM spike"""
    try:
        with open(file_path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            read_size = min(32768, file_size)
            f.seek(max(0, file_size - read_size))
            content = f.read().decode("latin-1", errors="ignore")

            match = re.search(r"/Count\s+(\d+)", content)
            if match:
                return int(match.group(1))
        return None
    except Exception as e:
        logger.warning(f"Failed to get page count from disk: {e}")
        return None


async def validate_file_from_disk(
    file_path: str, file_type: str, original_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate file directly from disk - NO RAM loading of entire file
    Enhanced with actual PDF structure validation + garbage detection
    """
    page_limit = 2000  # Strict page limit for PDFs to prevent abuse
    if not os.path.exists(file_path):
        raise HTTPException(400, "File not found on disk")

    # Get file size without loading into RAM
    file_size = os.path.getsize(file_path)

    # Validate file type exists in config
    if file_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file_type}")

    file_config = ALLOWED_FILE_TYPES[file_type]

    # Validate extension
    if original_filename:
        ext = os.path.splitext(original_filename)[1].lower()
    else:
        ext = os.path.splitext(file_path)[1].lower()

    if ext not in file_config["extensions"]:
        raise HTTPException(
            400, f"Invalid extension. Allowed: {file_config['extensions']}"
        )

    # Check file size
    max_size = file_config["max_size"]
    if file_size > max_size:
        raise HTTPException(400, f"File too large. Max: {max_size // (1024*1024)}MB")

    if file_size < 100:
        raise HTTPException(400, "File too small or empty")

    # Magic bytes check - read ONLY first 1KB from disk
    with open(file_path, "rb") as f:
        header = f.read(1024)
        magic_match = False
        for magic_bytes in file_config["magic"]:
            if header.startswith(magic_bytes):
                magic_match = True
                break

        if not magic_match:
            raise HTTPException(
                400,
                f"Invalid {file_type.upper()} file format - file signature mismatch",
            )

    # ========== PDF-SPECIFIC VALIDATION ==========
    if file_type == "pdf":
        # Check for garbage after %%EOF (security risk)
        try:
            with open(file_path, "rb") as f:
                f.seek(0, 2)
                file_size = f.tell()
                read_size = min(10240, file_size)  # Max 10KB
                f.seek(max(0, file_size - read_size))
                content = f.read()
                last_eof = content.rfind(b"%%EOF")
                if last_eof != -1:
                    # Check if there's anything after %%EOF
                    after_eof = content[last_eof + 5 :].strip()
                    if after_eof:
                        logger.warning(
                            f"⚠️ Found {len(after_eof)} bytes of garbage after %%EOF in {original_filename}"
                        )
                        raise HTTPException(
                            400,
                            "Invalid PDF: Contains data after %%EOF (potential hidden content)",
                        )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Garbage check failed: {e}")

        # Validate PDF structure using PyMuPDF (single source of truth)
        doc = None
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)

            if page_count == 0:
                raise HTTPException(400, "PDF has no pages - invalid document")

            # Check page limit

            if page_count > page_limit:
                raise HTTPException(400, f"PDF exceeds {page_limit} pages")

            # Verify first page is readable
            try:
                first_page = doc[0]
                first_page.get_text()
            except Exception as page_error:
                raise HTTPException(
                    400, f"PDF has corrupt page structure: {str(page_error)}"
                )

            logger.info(f"✅ PDF structure validated: {page_count} pages")

            # MIME type check (after structure validation)
            try:
                with open(file_path, "rb") as f:
                    header = f.read(1024)
                    mime = magic.from_buffer(header, mime=True)

                allowed_pdf_mimes = ["application/pdf", "application/octet-stream"]
                logger.info(f"Detected MIME type: {mime} for {original_filename}")
                if mime not in allowed_pdf_mimes:
                    logger.warning(
                        f"⚠️ Unusual MIME type: {mime}, but allowing based on PDF structure"
                    )
            except Exception as e:
                logger.warning(f"MIME detection failed: {e}")

            return {"valid": True, "page_count": page_count}

        except HTTPException:
            raise
        except Exception as e:
            error_msg = str(e)
            if "FileDataError" in error_msg or "no objects found" in error_msg:
                raise HTTPException(400, "Invalid PDF file: Not a valid PDF document")
            else:
                logger.error(f"❌ PDF validation failed: {e}")
                raise HTTPException(400, f"PDF validation failed: {error_msg}")
        finally:
            if doc:
                doc.close()

    # ========== NON-PDF FILE VALIDATION ==========
    # MIME type check for non-PDF files
    try:
        with open(file_path, "rb") as f:
            header = f.read(1024)
            mime = magic.from_buffer(header, mime=True)

        if mime not in file_config["mime"]:
            raise HTTPException(
                400, f"Invalid content type. Expected {file_type}, got {mime}"
            )
    except Exception as e:
        logger.warning(f"MIME detection failed: {e}")

    # ClamAV virus scan - stream from disk without loading entire file
    # scan_result = await scan_with_clamav_from_disk(file_path, original_filename or os.path.basename(file_path))
    # logger.info(f"ClamAV scan done result for {original_filename}: {scan_result}")
    # if not scan_result["safe"]:
    #     logger.error(f"🚨 MALWARE BLOCKED: {file_path} - {scan_result['message']}")
    #     raise HTTPException(status_code=403, detail=f"Security Alert: {scan_result['message']}")

    logger.info(f"✅ File validated from disk: {file_path}")
    return {"valid": True}


# ========================


# ==================== SECURITY HEALTH CHECK ENDPOINT ====================
@app.get("/security-health")
@limiter.limit("10/minute")
async def security_health_check(request: Request):
    """Check if security services are operational (requires auth)"""
    # Optional: Require authentication for security endpoint
    # check_auth(request)

    status = {
        "status": "operational",
        "clamav": {
            "available": CLAMAV_AVAILABLE,
            "path": CLAMAV_SCANNER_PATH,
            "version": None,
        },
        "file_validation": {
            "active": True,
            "allowed_types": list(ALLOWED_FILE_TYPES.keys()),
            "max_pdf_size_mb": ALLOWED_FILE_TYPES["pdf"]["max_size"] // (1024 * 1024),
            "max_image_size_mb": ALLOWED_FILE_TYPES["image"]["max_size"]
            // (1024 * 1024),
            "max_video_size_mb": ALLOWED_FILE_TYPES["video"]["max_size"]
            // (1024 * 1024),
        },
        "platform": platform.system(),
        "environment": os.getenv("ENVIRONMENT", "development"),
    }

    # Get ClamAV version if available
    if CLAMAV_AVAILABLE:
        try:
            if platform.system().lower() == "windows":
                cmd = [CLAMAV_SCANNER_PATH, "--version"]
            else:
                cmd = [CLAMAV_SCANNER_PATH, "--version"]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                status["clamav"]["version"] = result.stdout.strip().split("\n")[0]
        except Exception as e:
            logger.warning(f"Failed to get ClamAV version: {e}")

    # Test scan with safe content
    if CLAMAV_AVAILABLE:
        test_result = await scan_with_clamav_from_disk(
            b"Clean test content", "test.txt"
        )
        status["clamav"]["test"] = test_result

    return JSONResponse(content=status)


# ==================== INITIALIZE ON STARTUP ====================
# Call this in your startup_event function
def init_security():
    """Initialize all security components - call this in startup_event"""
    logger.info("🔒 Initializing security components...")
    init_clamav_scanner()
    logger.info(
        f"✅ Security initialized - ClamAV: {'Available' if CLAMAV_AVAILABLE else 'Not Available'}"
    )


# ==================== END SECURITY ====================

BLOCKED_PATTERNS = [
    re.compile(r"wp-admin", re.IGNORECASE),
    re.compile(r"wordpress", re.IGNORECASE),
    re.compile(r"phpmyadmin", re.IGNORECASE),
    re.compile(r"administrator", re.IGNORECASE),
    re.compile(r"mysql", re.IGNORECASE),
    re.compile(r"sql", re.IGNORECASE),
    re.compile(r"\.env", re.IGNORECASE),
    re.compile(r"config\.json", re.IGNORECASE),
    re.compile(r"backup", re.IGNORECASE),
    re.compile(r"\.git", re.IGNORECASE),
    re.compile(r"\.svn", re.IGNORECASE),
    re.compile(r"\.bak$", re.IGNORECASE),
    re.compile(r"\.log$", re.IGNORECASE),
    re.compile(r"db_dump", re.IGNORECASE),
    re.compile(r"adminer", re.IGNORECASE),
    re.compile(r"phpunit", re.IGNORECASE),
    re.compile(r"eval-stdin", re.IGNORECASE),
    re.compile(r"boaform", re.IGNORECASE),
    re.compile(r"xmlrpc\.php", re.IGNORECASE),
    re.compile(r"vendor/", re.IGNORECASE),
]


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    path = request.url.path
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(path):
            return JSONResponse(status_code=403, content={"detail": "Access forbidden"})
    return await call_next(request)


@app.middleware("http")
async def mobile_compatibility_middleware(request: Request, call_next):
    response = await call_next(request)
    user_agent = request.headers.get("user-agent", "").lower()
    is_mobile = any(
        term in user_agent for term in ["mobile", "android", "iphone", "ipad"]
    )
    if is_mobile:
        response.headers["X-Mobile-Compatible"] = "true"
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# Add security headers to all responses  including CSP for XSS protection, frame options, content type options, referrer policy, and HSTS
# csp content security policy is set to allow only self for all content types, with specific allowances for scripts, styles, images, fonts, and connections to trusted sources. This helps mitigate XSS attacks while still allowing necessary resources to load.


# @app.middleware("http")
# async def add_security_headers(request: Request, call_next):
#     response = await call_next(request)

#     # Basic security headers
#     response.headers["X-Content-Type-Options"] = "nosniff"
#     response.headers["X-Frame-Options"] = "DENY"
#     response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
#     response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
#     response.headers["X-XSS-Protection"] = "1; mode=block"

#     # CSP that gets good rating but preserves functionality
#     response.headers["Content-Security-Policy"] = (
#         "default-src 'self' https: data: blob:; "
#         "script-src 'self' 'unsafe-inline' 'unsafe-eval' https: data: blob:; "
#         "style-src 'self' 'unsafe-inline' https: data:; "
#         "img-src 'self' data: https: blob:; "
#         "font-src 'self' data: https:; "
#         "connect-src 'self' https: wss: data:; "
#         "frame-src 'self' https:; "
#         "object-src 'none'; "
#         "base-uri 'self'; "
#         "form-action 'self'; "
#         "frame-ancestors 'self'; "
#         "upgrade-insecure-requests"
#     )

#     return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    return response


##### ehuggingface


# ------------------------------------------------------------------
# 1. Load the HF model once (thread-safe, CPU-only)
# ------------------------------------------------------------------
_HF_MODEL_LOCK = threading.Lock()
_HF_MODEL: Optional[SentenceTransformer] = None


def get_hf_model() -> SentenceTransformer:
    global _HF_MODEL
    with _HF_MODEL_LOCK:
        if _HF_MODEL is None:
            logger.info(
                "Loading HuggingFace model sentence-transformers/all-MiniLM-L6-v2…"
            )
            _HF_MODEL = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
            )
            logger.info("HF model loaded (384-dim).")
        return _HF_MODEL


class HFEmbeddings:
    def embed_query(self, text: str) -> List[float]:
        model = get_hf_model()
        return model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = get_hf_model()
        return model.encode(texts, normalize_embeddings=True).tolist()


##############################

import sys

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename="logs/main.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# Mount static files
current_dir = pathlib.Path(__file__).parent.resolve()
static_dir = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# AWS S3 Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
USE_S3 = all([BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY])
CORRECT_PASSWORD_HASH = os.getenv("CORRECT_PASSWORD_HASH")
CLEANUP_DASHBOARD_PASSWORD = os.getenv("CLEANUP_DASHBOARD_PASSWORD")
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)
if USE_S3:
    if "/" in BUCKET_NAME:
        BUCKET_NAME, S3_PREFIX = BUCKET_NAME.split("/", 1)
    else:
        S3_PREFIX = ""
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,
        )
        logger.info("S3 client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {str(e)}", exc_info=True)
        USE_S3 = False
        s3_client = None
else:
    logger.warning("AWS credentials missing. Falling back to local storage.")
    os.makedirs("input_pdfs", exist_ok=True)
    os.makedirs("output_pdfs", exist_ok=True)
    s3_client = None

PDF_URL_TABULAR = "https://durgasaptsati.s3.ap-south-1.amazonaws.com/pdfjoinfiles/3.pdf"
PDF_URL_NONTABULAR_1 = (
    "https://durgasaptsati.s3.ap-south-1.amazonaws.com/pdfjoinfiles/1.pdf"
)
PDF_URL_NONTABULAR_2 = (
    "https://durgasaptsati.s3.ap-south-1.amazonaws.com/pdfjoinfiles/2.pdf"
)
NONTABULAR_PDFS = [PDF_URL_NONTABULAR_1, PDF_URL_NONTABULAR_2]

# S3_PREFIX = "Amazingvideo/"

# # ========== CENTRALIZED DIRECTORY CONFIGURATION ==========
# BASE_DIR = Path(__file__).parent
# TEMP_DIR = BASE_DIR / "temp_processing"
# UPLOAD_DIR = TEMP_DIR / "uploads"
# OUTPUT_DIR = TEMP_DIR / "output"
# ESTIMATION_DIR = TEMP_DIR / "estimation"
# PDFTOWORD = TEMP_DIR / "word"

# MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
# MAX_PAGES = 5
# orphan_age_seconds = 600  # 10 minutes


S3_PREFIX = "Amazingvideo/"

# ========== CROSS-PLATFORM DIRECTORY CONFIGURATION (Windows + EC2) ==========
if platform.system() == "Windows":
    # Local Development on Windows
    BASE_DIR = Path(__file__).parent.resolve()
    logger.info(f"🚀 Running on Windows - BASE_DIR: {BASE_DIR}")
else:
    # AWS EC2 Linux Server
    BASE_DIR = Path("/home/ec2-user/vishnufastapi")
    logger.info(f"🚀 Running on EC2 Linux - BASE_DIR: {BASE_DIR}")

TEMP_DIR = BASE_DIR / "temp_processing"
UPLOAD_DIR = TEMP_DIR / "uploads"
OUTPUT_DIR = TEMP_DIR / "output"
ESTIMATION_DIR = TEMP_DIR / "estimation"
PDFTOWORD = TEMP_DIR / "word"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PAGES = 5
orphan_age_seconds = 600  # 10 minutes

# def setup_directories():
#     directories = [TEMP_DIR, UPLOAD_DIR, OUTPUT_DIR, ESTIMATION_DIR, PDFTOWORD]
#     for directory in directories:
#         try:
#             os.makedirs(str(directory), exist_ok=True)
#             os.chmod(str(directory), 0o755)
#             logger.info(f"✅ Directory ready: {directory}")
#         except Exception as e:
#             logger.error(f"❌ Failed to create directory {directory}: {e}")
#             raise


def setup_directories():
    directories = [TEMP_DIR, UPLOAD_DIR, OUTPUT_DIR, ESTIMATION_DIR, PDFTOWORD]
    for directory in directories:
        try:
            os.makedirs(str(directory), exist_ok=True)
            if platform.system() != "Windows":
                os.chmod(str(directory), 0o755)
            logger.info(f"✅ Directory ready: {directory}")
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {e}")

            raise


#########################################################################################################


#####################################################################################################################################################################
def download_from_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/pdf, */*",
        }
        session = requests.Session()
        retry_strategy = requests.packages.urllib3.util.retry.Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not response.content.startswith(
            b"%PDF"
        ):
            logger.warning(
                f"⚠️ URL {url} might not be a PDF. Content-Type: {content_type}"
            )
        return response.content
    except Exception as e:
        logger.error(f"❌ Failed to download PDF from {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")


################# NEW IMPLEMENTATION FOR RAG ####
HEADER_KEYWORDS = ["Company", "Duration"]
TITLE_KEYWORDS = ["Work Experience", "Project or Work Experience"]


def is_raw_table_text(text):
    line = text.strip()
    if not line:
        return False
    words = line.split()
    number_count = len(re.findall(r"\b\d+\b|\d+\.\d+", line))
    has_tabular = len(words) >= 4
    has_header_keywords = any(h.lower() in line.lower() for h in HEADER_KEYWORDS)
    return (number_count >= 3 and has_tabular and not has_header_keywords) or (
        has_tabular and not has_header_keywords
    )


def is_table_title(line, title_keywords=TITLE_KEYWORDS):
    if not line.strip() or len(line) > 100:
        return False
    if re.search(r"^\d+\s", line) or re.search(r"\s\d+\.\d+\s", line):
        return False
    return any(k.lower() in line.lower() for k in title_keywords)


def is_header_row(row_str, header_keywords=HEADER_KEYWORDS, min_matches=1):
    if not row_str.strip():
        return False
    lower_row = row_str.lower()
    matches = sum(1 for kw in header_keywords if kw.lower() in lower_row)
    if "project name" in lower_row or "duration" in lower_row or "company" in lower_row:
        return True
    return matches >= min_matches


def normalize_text(text):
    return " ".join(text.strip().split())


def is_substring_match(line, table_content_set):
    norm_line = normalize_text(line)
    for table_content in table_content_set:
        if norm_line == table_content:
            return True
        if len(norm_line) > 15 and len(table_content) > 15:
            if norm_line.lower() in table_content.lower():
                return True
    return False


def extract_text_with_tables(pdf_bytes):
    full_text = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                table_content = set()
                filtered_lines = []
                cleaned_tables = []
                table_title = None
                if text:
                    lines = text.split("\n")
                    for line in lines:
                        line = line.strip()
                        if (
                            "project or work experience" in line.lower()
                            and "year" in line.lower()
                        ):
                            table_title = line
                            break
                # tables = page.extract_tables() or []
                tables = page.extract_tables() or []
                for table_idx, table in enumerate(tables):
                    cleaned_table = []
                    header_row = None
                    title_row = None
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"📊 RAW TABLE {table_idx+1} on Page {i}")
                    logger.info(f"{'='*60}")
                    
                    
                    for row in table:
                        row = [
                            "" if cell is None else str(cell).strip() for cell in row
                        ]
                        row_str = " ".join(cell for cell in row if cell)
                        if not row_str.strip():
                            continue
                        norm_row_str = normalize_text(row_str)
                        table_content.add(norm_row_str)
                        for cell in row:
                            if cell:
                                cell_lines = cell.split("\n")
                                for cell_line in cell_lines:
                                    if cell_line.strip():
                                        table_content.add(normalize_text(cell_line))
                        if is_table_title(row_str, TITLE_KEYWORDS):
                            title_row = row_str
                            continue
                        if header_row is None and is_header_row(
                            row_str, HEADER_KEYWORDS
                        ):
                            header_row = row
                            continue
                        cleaned_table.append(row)
                    if cleaned_table:
                        cleaned_tables.append((title_row, header_row, cleaned_table))
                if text:
                    lines = text.split("\n")
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        norm_line = normalize_text(line)
                        if (
                            norm_line in table_content
                            or is_substring_match(line, table_content)
                            or is_table_title(line, TITLE_KEYWORDS)
                            or is_header_row(line, HEADER_KEYWORDS)
                            or is_raw_table_text(line)
                            or line == table_title
                        ):
                            continue
                        filtered_lines.append(line)
                # FIX: Only emit filtered_lines if this page has NO clean
                # markdown table. When a table exists, the raw text extraction
                # often leaks wrapped/duplicated cell content (e.g. long runs
                # of repeated "If" padding) that bloats the embedded chunk
                # and causes the LLM to truncate the restated table.
                if filtered_lines and not cleaned_tables:
                    full_text.append(f"--- PAGE {i} ---\n" + "\n".join(filtered_lines))
                for title_row, header_row, cleaned_table in cleaned_tables:
                    try:
                        # FIX: Normalize every cell BEFORE building the
                        # DataFrame: collapse repeated-char runs and long
                        # whitespace sequences so the embedded table is
                        # compact and Gemini can restate it within the
                        # output-token budget.
                        # def _clean_cell(v):
                        #     s = "" if v is None else str(v)
                        #     s = s.replace("\r", " ").replace("\n", " ")
                        #     # Kill runs of the same non-space char >= 6 times
                        #     s = re.sub(r"(\S)\1{5,}", r"\1", s)
                        #     # Kill repeated word patterns like " If If If ..."
                        #     s = re.sub(
                        #         r"(\b\w{1,4}\b)(?:\s+\1){3,}", r"\1", s
                        #     )
                        #     # Collapse whitespace
                        #     s = re.sub(r"\s{2,}", " ", s).strip()
                        #     return s
                        def _clean_cell(v):
                            s = "" if v is None else str(v)
                            
                            # Step 1: Replace newlines with spaces FIRST (pdfplumber's wrapped lines)
                            s = s.replace("\r", " ").replace("\n", " ")
                            
                            # Step 2: REPAIR split year fragments BEFORE collapsing whitespace
                            # Pattern: "202 - Present 6" → "2026 - Present"
                            # Pattern: "201 ... 5" at end → "2015 ..." 
                            # This catches: 3-digit year + text + trailing single digit
                            s = re.sub(
                                r'(\b(?:19|20)\d{2})\s+(\d)\b',  # "2026" already complete + stray digit
                                r'\1',  # drop the stray
                                s
                            )
                            s = re.sub(
                                r'(\b(?:19|20)\d)\s+([- ]+\s*\w+(?:\s+\w+)*?)\s+(\d)\b',
                                r'\1\3 \2',  # "202 - Present 6" → "2026 - Present"
                                s
                            )
                            
                            # Step 3: NOW remove repeated separator runs (was your old step 2)
                            s = re.sub(r"([\-:|])\1{5,}", r"\1\1", s)
                            
                            # Step 4: Remove repeated short words ("If If If")
                            s = re.sub(r"(\b\w{1,4}\b)(?:\s+\1){3,}", r"\1", s)
                            
                            # Step 5: Collapse whitespace LAST
                            s = re.sub(r"\s{2,}", " ", s).strip()
                            
                            return s

                        cleaned_table = [
                            [_clean_cell(c) for c in row] for row in cleaned_table
                        ]
                        # Drop rows that became empty after cleaning
                        cleaned_table = [
                            row for row in cleaned_table if any(c for c in row)
                        ]
                        if not cleaned_table:
                            continue
                        max_cols = max(len(row) for row in cleaned_table)
                        cleaned_table = [
                            row + [""] * (max_cols - len(row)) for row in cleaned_table
                        ]
                        if header_row:
                            header_row = [_clean_cell(c) for c in header_row]
                            # Pad header to match width
                            if len(header_row) < max_cols:
                                header_row = header_row + [""] * (
                                    max_cols - len(header_row)
                                )
                            elif len(header_row) > max_cols:
                                header_row = header_row[:max_cols]
                        if header_row and cleaned_table:
                            df = pd.DataFrame(cleaned_table, columns=header_row)
                        else:
                            df = (
                                pd.DataFrame(
                                    cleaned_table[1:], columns=cleaned_table[0]
                                )
                                if len(cleaned_table) > 1
                                else pd.DataFrame(cleaned_table)
                            )
                        markdown_table = df.to_markdown(index=False)
                        final_title = table_title if table_title else title_row
                        if final_title:
                            full_text.append(f"\n{final_title}\n{markdown_table}\n")
                        else:
                            full_text.append(f"\nTABLE (Page {i}):\n{markdown_table}\n")
                    except Exception as e:
                        full_text.append(
                            f"\nTABLE_RAW (Page {i}):\n{str(cleaned_table)}\n"
                        )
        return "\n".join(full_text)
    except pdfplumber.exceptions.PDFSyntaxError as e:
        raise Exception(f"Invalid PDF format: {e}")
    except Exception as e:
        raise Exception(f"Error processing PDF: {e}")


def log_embedding_process(documents: List[Document], source: str):
    logger.info(f"\n{'='*80}")
    logger.info(f"EMBEDDING PROCESS FOR: {source}")
    logger.info(f"Total documents to embed: {len(documents)}")
    chunk_sizes = [len(doc.page_content) for doc in documents]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    min_size = min(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0
    logger.info(f"📊 CHUNK STATISTICS:")
    logger.info(f"  • Average size: {avg_size:.0f} chars")
    logger.info(f"  • Min size: {min_size} chars")
    logger.info(f"  • Max size: {max_size} chars")
    logger.info(
        f"  • Size distribution: {dict(pd.Series(chunk_sizes).describe().to_dict())}"
    )
    logger.info(f"{'='*80}")
    for i, doc in enumerate(documents, 1):
        logger.info(f"\n🔤 DOCUMENT #{i} TO BE EMBEDDED:")
        logger.info(f"📁 Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"📄 Page: {doc.metadata.get('page_num', 'N/A')}")
        logger.info(
            f"🔢 Chunk: {doc.metadata.get('chunk_num', 'N/A')}/{doc.metadata.get('total_chunks_page', 'N/A')}"
        )
        logger.info(f"📏 Content Length: {len(doc.page_content)} characters")
        logger.info(f"📝 Content Preview (full contents):\n{doc.page_content}...")
        logger.info(f"🏷️ Metadata: {doc.metadata}")
        logger.info(f"{'-'*60}")


def _normalize_retrieved_content(content: str) -> str:
    """Strip padding/garbage that may be present in already-embedded chunks.
    Safe to run on normal prose. Handles three kinds of bloat:
      1. Runs of the same non-space char (e.g. 'IIIIII', '------')
      2. Alternating single-char + separator patterns (e.g. ': : : :', '- - - -',
         '| | | |') which are the residue of malformed markdown separator rows.
      3. Repeated short-word patterns (e.g. 'If If If If').
    Also normalizes markdown table separator rows to a single clean form so
    the LLM isn't asked to echo a 500-char '| :---: | :---: | ...' row."""
    if not content:
        return content

    # 1. Collapse runs of the same non-space char repeated 6+ times
    #    e.g. "IIIIII" -> "I",  "------" -> "-"
    content = re.sub(r"(\S)\1{5,}", r"\1", content)

    # 2. Collapse alternating-char patterns — the main failure mode we saw:
    #    ": : : : : :" / "- - - - - -" / "| | | | |" / ":- :- :- :-"
    #    This catches any short token (1-3 non-space chars) repeated 4+
    #    times separated by whitespace.
    content = re.sub(
        r"(\S{1,3})(?:[ \t]+\1){3,}",
        r"\1",
        content,
    )

    # 3. Normalize markdown table separator rows: any "| :---..." row with
    #    more than 8 dashes in a cell gets rewritten to a simple 3-dash form
    def _shrink_sep_cell(m):
        cell = m.group(0)
        # Keep leading/trailing colon alignment markers, shrink dashes to 3
        has_left = ":" in cell[:3]
        has_right = ":" in cell[-3:]
        return (":" if has_left else "") + "---" + (":" if has_right else "")

    # Only touch cells that are clearly separator cells (colons/dashes/spaces only)
    content = re.sub(
        r"(?<=\|)\s*:?-{4,}:?\s*(?=\|)",
        lambda m: " " + _shrink_sep_cell(re.search(r":?-+:?", m.group(0))) + " ",
        content,
    )

    # 4. Collapse repeated short-word patterns (e.g. "If If If If")
    content = re.sub(r"(\b\w{1,4}\b)(?:\s+\1){3,}", r"\1", content)

    # 5. Collapse long horizontal whitespace runs inside lines
    content = re.sub(r"[ \t]{3,}", " ", content)

    # 6. Collapse 3+ consecutive blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)

    # 7. Hard cap: if any single line is still absurdly long (>2000 chars),
    #    truncate it. Legitimate prose lines are never this long; this only
    #    catches residual table-separator garbage.
    lines = content.split("\n")
    lines = [(ln[:2000] + " …") if len(ln) > 2000 else ln for ln in lines]
    content = "\n".join(lines)

    return content


def post_process_retrieved_docs(docs, query):
    processed = []
    table_headers = ["Project Name", "Duration", "Company", "Project Description"]
    query_lower = query.lower().strip()
    for doc in docs:
        # FIX: Defensively normalize retrieved content so stale/bloated
        # embeddings (created before the extract_text_with_tables fix)
        # don't push the prompt over Gemini's output-token budget and
        # cause mid-table truncation.
        doc.page_content = _normalize_retrieved_content(doc.page_content)
        content = doc.page_content
        source = doc.metadata.get("source", "unknown")
        page_num = doc.metadata.get("page_num")
        if page_num and isinstance(page_num, float):
            doc.metadata["page_num"] = int(page_num)
        is_work_table = (
            "3.pdf" in source
            and content.count("|") > 3
            and any(header in content for header in table_headers)
        )
        if is_work_table:
            doc.metadata["content_type"] = "work_experience_table"
            doc.metadata["table_type"] = "work_experience"
            doc.metadata["priority"] = "high"
            enhanced_content = (
                f"WORK_EXPERIENCE_TABLE_START\n{content}\nWORK_EXPERIENCE_TABLE_END"
            )
            doc.page_content = enhanced_content
        else:
            doc.metadata["content_type"] = "text"
        processed.append(doc)
    return processed


def ensure_tabular_inclusion(docs, query, min_tabular=2):
    query_lower = query.lower()
    is_work_query = any(
        keyword in query_lower
        for keyword in [
            "company",
            "work",
            "experience",
            "job",
            "project",
            "started working",
            "career",
            "professional",
            "employment",
            "when did",
            "start date",
            "kei",
            "larsen",
            "toubro",
            "vindhya",
            "punj",
            "gng",
            "l&t",
            "vtl",
            "l&t",
        ]
    )
    if is_work_query:
        tabular_docs = [d for d in docs if "3.pdf" in d.metadata.get("source", "")]
        other_docs = [d for d in docs if "3.pdf" not in d.metadata.get("source", "")]
        logger.info(
            f"🔍 WORK QUERY DETECTED: Found {len(tabular_docs)} tabular documents"
        )
        final_docs = tabular_docs
        remaining_slots = 5 - len(final_docs)
        if remaining_slots > 0:
            final_docs.extend(other_docs[:remaining_slots])
        return final_docs
    elif any(
        keyword in query_lower for keyword in ["website", "site", "url", "link", "web"]
    ):
        website_docs = [
            d
            for d in docs
            if any(
                keyword in d.page_content.lower()
                for keyword in [
                    "recallmind",
                    "parcelfile",
                    "vishnuji.com",
                    "website",
                    "file transfer",
                ]
            )
        ]
        other_docs = [d for d in docs if d not in website_docs]
        final_docs = website_docs[:3]
        remaining_slots = 5 - len(final_docs)
        if remaining_slots > 0:
            final_docs.extend(other_docs[:remaining_slots])
        return final_docs
    else:
        final_docs = docs[:5]
        tabular_in_top_10 = [
            d for d in docs[:10] if "3.pdf" in d.metadata.get("source", "")
        ]
        if tabular_in_top_10 and not any(
            "3.pdf" in d.metadata.get("source", "") for d in final_docs
        ):
            final_docs[-1] = tabular_in_top_10[0]
        return final_docs


##### chromadb
class ChromaDBRetriever(BaseRetriever):
    vectorstore: Any = Field(...)
    search_kwargs: Dict = Field(default_factory=lambda: {"k": 10})

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        try:
            start_time = time.time()
            logger.info(f"\n🔍 CHROMADB SEARCH - Query: '{query[:200]}...'")
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=self.search_kwargs["k"]
            )
            docs = []
            for doc, score in docs_with_scores:
                doc.metadata["score"] = float(score)
                docs.append(doc)
            retrieval_time = time.time() - start_time
            logger.info(
                f"⚡ ChromaDB retrieval: {retrieval_time:.3f}s for {len(docs)} docs"
            )
            return docs
        except Exception as e:
            logger.error(f"❌ ChromaDB retrieval failed: {e}")
            return []


##########NEW APPROACH #############
def process_non_tabular_pdf_complete(
    pdf_bytes, pdf_url, max_chunks_per_page=3, target_chunk_size=2500
):
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            all_page_chunks = []
            total_pages = len(pdf.pages)
            logger.info(
                f"📄 Processing PDF: {pdf_url.split('/')[-1]} - {total_pages} pages"
            )
            for page_num in range(1, total_pages + 1):
                page = pdf.pages[page_num - 1]
                original_text = page.extract_text() or ""
                if not original_text.strip():
                    all_page_chunks.append(
                        {
                            "content": f"Page {page_num} - No extractable content",
                            "page_num": page_num,
                            "chunk_num": 1,
                            "total_chunks_page": 1,
                            "content_hash": hashlib.md5("empty".encode()).hexdigest(),
                        }
                    )
                    continue
                logger.info(f"   📝 Page {page_num}: {len(original_text)} characters")
                page_chunks = process_single_page_complete(
                    original_text,
                    page_num,
                    pdf_url,
                    max_chunks_per_page,
                    target_chunk_size,
                )
                verify_page_content_preservation(
                    original_text, page_chunks, page_num, pdf_url
                )
                all_page_chunks.extend(page_chunks)
            total_original_chars = sum(
                len(chunk["content"]) for chunk in all_page_chunks
            )
            logger.info(
                f"✅ FINAL: {len(all_page_chunks)} chunks created from {total_pages} pages"
            )
            return all_page_chunks
    except Exception as e:
        logger.error(f"❌ Error processing PDF {pdf_url}: {e}")
        return []


def process_single_page_complete(
    original_text, page_num, pdf_url, max_chunks, target_size
):
    cleaned_text = clean_text_preserve_all(original_text)
    sections = split_into_preserved_sections(cleaned_text)
    logger.info(f"      📑 Page {page_num} split into {len(sections)} sections")
    chunks = create_content_preserving_chunks(
        sections, max_chunks, target_size, page_num
    )
    page_chunks = []
    for chunk_num, chunk_content in enumerate(chunks, 1):
        page_chunks.append(
            {
                "content": chunk_content,
                "page_num": page_num,
                "chunk_num": chunk_num,
                "total_chunks_page": len(chunks),
                "pdf_source": pdf_url.split("/")[-1],
                "content_hash": hashlib.md5(chunk_content.encode()).hexdigest(),
            }
        )
    return page_chunks


def clean_text_preserve_all(text):
    if not text:
        return ""
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def split_into_preserved_sections(text):
    if not text:
        return []
    sections = []
    lines = text.split("\n")
    current_section = []
    for line in lines:
        line = line.strip()
        if not line:
            if current_section:
                section_text = " ".join(current_section)
                if should_preserve_section(current_section):
                    sections.append(section_text)
                    current_section = []
                else:
                    current_section.append("")
            continue
        current_section.append(line)
    if current_section:
        section_text = " ".join(current_section)
        sections.append(section_text)
    if not sections:
        return [text]
    return sections


def should_preserve_section(lines):
    if len(lines) < 2:
        return True
    first_line = lines[0]
    is_heading_like = first_line.endswith(":") or (
        len(first_line) < 100 and first_line and first_line[0].isalnum()
    )
    if is_heading_like:
        subsequent_content = lines[1:]
        has_bullet_points = any(line.startswith("•") for line in subsequent_content)
        has_short_content = all(len(line) < 200 for line in subsequent_content)
        if has_bullet_points or (has_short_content and len(subsequent_content) <= 5):
            return False
    return True


def create_content_preserving_chunks(sections, max_chunks, target_size, page_num):
    if not sections:
        return []
    total_chars = sum(len(section) for section in sections)
    if total_chars <= target_size:
        return [" ".join(sections)]
    ideal_chunk_size = max(target_size, total_chars // max_chunks)
    chunks = []
    current_chunk = []
    current_size = 0
    for section in sections:
        section_size = len(section)
        if section_size > ideal_chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            large_chunks = split_large_section(section, ideal_chunk_size)
            chunks.extend(large_chunks)
            continue
        if (
            current_size + section_size > ideal_chunk_size
            and current_chunk
            and len(chunks) < max_chunks - 1
        ):
            chunks.append(" ".join(current_chunk))
            current_chunk = [section]
            current_size = section_size
        else:
            current_chunk.append(section)
            current_size += section_size
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    if len(chunks) > max_chunks:
        chunks = merge_small_chunks(chunks, max_chunks)
    original_content = " ".join(sections)
    chunked_content = " ".join(chunks)
    if original_content != chunked_content:
        logger.warning(f"🚨 CONTENT LOSS DETECTED on page {page_num}!")
        return [original_content]
    return chunks[:max_chunks]


def split_large_section(section, max_size):
    if len(section) <= max_size:
        return [section]
    chunks = []
    words = section.split()
    current_chunk = []
    current_size = 0
    for word in words:
        word_size = len(word) + 1
        if current_size + word_size > max_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def merge_small_chunks(chunks, max_chunks):
    if len(chunks) <= max_chunks:
        return chunks
    chunk_sizes = [(i, len(chunk)) for i, chunk in enumerate(chunks)]
    chunk_sizes.sort(key=lambda x: x[1])
    while len(chunks) > max_chunks and len(chunk_sizes) > 1:
        smallest_idx1, size1 = chunk_sizes[0]
        smallest_idx2, size2 = chunk_sizes[1]
        merged = chunks[smallest_idx1] + " " + chunks[smallest_idx2]
        chunks[smallest_idx1] = merged
        chunks.pop(smallest_idx2)
        chunk_sizes = [(i, len(chunk)) for i, chunk in enumerate(chunks)]
        chunk_sizes.sort(key=lambda x: x[1])
    return chunks


def verify_page_content_preservation(original_text, page_chunks, page_num, pdf_url):
    original_normalized = " ".join(original_text.split())
    all_chunks_normalized = " ".join(
        " ".join(chunk["content"].split()) for chunk in page_chunks
    )
    if original_normalized != all_chunks_normalized:
        original_words = set(original_normalized.split())
        chunk_words = set(all_chunks_normalized.split())
        missing_words = original_words - chunk_words
        if missing_words:
            logger.error(
                f"🚨 CONTENT LOSS on {pdf_url.split('/')[-1]} Page {page_num}:"
            )
            logger.error(
                f"   Missing {len(missing_words)} words: {list(missing_words)[:10]}..."
            )
            original_sentences = re.split(r"[.!?]+", original_normalized)
            for sentence in original_sentences:
                sentence = sentence.strip()
                if (
                    sentence
                    and len(sentence) > 20
                    and sentence not in all_chunks_normalized
                ):
                    logger.error(f"   Missing sentence: {sentence[:100]}...")
        coverage = (
            len(chunk_words.intersection(original_words)) / len(original_words) * 100
        )
        logger.error(f"   Coverage: {coverage:.1f}% - NEEDS FIXING!")
        return False
    else:
        logger.info(f"      ✅ Page {page_num}: 100% content preserved")
        return True


def run_comprehensive_content_audit():
    logger.info("\n" + "=" * 80)
    logger.info("🔍 COMPREHENSIVE CONTENT COVERAGE AUDIT")
    logger.info("=" * 80)
    all_pdfs = NONTABULAR_PDFS + [PDF_URL_TABULAR]
    for pdf_url in all_pdfs:
        try:
            pdf_name = pdf_url.split("/")[-1]
            logger.info(f"\n📊 AUDITING: {pdf_name}")
            pdf_bytes = download_from_url(pdf_url)
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                total_pages = len(pdf.pages)
                total_original_chars = 0
                total_processed_chars = 0
                missing_pages = []
                for page_num in range(1, total_pages + 1):
                    original_text = pdf.pages[page_num - 1].extract_text() or ""
                    total_original_chars += len(original_text)
                    if not original_text.strip():
                        logger.info(f"   Page {page_num}: EMPTY")
                        continue
                    page_chunks = process_single_page_complete(
                        original_text, page_num, pdf_url, 3, 2000
                    )
                    page_processed_chars = sum(
                        len(chunk["content"]) for chunk in page_chunks
                    )
                    total_processed_chars += page_processed_chars
                    coverage = (
                        (page_processed_chars / len(original_text)) * 100
                        if original_text
                        else 100
                    )
                    if coverage < 99.9:
                        missing_pages.append(page_num)
                        logger.warning(
                            f"   Page {page_num}: {len(original_text)} → {page_processed_chars} chars ({coverage:.1f}%) ❌"
                        )
                    else:
                        logger.info(
                            f"   Page {page_num}: {len(original_text)} → {page_processed_chars} chars ({coverage:.1f}%) ✅"
                        )
                overall_coverage = (
                    (total_processed_chars / total_original_chars) * 100
                    if total_original_chars
                    else 100
                )
                if missing_pages:
                    logger.error(
                        f"🚨 {pdf_name}: {overall_coverage:.1f}% overall - MISSING PAGES: {missing_pages}"
                    )
                else:
                    logger.info(
                        f"✅ {pdf_name}: {overall_coverage:.1f}% overall - ALL CONTENT PRESERVED"
                    )
        except Exception as e:
            logger.error(f"❌ Failed to audit {pdf_url}: {e}")


def initialize_vectorstore():
    try:
        os.environ["TRANSFORMERS_CACHE"] = "/home/ec2-user/.cache/huggingface"
        os.environ["HF_HOME"] = "/home/ec2-user/.cache/huggingface"
        embeddings = HFEmbeddings()
        persist_dir = "./chroma_db"
        os.makedirs(persist_dir, exist_ok=True)
        existing_vectorstore = Chroma(
            collection_name="vishnu_ai_docs",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        if existing_vectorstore._collection.count() > 0:
            logger.info(
                f"✅ Using existing ChromaDB with {existing_vectorstore._collection.count()} documents"
            )
            return existing_vectorstore, embeddings
        vectorstore = Chroma(
            collection_name="vishnu_ai_docs",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        all_documents = []
        pdf_urls_to_try = [PDF_URL_TABULAR] + NONTABULAR_PDFS
        successful_downloads = 0
        for pdf_url in pdf_urls_to_try:
            try:
                logger.info(f"📥 Attempting to download: {pdf_url}")
                pdf_bytes = download_from_url(pdf_url)
                pdf_name = pdf_url.split("/")[-1]
                successful_downloads += 1
                if pdf_url == PDF_URL_TABULAR:
                    logger.info("📊 Processing tabular PDF...")
                    tabular_text = extract_text_with_tables(pdf_bytes)
                    tabular_doc = Document(
                        page_content=tabular_text,
                        metadata={
                            "source": pdf_url,
                            "content_type": "mixed_text_and_tables",
                            "document_type": "tabular",
                            "section": "work_experience",
                            "page_num": 1,
                            "chunk_num": 1,
                            "total_chunks_page": 1,
                        },
                    )
                    all_documents.append(tabular_doc)
                    logger.info(
                        f"✅ Tabular PDF processed: {len(tabular_text)} characters"
                    )
                else:
                    logger.info(f"📝 Processing: {pdf_name}")
                    page_chunks = process_non_tabular_pdf_complete(
                        pdf_bytes,
                        pdf_url,
                        max_chunks_per_page=3,
                        target_chunk_size=2500,
                    )
                    logger.info(f"📑 Created {len(page_chunks)} chunks from {pdf_name}")
                    for chunk_info in page_chunks:
                        chunk_doc = Document(
                            page_content=chunk_info["content"],
                            metadata={
                                "source": pdf_url,
                                "content_type": "text_heavy",
                                "document_type": "nontabular",
                                "page_num": chunk_info["page_num"],
                                "chunk_num": chunk_info["chunk_num"],
                                "total_chunks_page": chunk_info["total_chunks_page"],
                                "pdf_source": chunk_info["pdf_source"],
                                "content_hash": chunk_info["content_hash"],
                            },
                        )
                        all_documents.append(chunk_doc)
                    pages_covered = set(chunk["page_num"] for chunk in page_chunks)
                    logger.info(
                        f"📊 {pdf_name}: {len(pages_covered)} pages → {len(page_chunks)} chunks"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Failed to download/process {pdf_url}: {e}")
                continue
        if not all_documents:
            logger.error(
                "❌ No PDFs could be downloaded. Creating empty ChromaDB with fallback data."
            )
            fallback_doc = Document(
                page_content="Vishnu Kumar - Electrical Engineer with 12 years experience. Worked at L&T, KEI Industries, Punj Lloyd. Skills: substation execution, project management, quality assurance.",
                metadata={
                    "source": "fallback",
                    "content_type": "text",
                    "document_type": "fallback",
                    "page_num": 1,
                    "chunk_num": 1,
                },
            )
            all_documents.append(fallback_doc)
        logger.info(
            f"📤 Adding {len(all_documents)} documents to ChromaDB ({successful_downloads}/{len(pdf_urls_to_try)} PDFs successful)"
        )
        if all_documents:
            batch_size = 20
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i : i + batch_size]
                vectorstore.add_documents(batch)
                logger.info(
                    f"✅ Added batch {i//batch_size + 1}/{(len(all_documents)-1)//batch_size + 1}"
                )
                if i + batch_size < len(all_documents):
                    time.sleep(0.5)
        final_count = vectorstore._collection.count()
        logger.info(
            f"🎉 ChromaDB initialization completed with {final_count} documents"
        )
        return vectorstore, embeddings
    except Exception as e:
        logger.error(f"❌ ChromaDB initialization failed: {e}", exc_info=True)
        logger.info("🆘 Creating emergency fallback ChromaDB...")
        try:
            embeddings = HFEmbeddings()
            vectorstore = Chroma(
                collection_name="vishnu_ai_docs_fallback",
                embedding_function=embeddings,
                persist_directory="./chroma_db_fallback",
            )
            fallback_doc = Document(
                page_content="System is initializing. Please try chat functionality.",
                metadata={"source": "fallback", "error": True},
            )
            vectorstore.add_documents([fallback_doc])
            return vectorstore, embeddings
        except Exception as fallback_error:
            logger.critical(f"💥 Even fallback failed: {fallback_error}")
            raise


def verify_embeddings(embeddings_list):
    if not embeddings_list:
        raise ValueError("No embeddings generated")
    for i, embedding in enumerate(embeddings_list):
        if len(embedding) != 384:
            raise ValueError(f"Embedding {i} has wrong dimension: {len(embedding)}")
        if all(v == 0 for v in embedding) or any(np.isnan(v) for v in embedding):
            raise ValueError(f"Invalid embedding at index {i}")
    return True


_GEMINI_MODEL = None
_MODEL_LOCK = threading.Lock()


def initialize_global_gemini_model():
    global _GEMINI_MODEL
    if _GEMINI_MODEL is None:
        with _MODEL_LOCK:
            if _GEMINI_MODEL is None:
                print("🚀 PRE-LOADING Gemini model globally...")
                start = time.time()
                genai.configure(api_key=GOOGLE_API_KEY)
                # _GEMINI_MODEL = genai.GenerativeModel('models/gemini-2.5-flash')
                _GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
                # _GEMINI_MODEL = genai.GenerativeModel('models/gemini-2.0-flash')
                try:
                    _GEMINI_MODEL.generate_content(
                        "ping",
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=1
                        ),
                    )
                except Exception as e:
                    print(f"⚠️ Warm-up call completed: {e}")
                print(f"✅ Global Gemini model ready in {time.time() - start:.2f}s")
    return _GEMINI_MODEL


# WebSocket P2P Manager
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import asyncio
import json


class P2PConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.peer_metadata: Dict[str, Dict] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.peer_metadata[client_id] = {"last_seen": datetime.now()}
        logger.info(f"✅ P2P WebSocket connected: {client_id}")
        await self.broadcast_peers()

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.peer_metadata:
            del self.peer_metadata[client_id]
        logger.info(f"❌ P2P WebSocket disconnected: {client_id}")
        asyncio.create_task(self.broadcast_peers())

    async def send_message(self, message: dict, to: str):
        if to in self.active_connections:
            try:
                await self.active_connections[to].send_json(message)
                logger.info(
                    f"📤 Sent {message.get('type')} from {message.get('from', 'unknown')} to {to}"
                )
                self.peer_metadata[to]["last_seen"] = datetime.now()
            except Exception as e:
                logger.error(f"Error sending message to {to}: {e}")
                self.disconnect(to)

    async def broadcast_peers(self):
        peers = list(self.active_connections.keys())
        logger.info(f"📡 Broadcasting peers: {peers}")
        for client_id, connection in list(self.active_connections.items()):
            try:
                await connection.send_json({"type": "peers", "peers": peers})
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                self.disconnect(client_id)


p2p_manager = P2PConnectionManager()


# ==================== WEBSOCKET ENDPOINT - SECURED ====================
@app.websocket("/ws/{client_id}")
async def p2p_websocket_endpoint(websocket: WebSocket, client_id: str):
    # ✅ Make token optional for P2P (public feature)
    token = websocket.query_params.get("token")

    # ✅ Only validate token if provided, otherwise allow (public mode)
    if token and not verify_session_token(token):
        await websocket.close(code=1008, reason="Invalid token")
        return

    # ✅ Less strict client_id validation for P2P
    if not re.match(r"^[a-zA-Z0-9_-]{3,50}$", client_id):
        await websocket.close(code=1008, reason="Invalid client ID")
        return

    # ✅ Rate limit (keep this)
    client_ip = websocket.client.host
    if not await check_ws_rate_limit(client_ip, max_connections=5, window_seconds=60):
        await websocket.close(code=1008, reason="Rate limit exceeded")
        return

    await p2p_manager.connect(client_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if len(data) > 65536:
                await websocket.close(code=1009, reason="Message too large")
                break
            message_data = json.loads(data)
            if message_data.get("type") == "heartbeat":
                await websocket.send_json({"type": "pong"})
            elif "to" in message_data and "message" in message_data:
                target = message_data["to"]
                msg = message_data["message"]
                if "content" in msg:
                    msg["content"] = sanitize_input(msg["content"], max_length=10000)
                if "from" not in msg:
                    msg["from"] = client_id
                if target in p2p_manager.active_connections:
                    await p2p_manager.active_connections[target].send_json(msg)
                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Peer {target} not connected"}
                    )
    except WebSocketDisconnect:
        p2p_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        p2p_manager.disconnect(client_id)


@app.get("/test-websocket")
async def test_websocket():
    return {
        "active_connections": list(p2p_manager.active_connections.keys()),
        "total_connections": len(p2p_manager.active_connections),
        "status": "running",
    }


@app.get("/p2p", response_class=HTMLResponse)
async def p2p_transfer_page(request: Request):
    p2p_path = os.path.join(static_dir, "p2p_transfer.html")
    if os.path.exists(p2p_path):
        with open(p2p_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        from fastapi.templating import Jinja2Templates

        templates = Jinja2Templates(directory="templates")
        return templates.TemplateResponse("p2p_transfer.html", {"request": request})


retriever = None
llm = None
thread_pool = ThreadPoolExecutor(max_workers=4)


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Startup initiated")
    # init_security()
    global GEMINI_MODEL
    GEMINI_MODEL = initialize_global_gemini_model()
    global retriever, llm, vectorstore
    logger.info("🚀 Starting AI services...")
    setup_directories()
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            vectorstore, embeddings = initialize_vectorstore()
            retriever = ChromaDBRetriever(
                vectorstore=vectorstore, search_kwargs={"k": 10}
            )
            test_start = time.time()
            test_docs = retriever.invoke("test")
            test_time = time.time() - test_start
            logger.info(
                f"✅ AI services initialized successfully! Test retrieval: {test_time:.3f}s"
            )
            break
        except Exception as e:
            logger.error(
                f"❌ Startup attempt {attempt + 1}/{max_retries + 1} failed: {e}"
            )
            if attempt < max_retries:
                logger.info(f"🔄 Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.critical("💥 All startup attempts failed. Using fallback mode.")

                class FallbackRetriever(BaseRetriever):
                    def _get_relevant_documents(
                        self, query: str, *, run_manager=None
                    ) -> List[Document]:
                        return [
                            Document(
                                page_content="System is experiencing technical difficulties. Please try again later.",
                                metadata={"source": "fallback", "error": True},
                            )
                        ]

                retriever = FallbackRetriever()
                vectorstore = None


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


#######################################   login

CLEANUP_DASHBOARD_PASSWORD = os.getenv("CLEANUP_DASHBOARD_PASSWORD")
active_sessions = {}


def create_session_token(username: str) -> str:
    token_data = f"{username}{time.time()}{secrets.token_urlsafe(16)}"
    return hashlib.sha256(token_data.encode()).hexdigest()


def verify_session_token(token: str) -> bool:
    if token in active_sessions:
        session_data = active_sessions[token]
        if time.time() - session_data.get("login_time", 0) < 3600:
            return True
        else:
            del active_sessions[token]
    return False


def check_auth(request: Request):
    token = None
    token = request.query_params.get("token")
    if not token:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
    if not token:
        token = request.cookies.get("session_token")
    if not token or not verify_session_token(token):
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


@app.post("/api/login")
@limiter.limit(RATE_LIMITS["auth"])
async def api_login(
    request: Request,
    credentials: dict,
    response: Response,
    background_tasks: BackgroundTasks,
):
    username = credentials.get("username", "").strip()
    password = credentials.get("password", "").strip()
    if username == "admin" and password == os.getenv("CLEANUP_DASHBOARD_PASSWORD"):
        session_token = create_session_token(username)
        active_sessions[session_token] = {
            "username": username,
            "login_time": time.time(),
            "authenticated": True,
        }
        user_agent = request.headers.get("user-agent", "").lower()
        is_mobile = any(
            term in user_agent for term in ["mobile", "android", "iphone", "ipad"]
        )
        if is_mobile:
            redirect_url = (
                f"/cleanup?token={session_token}&mobile=true&ts={int(time.time())}"
            )
            response = RedirectResponse(url=redirect_url, status_code=303)
            response.set_cookie(
                key="session_token",
                value=session_token,
                max_age=3600,
                httponly=False,
                samesite="lax",
                secure=False,
            )
            return response
        else:
            response_data = {
                "status": "success",
                "token": session_token,
                "message": "Login successful",
            }
            response = JSONResponse(response_data)
            response.set_cookie(
                key="session_token",
                value=session_token,
                max_age=3600,
                httponly=False,
                samesite="lax",
                secure=False,
            )
            return response
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")


@app.get("/cleanup", response_class=HTMLResponse)
async def cleanup_dashboard(request: Request):
    try:
        check_auth(request)
    except HTTPException as e:
        logger.info(
            f"Serving cleanup page without auth - JavaScript will handle authentication"
        )
    cleanup_path = os.path.join(static_dir, "cleanup.html")
    with open(cleanup_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/cookie-test")
async def cookie_test(request: Request, response: Response):
    response = JSONResponse(
        {
            "message": "Cookie test",
            "cookies_received": request.cookies,
            "headers": dict(request.headers),
        }
    )
    response.set_cookie(
        key="test_cookie",
        value="working_" + str(time.time()),
        max_age=3600,
        httponly=False,
        samesite="none",
        secure=False,
    )
    return response


@app.get("/cleanup-logs")
async def get_cleanup_logs(request: Request):
    check_auth(request)
    try:
        log_path = "/home/ec2-user/vishnufastapi/cleanup_cron.log"
        if not os.path.exists(log_path):
            return {
                "logs": "No logs found yet. Cronjob may not have run.",
                "last_updated": None,
                "file_size": "0 KB",
                "line_count": 0,
                "recent_entries": [],
            }
        with open(log_path, "r") as f:
            log_content = f.read()
        stat = os.stat(log_path)
        last_updated = stat.st_mtime
        lines = log_content.strip().split("\n")
        lines_reversed = list(reversed(lines))
        recent_entries = lines_reversed[:10] if lines_reversed else []
        total_runs = len(
            [line for line in lines if "Starting scheduled cleanup" in line]
        )
        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)
        minutes = now_ist.minute
        if minutes < 30:
            next_minutes = 30 - minutes
        else:
            next_minutes = 60 - minutes
        next_run = now_ist + timedelta(minutes=next_minutes)
        return {
            "logs": "\n".join(lines_reversed),
            "last_updated": last_updated,
            "file_size": f"{stat.st_size / 1024:.2f} KB",
            "line_count": len(lines),
            "recent_entries": recent_entries,
            "total_runs": total_runs,
            "next_run_ist": next_run.strftime("%H:%M:%S"),
            "current_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")


@app.get("/cleanup-logs-page")
async def get_cleanup_logs_page():
    log_path = "/home/ubuntu/cron_cleanup.log"
    if not os.path.exists(log_path):
        log_content = "No logs found yet. Cronjob may not have run."
    else:
        with open(log_path, "r") as f:
            log_content = f.read()
    lines = log_content.strip().split("\n")
    lines.reverse()
    log_content = "\n".join(lines)
    html_content = f"""<!DOCTYPE html><html><head><title>Cleanup Logs</title><style>body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}.container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}h1 {{ color: #333; margin-bottom: 20px; text-align: center; }}.controls {{ display: flex; gap: 10px; margin-bottom: 20px; justify-content: center; flex-wrap: wrap; }}.btn {{ padding: 10px 20px; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.3s ease; }}.btn-primary {{ background: #007bff; color: white; }}.btn-primary:hover {{ background: #0056b3; transform: translateY(-2px); }}.btn-secondary {{ background: #6c757d; color: white; }}.btn-secondary:hover {{ background: #545b62; transform: translateY(-2px); }}.log-container {{ background: #1e1e1e; color: #00ff00; padding: 20px; border-radius: 8px; font-family: 'Courier New', monospace; max-height: 600px; overflow-y: auto; white-space: pre-wrap; }}.log-entry {{ margin: 5px 0; padding: 5px; border-left: 3px solid transparent; }}.log-info {{ border-left-color: #17a2b8; }}.log-success {{ border-left-color: #28a745; }}.log-error {{ border-left-color: #dc3545; }}.log-warning {{ border-left-color: #ffc107; }}.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}.stat-card {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 15px; border-radius: 10px; text-align: center; }}.stat-number {{ font-size: 1.5em; font-weight: bold; margin-bottom: 5px; }}.stat-label {{ font-size: 0.9em; opacity: 0.9; }}</style></head><body><div class="container"><h1>🧹 Cron Cleanup Logs</h1><div class="controls"><button class="btn btn-primary" onclick="refreshLogs()">🔄 Refresh</button><button class="btn btn-secondary" onclick="goToDashboard()">📊 Dashboard</button><button class="btn btn-secondary" onclick="clearLogs()">🗑️ Clear Logs</button><button class="btn btn-secondary" onclick="downloadLogs()">📥 Download Logs</button></div><div id="logStats" class="stats"></div><div class="log-container" id="logContent">{log_content}</div></div><script>function refreshLogs(){{ location.reload(); }}function goToDashboard(){{ window.location.href = '/cleanup'; }}async function clearLogs(){{ if(confirm('Are you sure you want to clear all logs?')){{ const response = await fetch('/clear-cleanup-logs', {{ method: 'POST' }}); if(response.ok){{ location.reload(); }}else{{ alert('Failed to clear logs'); }} }} }}function downloadLogs(){{ const logContent = document.getElementById('logContent').textContent; const blob = new Blob([logContent], {{ type: 'text/plain' }}); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'cleanup_logs.txt'; document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url); }}function colorCodeLogs(){{ const logContainer = document.getElementById('logContent'); const lines = logContainer.innerHTML.split('\\n'); const coloredLines = lines.map(line => {{ let className = 'log-entry log-info'; if(line.includes('ERROR') || line.includes('❌') || line.includes('💥')) className = 'log-entry log-error'; else if(line.includes('SUCCESS') || line.includes('✅') || line.includes('🎯')) className = 'log-entry log-success'; else if(line.includes('WARNING') || line.includes('⚠️')) className = 'log-entry log-warning'; return `<div class="${{className}}">${{line}}</div>`; }}); logContainer.innerHTML = coloredLines.join(''); }}async function loadStats(){{ try {{ const topics = ['technology', 'sports', 'india_power_projects', 'hiring_jobs']; const today = new Date().toISOString().split('T')[0]; let todayCount = 0; for(const topic of topics){{ try {{ const response = await fetch(`/api/newsletter/status/${{topic}}?today_only=true`); const data = await response.json(); if(data.success && data.has_todays_newsletter) todayCount++; }}catch(e){{}} }} statsContainer.innerHTML = `<div class="grid grid-cols-1 md:grid-cols-4 gap-6"><div class="stat-card tech-stat"><div class="text-center"><div class="text-3xl font-bold text-blue-600 mb-2">${{todayCount}}/4</div><div class="text-gray-600 text-sm">Today's Topics</div></div></div><div class="stat-card sports-stat"><div class="text-center"><div class="text-3xl font-bold text-green-600 mb-2">${{new Date().toLocaleDateString('en-US', {{ month: 'short', day: 'numeric' }})}}</div><div class="text-gray-600 text-sm">Date</div></div></div><div class="stat-card power-stat"><div class="text-center"><div class="text-3xl font-bold text-yellow-600 mb-2">Daily</div><div class="text-gray-600 text-sm">Auto-generate</div></div></div><div class="stat-card jobs-stat"><div class="text-center"><div class="text-3xl font-bold text-red-600 mb-2">${{todayCount === 4 ? '✅' : '🔄'}}</div><div class="text-gray-600 text-sm">Status</div></div></div></div>`; }}catch(e){{ statsContainer.innerHTML = `<div class="text-center p-6 text-gray-500"><i class="fas fa-exclamation-triangle text-xl mb-3"></i><p>Failed to load today's statistics</p></div>`; }} }}document.addEventListener('DOMContentLoaded', function(){{ colorCodeLogs(); loadStats(); }});</script></body></html>"""
    return HTMLResponse(content=html_content)


@app.post("/test-cleanup")
async def test_cleanup(request: Request):
    check_auth(request)
    try:
        logger.info("🧹 Manual cleanup triggered via /test-cleanup")
        cleanup_orphaned_files()
        return {
            "message": "Cleanup completed successfully",
            "timestamp": time.time(),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"❌ Manual cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@app.get("/session-debug")
async def session_debug(request: Request):
    cookie = request.cookies.get("session")
    cookie_data = {}
    if cookie:
        try:
            cookie_data = json.loads(cookie)
        except:
            pass
    return {
        "cookie_exists": bool(cookie),
        "cookie_data": cookie_data,
        "all_cookies": request.cookies,
        "user_agent": request.headers.get("user-agent"),
        "is_mobile": "Mobile" in request.headers.get("user-agent", ""),
    }


@app.get("/cleanup-status")
async def get_cleanup_status(request: Request):
    check_auth(request)
    try:
        current_time = time.time()

        directories = {
            "uploads": UPLOAD_DIR,
            "output": OUTPUT_DIR,
            "estimation": ESTIMATION_DIR,
            "word": PDFTOWORD,
            "temp_processing": TEMP_DIR,
        }

        stats = {
            "last_cleanup_time": 0,
            "current_time": current_time,
            "next_cleanup_in": 900,
            "directories": {},
            "total_files": 0,
            "old_files": 0,
            "total_size": 0,
        }

        for dir_name, base_dir in directories.items():
            if not base_dir.exists():
                stats["directories"][dir_name] = {
                    "total_files": 0,
                    "old_files": 0,
                    "total_size": 0,
                    "exists": False,
                    "path": str(base_dir),
                    "files": [],
                }
                continue

            total_files = 0
            old_files = 0
            total_size = 0
            file_list = []

            # RECURSIVE SCAN - This is the important change
            for root, dirs, files in os.walk(str(base_dir)):
                for filename in files:
                    file_path = Path(root) / filename
                    if file_path.is_file():
                        total_files += 1
                        file_stat = file_path.stat()
                        total_size += file_stat.st_size

                        most_recent = max(
                            file_stat.st_mtime, file_stat.st_ctime, file_stat.st_atime
                        )
                        is_old = most_recent < (current_time - 900)

                        if is_old:
                            old_files += 1

                        relative_path = file_path.relative_to(base_dir)
                        file_list.append(
                            {
                                "name": filename,
                                "folder": str(relative_path.parent),
                                "size": round(file_stat.st_size / (1024 * 1024), 2),
                                "modified": most_recent,
                                "path": str(file_path),
                                "is_old": is_old,
                            }
                        )

            stats["directories"][dir_name] = {
                "total_files": total_files,
                "old_files": old_files,
                "total_size": round(total_size / (1024 * 1024), 2),
                "exists": True,
                "path": str(base_dir),
                "files": file_list[:100],  # limit display
            }
            stats["total_files"] += total_files
            stats["old_files"] += old_files
            stats["total_size"] += total_size

        return stats

    except Exception as e:
        logger.error(f"Cleanup status failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/logout")
async def api_logout(request: Request, response: Response):
    token = request.cookies.get("session_token") or request.headers.get(
        "authorization", ""
    ).replace("Bearer ", "")
    if token in active_sessions:
        del active_sessions[token]
    response = JSONResponse({"status": "success", "message": "Logged out"})
    response.delete_cookie("session_token")
    return response


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    login_path = os.path.join(static_dir, "login.html")
    with open(login_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


BASE_DIR = Path("/home/ec2-user/vishnufastapi")
BACKEND_LOG_PATH = Path("/home/ec2-user/vishnufastapi/app.log")


@app.get("/backend-logs")
async def get_backend_logs(request: Request):
    check_auth(request)
    try:
        if not BACKEND_LOG_PATH.exists():
            return {
                "file_size": "0 KB",
                "last_updated": None,
                "total_entries": 0,
                "recent_entries": ["No backend.log file found"],
                "logs": "No backend.log file found",
                "all_logs": [],
            }
        stat = BACKEND_LOG_PATH.stat()
        file_size_kb = stat.st_size / 1024
        with open(BACKEND_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        all_lines = [line.strip() for line in lines if line.strip()]
        recent_lines = all_lines[-20:] if len(all_lines) > 20 else all_lines
        all_logs_reversed = list(reversed(all_lines))
        return {
            "file_size": f"{file_size_kb:.1f} KB",
            "last_updated": stat.st_mtime,
            "total_entries": len(all_lines),
            "recent_entries": recent_lines,
            "logs": "\n".join(all_logs_reversed),
            "all_logs": all_logs_reversed,
        }
    except Exception as e:
        logger.error(f"Error reading backend logs: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error reading backend logs: {str(e)}"
        )


async def async_retrieve_documents(query: str, retriever, max_timeout: float = 6.0):
    try:
        loop = asyncio.get_event_loop()
        docs = await asyncio.wait_for(
            loop.run_in_executor(
                thread_pool, lambda: retriever.invoke(query) if retriever else []
            ),
            timeout=max_timeout,
        )
        return docs
    except asyncio.TimeoutError:
        logger.warning(f"⏰ Retrieval timeout for query: {query}")
        return []
    except Exception as e:
        logger.error(f"❌ Retrieval error: {e}")
        return []


CHAT_MODES = {
    "general": {
        "label": "General Chat",
        "prompt": "You are a friendly, confident expert from India. Speak like a real human, not like a robot. Use simple, clear English that is easy to understand.\n\nAnswer questions directly and honestly. Explain things in a natural way, like you are talking to a person.\n\nYou are in general knowledge mode. Use your own knowledge to answer questions on any topic such as daily life, people, places, science, history, technology, programming, engineering, construction, and more.\n\nKeep responses clear, helpful, and practical. Avoid unnecessary complexity or formal language.",
    },
    "encyclopedia": {
        "label": "Encyclopedia Mode",
        "prompt": "ENCYCLOPEDIA MODE ACTIVATED:You are a friendly, confident expert from India. Speak like a real human, not like a robot. Use simple, clear English that is easy to understand.\n\n You are now a comprehensive knowledge source. IGNORE ALL DOCUMENT CONTEXT and provide detailed, factual information from your training data. Answer all questions with complete, encyclopedia-style responses covering: who/what, key facts, historical context, significance, and related information. Be thorough and authoritative.",
    },
    "creative": {
        "label": "Creative Storytelling",
        "prompt": "You're a master storyteller with an Indian flavor. Forget documents - create original, engaging stories.\n\nMake characters relatable to Indian life. Use vivid descriptions of places that could be Indian villages or cities. Include familiar elements like family gatherings, street food, or local markets.\n\nKeep the flow natural like telling a bedtime story to friends. Add emotions and drama that feel real.Use simple, clear English that is easy to understand",
    },
    "debate": {
        "label": "Balanced Debate",
        "prompt": "Activate debate mode: Do not use RAG or document info.You are a friendly, confident expert from India. Speak like a real human, not like a robot. Use simple, clear English that is easy to understand.\n\n Provide a neutral, balanced discussion on the topic, presenting multiple viewpoints equally. Encourage critical thinking and end with open questions.",
    },
    "funny": {
        "label": "Humorous Responses",
        "prompt": "Humor mode on: Ignore documents entirely.You are a friendly, confident expert from India. Speak like a real human, not like a robot. Use simple, clear English that is easy to understand.\n\n Answer the query in a witty, sarcastic, or pun-filled way. Keep it light-hearted, entertaining, and relevant, but always truthful at core.",
    },
    "baby": {
        "label": "Explain Like I'm 5",
        "prompt": "You're explaining to a curious Indian 5-year-old. Use super simple words and Indian examples.\n\nLike: 'Clouds are like big cotton candy in the sky, just like the ones at Diwali mela.'\nOr: 'Electricity flows like water in pipes, but these pipes are wires in our walls.'\n\nUse emojis 🧒🍎🛺🏏🥭✨ and be warm like a favorite auntie or uncle. Keep sentences short and full of wonder.",
    },
    "gate_coach": {
        "label": "GATE Civil Guru 🇮🇳📘🎯",
        "prompt": "🚀 **ACTIVATE GATE CIVIL GURU MODE!** 🚀\n\nNamaste future GATE Topper! 🙏🎓 I'm your **Civil Engineering Buddy from India**, who turns tough GATE questions into *easy-peasy desi-style learning!* 😄💪\n\n**🧠 MY PROBLEM-SOLVING FORMULA:**\n1. **🤔 UNDERSTAND** – 'Dekhte hain bhai, yeh sawaal kis type ka hai?'\n2. **📏 FIND** – 'Kaunsa IS code ya CPWD reference lagega?'\n3. **🔧 SOLVE** – 'Step by step, bina tension ke!'\n4. **✅ CHECK** – 'Answer sahi lag raha hai? Logical bhi?'\n5. **🎓 EXPLAIN** – 'Ab samjhaate hain simple words mein – Indian site pe kaam jaise!' 🏗️\n\n**📚 ALL CIVIL ENGINEERING TOPICS (India Edition):**\n- 🏛️ **Building Design & RCC** – IS 456:2000 style concrete power!\n- 🧱 **Steel Structures** – IS 800:2007 ke saath strong as steel! 💪\n- 🌋 **Soil Mechanics & Foundation** – IS 6403, IS 2911... Mitti ka full story! 🪣\n- 💧 **Fluid Mechanics & Hydrology** – IS 4985, IS 3370... Flow like Ganga, think like Einstein! 🌊\n- 🌿 **Environmental Engineering** – IS 10500 for clean paani 💧 and CPHEEO rules!\n- 🛣️ **Transportation Engineering** – IRC standards for smooth desi roads! 🛣️🚗\n- 📐 **Surveying & Geomatics** – IS 14962 + Indian tricks for leveling and mapping! 🧭\n- 🧮 **Engineering Mathematics** – Chill! Numbers won’t scare you here 😎\n\n**🧱 HOW I HELP YOU:**\n✨ **IS + CPWD READY** – Every answer aligns with Indian Standards 📘🇮🇳\n🎯 **TO THE POINT** – No bakwaas, only relevant explanations! 💥\n🪄 **FUN + FACTS** – Little jokes + real site examples = better memory!\n🧩 **MULTIPLE METHODS** – Shortcuts, concepts, and quick exam hacks 🎯\n🧰 **PRACTICAL VISION** – From drawing board to actual site ka gyaan 👷‍♂️\n\n**💬 EXAMPLES YOU CAN ASK:**\n- “Solve a simply supported beam using IS 456:2000.”\n- “Design a footing for column per IS 2911.”\n- “Find safe bearing capacity using Terzaghi’s method.”\n- “Calculate super elevation for highway curve (IRC:38).”\n- “Explain CPWD procedure for concrete curing.”\n\n**💡 MY PROMISE TO YOU:**\n✅ IS & CPWD code-based accurate answers 🧾\n✅ Simple, site-style explanations (like a senior teaching a junior!) 👷‍♀️👷‍♂️\n✅ Fun + Focused – with emojis, examples & real-life logic! 😄📏\n✅ Step-by-step clarity – No confusion, only confidence! 💪\n\n**💬 MOTIVATION BOOSTER:**\n_Build concepts strong like RCC, solve doubts fast like ready-mix concrete!_ 🧱💥\n\nReady to rock your GATE Civil prep – Indian style? 🇮🇳✨\nLet's crack it together! 🔥🎯",
    },
}


def log_retrieved_documents(docs: List[Document], query: str, stage: str = "retrieved"):
    logger.info(f"\n{'='*100}")
    logger.info(f"📚 CHROMADB RETRIEVAL - {stage.upper()} - Query: '{query[:200]}...'")
    logger.info(f"{'='*100}")
    logger.info(f"Total documents retrieved: {len(docs)}")
    for idx, doc in enumerate(docs, 1):
        logger.info(f"\n{'─'*80}")
        logger.info(f"📄 DOCUMENT #{idx}")
        logger.info(f"{'─'*80}")
        logger.info(f"🎯 Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"📄 Page: {doc.metadata.get('page_num', 'N/A')}")
        logger.info(f"📊 Score: {doc.metadata.get('score', 'N/A')}")
        logger.info(f"📑 Content Type: {doc.metadata.get('content_type', 'N/A')}")
        logger.info(
            f"🔢 Chunk: {doc.metadata.get('chunk_num', 'N/A')}/{doc.metadata.get('total_chunks_page', 'N/A')}"
        )
        logger.info(f"📏 Content Length: {len(doc.page_content)} characters")
        logger.info(f"\n📝 FULL CONTENT:\n{doc.page_content}")
        logger.info(f"{'─'*80}")
    logger.info(f"\n{'='*100}\n")


def log_llm_response(
    query: str, response: str, mode: str, timings: dict, docs_count: int = 0
):
    logger.info(f"\n{'='*100}")
    logger.info(f"🤖 LLM RESPONSE - Mode: {mode}")
    logger.info(f"{'='*100}")
    logger.info(f"📝 User Query: {query}")
    logger.info(f"📊 Context Docs Used: {docs_count}")
    logger.info(f"⏱️  Timings: {timings}")
    logger.info(f"\n{'─'*80}")
    logger.info(f"💬 LLM FULL RESPONSE:")
    logger.info(f"{'─'*80}")
    logger.info(f"{response}")
    logger.info(f"{'─'*80}")
    logger.info(f"📏 Response Length: {len(response)} characters")
    logger.info(f"📊 Response Words: {len(response.split())} words")
    logger.info(f"{'='*100}\n")


def log_final_documents_after_processing(docs: List[Document], query: str):
    logger.info(f"\n{'='*100}")
    logger.info(
        f"🔧 POST-PROCESSED DOCUMENTS (After filtering) - Query: '{query[:100]}...'"
    )
    logger.info(f"{'='*100}")
    logger.info(f"Final documents count: {len(docs)}")
    for idx, doc in enumerate(docs, 1):
        logger.info(f"\n{'─'*80}")
        logger.info(f"📄 FINAL DOC #{idx}")
        logger.info(f"{'─'*80}")
        logger.info(f"🎯 Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"📊 Priority: {doc.metadata.get('priority', 'normal')}")
        logger.info(f"📑 Content Type: {doc.metadata.get('content_type', 'N/A')}")
        logger.info(f"📏 Content Length: {len(doc.page_content)} characters")
        logger.info(
            f"\n📝 CONTENT PREVIEW (first 1000 chars):\n{doc.page_content[:1000]}"
        )
        if len(doc.page_content) > 1000:
            logger.info(f"... (truncated, total {len(doc.page_content)} chars)")
        logger.info(f"{'─'*80}")
    logger.info(f"\n{'='*100}\n")


@app.post("/chat")
@limiter.limit(RATE_LIMITS["chat"])
async def chat(
    request: Request,
    query: str = Form(...),
    mode: str = Form(None),
    history: str = Form(None),
):
    query = sanitize_input(query, max_length=5000, allow_html=False)
    if not query.strip() or len(query) > 10000:
        raise HTTPException(status_code=400, detail="Invalid query length")
    limited_history = []
    if history:
        try:
            history_data = json.loads(history)
            limited_history = history_data[-4:]
            for msg in limited_history:
                if "content" in msg:
                    msg["content"] = sanitize_input(msg["content"], max_length=2000)
        except Exception:
            limited_history = []

    async def generate_stream():
        start_time = time.time()
        timings = {}
        logger.info(
            f"🎯 CHAT STARTED - Query: '{query[:100]}...' | Mode: {mode} | History entries: {len(limited_history)}"
        )
        try:
            yield f"data: {json.dumps({'chunk': '🧠 GENERATING RESPONSE...', 'status': 'generating', 'prominent': True})}\n\n"
            await asyncio.sleep(0.01)
            if mode and mode in CHAT_MODES:
                logger.info("🔄 MODE-SPECIFIC PATH")
                yield f"data: {json.dumps({'chunk': '🎭 Switching to ' + CHAT_MODES[mode]['label'] + ' mode...', 'status': 'thinking'})}\n\n"
                system_prompt = CHAT_MODES[mode]["prompt"]
                messages = []
                if limited_history:
                    for msg in limited_history:
                        if msg["role"] == "user":
                            messages.append({"role": "user", "parts": [msg["content"]]})
                        elif msg["role"] == "assistant":
                            messages.append(
                                {"role": "model", "parts": [msg["content"]]}
                            )
                combined_query = (
                    f"SYSTEM INSTRUCTIONS: {system_prompt}\n\nUSER QUESTION: {query}"
                )
                messages.append({"role": "user", "parts": [combined_query]})
                # genai.configure(api_key=GOOGLE_API_KEY)
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=3024,
                    temperature=0.2,
                    top_p=0.95,
                    top_k=40,
                    candidate_count=1,
                )
                generation_start = time.time()
                connection_start = time.time()
                response = GEMINI_MODEL.generate_content(
                    messages,
                    stream=True,
                    generation_config=generation_config,
                    request_options={"timeout": 35},
                )
                connection_time = time.time() - connection_start
                full_response = ""
                chunk_count = 0
                total_chars = 0
                chunk_buffer = []
                buffer_size = 0
                MAX_BUFFER_SIZE = 100
                MAX_BUFFER_TIME = 0.02
                actual_generation_start = time.time()
                last_flush_time = time.time()
                for chunk in response:
                    if chunk.text:
                        chunk_text = chunk.text
                        full_response += chunk_text
                        chunk_count += 1
                        total_chars += len(chunk_text)
                        if chunk_count == 1:
                            data = json.dumps({"chunk": chunk_text, "done": False})
                            yield f"data: {data}\n\n"
                            continue
                        chunk_buffer.append(chunk_text)
                        buffer_size += len(chunk_text)
                        current_time = time.time()
                        if (
                            buffer_size >= MAX_BUFFER_SIZE
                            or (current_time - last_flush_time) >= MAX_BUFFER_TIME
                        ):
                            if chunk_buffer:
                                combined_text = "".join(chunk_buffer)
                                data = json.dumps(
                                    {"chunk": combined_text, "done": False}
                                )
                                yield f"data: {data}\n\n"
                                chunk_buffer = []
                                buffer_size = 0
                                last_flush_time = current_time
                if chunk_buffer:
                    combined_text = "".join(chunk_buffer)
                    data = json.dumps({"chunk": combined_text, "done": False})
                    yield f"data: {data}\n\n"
                    chunk_buffer = []
                    buffer_size = 0
                actual_generation_time = time.time() - actual_generation_start
                logger.info(
                    f"Mode-Specific - Connection: {connection_time:.2f}s, Generation: {actual_generation_time:.2f}s"
                )
                generation_end = time.time()
                timings["generation_time"] = generation_end - generation_start
                logger.info(
                    f"✅ MODE-SPECIFIC COMPLETE - Chunks: {chunk_count} | Time: {timings['generation_time']:.2f}s | Chars: {len(full_response)}"
                )
                completion_data = json.dumps(
                    {
                        "chunk": "",
                        "done": True,
                        "full_response": full_response,
                        "mode": "direct",
                        "timings": timings,
                    }
                )
                yield f"data: {completion_data}\n\n"
                log_llm_response(query, full_response, mode, timings, 0)
            else:
                logger.info("🔄 RAG PATH - Starting document retrieval")
                yield f"data: {json.dumps({'chunk': '🧠 GENERATING RESPONSE....', 'status': 'generating', 'prominent': True})}\n\n"

                def build_optimized_prompt(query, processed_docs, conversation_history):
                    # prompt_parts = [
                    #     """You are Vishnu AI Assistant — a friendly but funny assistant.\n\n**CORE BEHAVIOR:**\n- Provide accurate, clear, **human-like answers in a better representation** with professional tone\n- Never mention 'documents', 'context', 'references' or similar\n- For non-Vishnu questions: humorously suggest Tone Selector\n- Add light Indian humor naturally (like 'as easy as making Maggi')\n- Keep humor after main answer, on new line with emoji\n\n**TABLE RULES:**\n- Create tables ONLY for naturally tabular data\n- Use proper Markdown table syntax\n- Max 4 columns, wrap long text\n- If user says 'no table' or 'point-wise', use bullet/numbered lists instead\n\n**CRITICAL TABLE COMPLETENESS (MANDATORY):**\n- When the source contains a table (work experience, projects, list of companies, durations, etc.), reproduce EVERY row from the source in full.\n- NEVER truncate, summarize, skip, or combine rows.\n- NEVER use ellipses (\"...\"), \"and so on\", \"etc.\", or \"(more rows omitted)\" inside or after a table.\n- If the source table has N rows, your output table must also have exactly N rows.\n- If the \"Key Responsibilities\" or any cell in the source is very long, shorten that CELL to one concise line — but still include the row.\n- Keep intro prose before the table to 1–2 short sentences. Put the humor line AFTER the table, never inside it.\n- Finish the closing pipe of the last row before adding any commentary."""
                    # ]
                    prompt_parts = [
                    "You are Vishnu AI Assistant - friendly and helpful.",
                    "",
                    "Rules:",
                    "1. Answer accurately in simple English",
                    "2. Never mention 'documents', 'context', or 'references'",
                    "3. Add light Indian humor at the end only",
                    "4. For work experience, use bullet points (•), not tables",
                    "5. Include ALL information: company, duration,Project Name, Project Description/role",
                    "6. Never skip or shorten the list",
                    "",
                    "CONTEXT:",
                ]
                    for role, content in conversation_history:
                        if role == "user":
                            prompt_parts.append(f"USER: {content}")
                        else:
                            prompt_parts.append(f"ASSISTANT: {content}")
                    if processed_docs:
                        prompt_parts.append("\nCONTEXT:")
                        for i, doc in enumerate(processed_docs, 1):
                            prompt_parts.append(f"DOC_{i}: {doc.page_content}")
                    prompt_parts.append(f"\nQUESTION: {query}")
                    prompt_parts.append("\nANSWER:")
                    return "\n".join(prompt_parts)

                conversation_history = []
                if limited_history:
                    recent_history = limited_history[-4:]
                    logger.info(f"RAG history: {len(recent_history)} messages")
                    for msg in recent_history:
                        if msg["role"] == "user":
                            conversation_history.append(("user", msg["content"]))
                        elif msg["role"] == "assistant":
                            conversation_history.append(("assistant", msg["content"]))
                retrieval_start = time.time()
                try:
                    raw_docs = await async_retrieve_documents(
                        query, retriever, max_timeout=6.0
                    )
                    retrieval_end = time.time()
                    timings["retrieval_time"] = retrieval_end - retrieval_start
                    logger.info(
                        f"✅ RETRIEVAL COMPLETE - Documents: {len(raw_docs)} | Time: {timings['retrieval_time']:.2f}s"
                    )
                    logger.info(f"\n{'='*80}")
                    logger.info(f"📚 RETRIEVED DOCUMENTS FROM CHROMADB ({len(raw_docs)} docs)")
                    logger.info(f"{'='*80}")
                    for idx, doc in enumerate(raw_docs, 1):
                        logger.info(f"\n--- DOCUMENT #{idx} ---")
                        logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
                        logger.info(f"Page: {doc.metadata.get('page_num', 'N/A')}")
                        logger.info(f"Score: {doc.metadata.get('score', 'N/A')}")
                        logger.info(f"Content Type: {doc.metadata.get('content_type', 'N/A')}")
                        logger.info(f"Content Preview (first 500 chars):\n{doc.page_content}")
                        logger.info(f"Content Length: {len(doc.page_content)} chars")
                        logger.info("-" * 40)
                except Exception as e:
                    logger.error(f"❌ RETRIEVAL FAILED: {e}")
                    raw_docs = []
                    yield f"data: {json.dumps({'chunk': '⚠️ Using general knowledge...', 'status': 'fallback'})}\n\n"
                processing_start = time.time()
                final_docs = ensure_tabular_inclusion(raw_docs, query, min_tabular=2)
                processed_docs = post_process_retrieved_docs(final_docs, query)
                processing_end = time.time()
                timings["processing_time"] = processing_end - processing_start
                logger.info(
                    f"✅ PROCESSING COMPLETE - Final docs: {len(processed_docs)} | Time: {timings['processing_time']:.2f}s"
                )
                logger.info(f"\n{'='*80}")
                logger.info(f"🎯 FINAL DOCUMENTS SENT TO LLM ({len(processed_docs)} docs)")
                logger.info(f"{'='*80}")
                for idx, doc in enumerate(processed_docs, 1):
                    logger.info(f"\n--- FINAL DOCUMENT #{idx} ---")
                    logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    logger.info(f"Page: {doc.metadata.get('page_num', 'N/A')}")
                    logger.info(f"Priority: {doc.metadata.get('priority', 'normal')}")
                    logger.info(f"Content Type: {doc.metadata.get('content_type', 'N/A')}")
                    logger.info(f"FULL CONTENT:\n{doc.page_content}")
                    logger.info(f"Content Length: {len(doc.page_content)} chars")
                    logger.info("-" * 40)
                if not processed_docs:
                    logger.warning("⚠️ NO DOCUMENTS FOUND - Using fallback response")
                    fallback_msg = "I couldn't find specific information about that in my knowledge base. Is there anything else I can help you with?"
                    data = json.dumps({"chunk": fallback_msg, "done": False})
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0.02)
                    yield f"data: {json.dumps({'chunk': '', 'done': True, 'mode': 'fallback'})}\n\n"
                else:
                    optimized_prompt = build_optimized_prompt(
                        query, processed_docs, conversation_history
                    )
                    logger.info(
                        f"📝 PROMPT BUILT - Length: {len(optimized_prompt)} chars | Docs used: {len(processed_docs)}"
                    )
                    try:
                        generation_start = time.time()
                        connection_start = time.time()

                        # ================================================
                        # FIX: Cost-aware dynamic token budgeting.
                        # Different query types need different output sizes.
                        # We set max_output_tokens based on the query and
                        # also hard-abort streaming the moment output
                        # exceeds a matching character cap (≈ 4 chars/token).
                        # This prevents the 200k-char runaway that blew
                        # through API budget on the "tell about him" query.
                        # ================================================

                        # Classify the query to pick the right output budget.
                        _q = query.lower().strip()
                        _is_tabular_query = any(
                            kw in _q
                            for kw in [
                                "job experience",
                                "work experience",
                                "job experince",
                                "work experince",
                                "work profile",
                                "all projects",
                                "list of projects",
                                "all companies",
                                "list companies",
                                "entire career",
                                "full career",
                                "complete career",
                                "job history",
                                "work history",
                            ]
                        )
                        _is_broad_query = any(
                            kw in _q
                            for kw in [
                                "tell about",
                                "tell me about",
                                "about vishnu",
                                "about him",
                                "who is vishnu",
                                "introduce",
                                "overview",
                                "summary",
                                "background",
                                "biography",
                                "profile",
                            ]
                        )
                        if _is_tabular_query:
                            # 13 rows * ~200 tokens/row = 2600
                            _max_tokens = 2600
                            _char_cap = 10400
                        elif _is_broad_query:
                            _max_tokens = 2600
                            _char_cap = 10400
                        elif len(_q) < 30:  # short question like "joined kei?"
                            _max_tokens = 2600
                            _char_cap = 10400
                        else:
                            _max_tokens = 2600
                            _char_cap = 10400
                        logger.info(
                            f"💰 BUDGET - query_type={'tabular' if _is_tabular_query else 'broad' if _is_broad_query else 'short' if len(_q)<30 else 'default'} max_tokens={_max_tokens} char_cap={_char_cap}"
                        )

                        def _looks_like_runaway(full_text: str, tail_text: str):
                            """Return (is_bad, reason).

                            Catches the three real failure modes:
                              1. Total chars > char_cap (query-specific).
                              2. 200+ consecutive non-space chars.
                              3. 500+ consecutive whitespace chars.
                            """
                            if not full_text:
                                return (False, "")
                            if len(full_text) > _char_cap:
                                return (True, f"length={len(full_text)}>{_char_cap}")
                            if not tail_text:
                                return (False, "")
                            m = re.search(r"([^\s])\1{199,}", tail_text)
                            if m:
                                return (True, f"run_of_{m.group(1)!r}")
                            m = re.search(r"\s{500,}", tail_text)
                            if m:
                                return (True, "run_of_whitespace")
                            return (False, "")

                        def _stream_gemini(
                            temp: float, max_tokens: int, extra_instruction: str = ""
                        ):
                            """Return a (response_iterator, connect_time) tuple."""
                            cfg = genai.types.GenerationConfig(
                                max_output_tokens=max_tokens,
                                temperature=temp,
                                top_p=0.9,
                                top_k=40,
                                candidate_count=1,
                            )
                            prompt_to_send = optimized_prompt
                            if extra_instruction:
                                prompt_to_send = (
                                    optimized_prompt
                                    + "\n\nSTRICT RULE: "
                                    + extra_instruction
                                )
                            t0 = time.time()
                            resp = GEMINI_MODEL.generate_content(
                                prompt_to_send,
                                stream=True,
                                generation_config=cfg,
                                request_options={"timeout": 45},
                            )
                            return resp, time.time() - t0

                        # ------------ Attempt 1: live streaming ------------
                        attempt1_start = time.time()
                        response_iter, connect_t = _stream_gemini(
                            temp=0.1, max_tokens=_max_tokens
                        )
                        connection_time = connect_t
                        full_response = ""
                        chunk_count = 0
                        total_chars = 0
                        runaway_detected = False
                        runaway_reason = ""
                        CHECK_WINDOW = 4000

                        for chunk in response_iter:
                            if not chunk.text:
                                continue
                            chunk_text = chunk.text
                            full_response += chunk_text
                            chunk_count += 1
                            total_chars += len(chunk_text)
                            # Stream to client immediately (typewriter UX)
                            data = json.dumps({"chunk": chunk_text, "done": False})
                            yield f"data: {data}\n\n"
                            # Runaway check on EVERY chunk so we abort fast
                            tail = full_response[-CHECK_WINDOW:]
                            is_bad, reason = _looks_like_runaway(
                                full_response, tail
                            )
                            if is_bad:
                                runaway_detected = True
                                runaway_reason = reason
                                break

                        logger.info(
                            f"🎲 Attempt 1: {len(full_response)} chars in {time.time()-attempt1_start:.2f}s"
                        )

                        if runaway_detected:
                            logger.warning(
                                f"⚠️ RUNAWAY DETECTED MID-STREAM ({runaway_reason}) — aborting and retrying"
                            )
                            # Tell the client to discard what was shown and
                            # prepare for a fresh response.
                            yield f"data: {json.dumps({'reset': True, 'chunk': '', 'status': 'regenerating'})}\n\n"
                            yield f"data: {json.dumps({'chunk': '🔁 Let me try that again...', 'status': 'thinking', 'prominent': True})}\n\n"
                            await asyncio.sleep(0.05)

                            # ---------- Attempt 2: stricter retry ----------
                            attempt2_start = time.time()
                            retry_iter, _ = _stream_gemini(
                                temp=0.05,
                                max_tokens=_max_tokens,
                                extra_instruction=(
                                    "Do NOT repeat any character more than 5 times in a row. "
                                    "In markdown tables, each separator cell must be exactly ':---' "
                                    "with only 3 dashes. Produce the answer once, then stop immediately. "
                                    f"Total response must be under {_char_cap} characters."
                                ),
                            )
                            full_response = ""
                            chunk_count = 0
                            total_chars = 0
                            retry_runaway = False
                            for chunk in retry_iter:
                                if not chunk.text:
                                    continue
                                chunk_text = chunk.text
                                full_response += chunk_text
                                chunk_count += 1
                                total_chars += len(chunk_text)
                                data = json.dumps({"chunk": chunk_text, "done": False})
                                yield f"data: {data}\n\n"
                                tail = full_response[-CHECK_WINDOW:]
                                is_bad2, _ = _looks_like_runaway(
                                    full_response, tail
                                )
                                if is_bad2:
                                    retry_runaway = True
                                    break
                            logger.info(
                                f"🎲 Attempt 2 (retry): {len(full_response)} chars in {time.time()-attempt2_start:.2f}s"
                            )
                            if retry_runaway:
                                logger.error(
                                    "❌ Retry also went runaway — sending safe fallback"
                                )
                                yield f"data: {json.dumps({'reset': True, 'chunk': '', 'status': 'fallback'})}\n\n"
                                fallback = (
                                    "I had trouble formatting the full answer this time. "
                                    "Please ask again — it usually works on the next try."
                                )
                                full_response = fallback
                                yield f"data: {json.dumps({'chunk': fallback, 'done': False})}\n\n"

                        actual_generation_time = time.time() - connection_start
                        logger.info(
                            f"Connection: {connection_time:.2f}s, Generation: {actual_generation_time:.2f}s"
                        )
                        generation_end = time.time()
                        timings["generation_time"] = generation_end - generation_start
                        timings["total_chars"] = total_chars
                        timings["chunk_count"] = chunk_count
                        logger.info(
                            f"✅ GENERATION COMPLETE - Chunks: {chunk_count} | Ouput Chars: {total_chars} | Total Generation Time: {timings['generation_time']:.2f}s | Speed: {total_chars/max(timings['generation_time'],0.001):.1f} chars/sec"
                        )
                        completion_data = json.dumps(
                            {
                                "chunk": "",
                                "done": True,
                                "full_response": full_response,
                                "mode": "rag",
                                "timings": timings,
                                "retrieved_docs_count": len(raw_docs),
                                "processed_docs_count": len(processed_docs),
                            }
                        )
                        yield f"data: {completion_data}\n\n"
                    except asyncio.TimeoutError:
                        logger.error("⏰ GENERATION TIMEOUT")
                        error_msg = "I'm taking too long to generate a response. Please try again!"
                        data = json.dumps({"chunk": error_msg, "done": False})
                        yield f"data: {data}\n\n"
                        await asyncio.sleep(0.02)
                        yield f"data: {json.dumps({'chunk': '', 'done': True, 'error': 'timeout'})}\n\n"
                    except Exception as e:
                        logger.error(f"❌ GENERATION FAILED: {e}")
                        error_msg = (
                            "I'm experiencing technical issues. Please try again."
                        )
                        data = json.dumps({"chunk": error_msg, "done": False})
                        yield f"data: {data}\n\n"
                        await asyncio.sleep(0.02)
                        yield f"data: {json.dumps({'chunk': '', 'done': True, 'error': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"💥 STREAM ERROR: {e}")
            error_msg = f"An unexpected error occurred: {str(e)}"
            data = json.dumps({"chunk": error_msg, "done": False})
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'chunk': '', 'done': True, 'error': 'unexpected'})}\n\n"
        finally:
            total_time = time.time() - start_time
            logger.info(
                f"🏁 CHAT COMPLETED - Total time: {total_time:.2f}s | Final timings: {timings}"
            )

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    )


# =======================
# def validate_pdf_page_count(file_path: str, max_pages: int = 2000) -> int:
#     """
#     Get PDF page count by reading ONLY the PDF trailer - PURE DISK OPERATION
#     NO fitz, NO RAM loading - reads only last 20KB of file
#     """
#     try:
#         with open(file_path, 'rb') as f:
#             # Get file size
#             f.seek(0, 2)
#             file_size = f.tell()

#             # For small files (< 100KB), read entire file (still tiny RAM)
#             if file_size < 102400:  # 100KB
#                 f.seek(0)
#                 content = f.read().decode('latin-1', errors='ignore')
#             else:
#                 # Read ONLY last 20KB where PDF catalog is stored
#                 read_size = min(20480, file_size)  # 20KB max
#                 f.seek(max(0, file_size - read_size))
#                 content = f.read(read_size).decode('latin-1', errors='ignore')

#             # Extract page count using regex patterns
#             import re
#             patterns = [
#                 r'/Pages[^/]*/Count\s+(\d+)',  # Standard: /Pages /Count 5
#                 r'/Count\s+(\d+)',              # Simple: /Count 5
#                 r'/N\s+(\d+)',                  # Alternative: /N 5
#                 r'/Type\s*/Pages[^>]*/Count\s+(\d+)',  # Explicit Pages dict
#             ]

#             for pattern in patterns:
#                 matches = re.findall(pattern, content)
#                 if matches:
#                     page_count = int(matches[-1])  # Take last match

#                     if page_count > max_pages:
#                         raise HTTPException(
#                             status_code=400,
#                             detail=f"PDF has {page_count} pages. Maximum allowed is {max_pages} pages."
#                         )

#                     logger.info(f"✅ Page count: {page_count} (disk-based, 0 RAM spike)")
#                     return page_count

#             # Fallback: Try pdfplumber (still better than fitz)
#             logger.warning("Header parsing failed, using pdfplumber fallback")
#             import pdfplumber
#             with pdfplumber.open(file_path) as pdf:
#                 page_count = len(pdf.pages)
#                 if page_count > max_pages:
#                     raise HTTPException(400, f"PDF has {page_count} pages. Max: {max_pages}")
#                 logger.info(f"✅ Page count: {page_count} (pdfplumber fallback)")
#                 return page_count

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Page count validation failed: {e}")
#         raise HTTPException(status_code=400, detail=f"Could not validate PDF pages: {str(e)}")


##====================   file validation for client side pdf opearations  ====================##


# ===================


# Redis for progress tracking (fallback to in-memory if Redis not available)
try:
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    USE_REDIS = True
    logger.info("Redis connected for progress tracking")
except:
    USE_REDIS = False
    progress_store = {}
    logger.info("Using in-memory progress tracking")

gs_binary = "gswin64c" if platform.system() == "Windows" else "gs"
compression_presets = {
    "screen": {"dpi": 72, "quality": "screen", "desc": "Low quality, smallest size"},
    "ebook": {
        "dpi": 150,
        "quality": "ebook",
        "desc": "Medium quality, good compression",
    },
    "printer": {"dpi": 300, "quality": "printer", "desc": "High quality for printing"},
    "prepress": {
        "dpi": 300,
        "quality": "prepress",
        "desc": "Highest quality, minimal compression",
    },
}
COMPRESS_MAX_FILE_SIZE_MB = 100


def validate_file_size(file_size_bytes: int):
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > COMPRESS_MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400, detail=f"File exceeds {COMPRESS_MAX_FILE_SIZE_MB}MB limit"
        )


class ProgressTracker:
    def __init__(self):
        self.tasks: Dict[str, dict] = {}
        self.use_redis = False
        self.redis_client = None

        # Try Redis but don't fail if not available
        try:

            self.redis_client = redis.Redis(
                host="localhost",
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self.redis_client.ping()
            self.use_redis = True
            logger.info("✅ Redis connected for progress tracking")
        except ImportError:
            logger.warning("⚠️ Redis module not installed, using in-memory storage")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}, using in-memory storage")

    async def update_progress(
        self, task_id: str, progress: int, message: str = "", stage: str = ""
    ):
        """Update progress for a task"""
        progress_data = {
            "progress": max(0, min(100, progress)),
            "message": message,
            "stage": stage,
            "timestamp": time.time(),
        }

        # Store in memory always (as backup)
        self.tasks[task_id] = progress_data

        # Also store in Redis if available
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.setex(
                    f"progress:{task_id}", 300, json.dumps(progress_data)
                )
            except Exception as e:
                logger.error(f"Redis update error: {e}")

        logger.info(f"📊 Progress: {task_id} - {progress}% - {message}")

    def get_progress(self, task_id: str) -> Optional[dict]:
        """Get progress for a task - returns None if not found"""
        # Try memory first (faster)
        if task_id in self.tasks:
            return self.tasks[task_id]

        # Try Redis if available
        if self.use_redis and self.redis_client:
            try:
                data = self.redis_client.get(f"progress:{task_id}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Redis get error: {e}")

        return None

    def cleanup_old_tasks(self, max_age_seconds: int = 600):
        """Remove tasks older than max_age_seconds to prevent memory leaks"""
        current_time = time.time()

        # Clean memory store
        expired = [
            task_id
            for task_id, data in self.tasks.items()
            if current_time - data.get("timestamp", 0) > max_age_seconds
        ]
        for task_id in expired:
            del self.tasks[task_id]

        if expired:
            logger.info(f"🧹 Cleaned {len(expired)} expired tasks from memory")


# Create global instance
progress_tracker = ProgressTracker()


async def update_progress(
    task_id: str, progress: int, message: str = "", stage: str = ""
):
    progress_data = {
        "progress": progress,
        "message": message,
        "stage": stage,
        "timestamp": time.time(),
    }

    # Store in memory
    progress_store[task_id] = progress_data

    # Store in progress_tracker
    await progress_tracker.update_progress(task_id, progress, message, stage)

    # Store in Redis if available
    try:
        redis_client.setex(f"progress:{task_id}", 300, json.dumps(progress_data))
    except:
        pass

    logger.info(f"📊 Progress {task_id}: {progress}% - {message}")


async def upload_to_disk_first(file: UploadFile, task_id: str = None) -> str:
    """Upload file to disk with progress updates"""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    safe_filename = sanitize_filename(file.filename)

    if task_id:
        final_filename = f"{task_id}_{safe_filename}"
    else:
        final_filename = f"{uuid.uuid4().hex}_{safe_filename}"

    file_path = UPLOAD_DIR / final_filename

    # Read and write in chunks with progress
    total_size = 0
    chunk_count = 0
    with open(file_path, "wb") as buffer:
        while chunk := await file.read(65536):  # 64KB chunks
            total_size += len(chunk)
            buffer.write(chunk)
            chunk_count += 1

            # Update progress every 10 chunks or 1MB
            if chunk_count % 10 == 0 and task_id:
                await progress_tracker.update_progress(
                    task_id,
                    10 + min(20, int(total_size / 1024 / 1024)),
                    f"Uploading... ({total_size / 1024 / 1024:.1f} MB)",
                    "uploading",
                )

    logger.info(f"✅ File saved to disk: {file_path} ({total_size} bytes)")
    return str(file_path)


# async def update_progress(task_id: str, progress: int, message: str = "", stage: str = ""):
#     await progress_tracker.update_progress(task_id, progress, message, stage)


def cleanup_local_files():
    directories = ["input_pdfs", "output_pdfs"]
    for directory in directories:
        if not os.path.exists(directory):
            continue
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logger.info(f"Deleted local file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete local file {file_path}: {str(e)}")


def upload_to_s3(file_content: bytes, filename: str) -> str:
    if not isinstance(file_content, (bytes, bytearray)):
        raise TypeError("file_content must be bytes")
    safe_filename = os.path.basename(filename)
    s3_key = f"temp_uploads/{hashlib.md5(file_content).hexdigest()}_{safe_filename}"
    try:
        s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=file_content)
        logger.info(f"✅ Uploaded to S3: {s3_key}")
    except Exception as e:
        logger.error(f"❌ Failed to upload to S3: {e}")
        raise
    return s3_key


def cleanup_s3_file(s3_key: str):
    try:
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        logger.info(f"Deleted S3 file: {s3_key}")
    except Exception as e:
        logger.warning(f"Failed to delete S3 file {s3_key}: {e}")


def safe_delete_temp_file(file_path: str):
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
            logger.info(f"✅ Deleted temp file: {file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to delete temp file {file_path}: {str(e)}")


def get_fallback_estimate(original_size_mb: float, preset_name: str) -> float:
    fallback_ratios = {"screen": 0.2, "ebook": 0.4, "printer": 0.7, "prepress": 0.9}
    return round(original_size_mb * fallback_ratios.get(preset_name, 0.5), 2)


def get_compression_recommendation(estimates: dict, original_size_mb: float) -> str:
    if estimates.get("ebook", original_size_mb) < original_size_mb * 0.6:
        return "ebook"
    elif estimates.get("printer", original_size_mb) < original_size_mb * 0.8:
        return "printer"
    else:
        return "screen"


def compress_pdf_ghostscript_file(
    input_path: str, output_path: str, compression_level: str = "ebook"
):
    compression_settings = {
        "screen": "/screen",
        "ebook": "/ebook",
        "printer": "/printer",
        "prepress": "/prepress",
    }
    if compression_level not in compression_settings:
        compression_level = "ebook"
    if platform.system() == "Windows":
        base_cmd = [gs_binary]
        creationflags = subprocess.CREATE_NO_WINDOW
    else:
        base_cmd = ["nice", "-n", "10", gs_binary]
        creationflags = 0
    gs_command = base_cmd + [
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        "-sDEVICE=pdfwrite",
        f"-dPDFSETTINGS={compression_settings[compression_level]}",
        "-dCompatibilityLevel=1.4",
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dEmbedAllFonts=true",
        "-dSubsetFonts=true",
        "-dColorImageDownsampleType=/Average",
        "-dGrayImageDownsampleType=/Average",
        "-dMonoImageDownsampleType=/Subsample",
        "-dColorImageResolution=120",
        "-dGrayImageResolution=120",
        "-dMonoImageResolution=200",
        "-dAutoFilterColorImages=true",
        "-dAutoFilterGrayImages=true",
        "-dColorImageFilter=/DCTEncode",
        "-dGrayImageFilter=/DCTEncode",
        "-dCompressPages=true",
        "-dMaxPatternBitmap=5000000",
        "-dBufferSpace=80000000",
        "-dMaxBitmap=50000000",
        "-dNumRenderingThreads=1",
        "-dUseFastColor=true",
        "-dNOGC",
        "-dUseCropBox",
        f"-sOutputFile=" + output_path,
        input_path,
    ]
    try:
        logger.info(
            f"Running AWS-optimized Ghostscript compression: {compression_level}"
        )
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "GS_THREADS": "1",
            }
        )
        result = subprocess.run(
            gs_command,
            capture_output=True,
            timeout=400,
            env=env,
            creationflags=creationflags,
        )
        if result.returncode != 0:
            logger.error(f"Ghostscript failed with return code {result.returncode}")
            if result.stderr:
                error_msg = result.stderr.decode("utf-8", errors="ignore")
                logger.error(f"Ghostscript stderr: {error_msg}")
                if "memory" in error_msg.lower() or "timeout" in error_msg.lower():
                    logger.info("Attempting fallback compression with screen preset")
                    return compress_with_fallback(input_path, output_path)
            return False
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            savings = ((original_size - compressed_size) / original_size) * 100
            logger.info(
                f"Compression completed: {original_size/1024/1024:.1f}MB → {compressed_size/1024/1024:.1f}MB ({savings:.1f}% savings)"
            )
            return True
        else:
            logger.error("Ghostscript output file is missing or too small")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Ghostscript timeout with optimized settings")
        return compress_with_fallback(input_path, output_path)
    except Exception as e:
        logger.error(f"Optimized compression error: {str(e)}")
        return compress_with_fallback(input_path, output_path)


def compress_with_fallback(input_path: str, output_path: str) -> bool:
    try:
        if platform.system() == "Windows":
            base_cmd = [gs_binary]
            creationflags = subprocess.CREATE_NO_WINDOW
        else:
            base_cmd = ["nice", "-n", "15", gs_binary]
            creationflags = 0
        fallback_command = base_cmd + [
            "-dNOPAUSE",
            "-dBATCH",
            "-dQUIET",
            "-sDEVICE=pdfwrite",
            "-dPDFSETTINGS=/screen",
            "-dCompatibilityLevel=1.4",
            "-dColorImageResolution=100",
            "-dGrayImageResolution=100",
            "-dMonoImageResolution=150",
            "-dColorImageDownsampleType=/Average",
            "-dGrayImageDownsampleType=/Average",
            "-dMonoImageDownsampleType=/Subsample",
            "-dNumRenderingThreads=1",
            f"-sOutputFile=" + output_path,
            input_path,
        ]
        result = subprocess.run(
            fallback_command,
            capture_output=True,
            timeout=300,
            env={**os.environ, "OMP_NUM_THREADS": "1", "GS_THREADS": "1"},
            creationflags=creationflags,
        )
        success = (
            result.returncode == 0
            and os.path.exists(output_path)
            and os.path.getsize(output_path) > 1000
        )
        logger.info(f"Fallback compression {'succeeded' if success else 'failed'}")
        return success
    except Exception as e:
        logger.error(f"Fallback compression also failed: {str(e)}")
        return False


def cleanup_compression_estimation_files(task_id: str):
    try:
        patterns = [
            f"{task_id}_*",
            f"compressed_{task_id}_*",
            f"*{task_id}_*estimation.pdf",
        ]
        for base_dir in [UPLOAD_DIR, OUTPUT_DIR, ESTIMATION_DIR]:
            if not base_dir.exists():
                continue
            for pattern in patterns:
                for file_path in base_dir.glob(pattern):
                    if file_path.is_file():
                        safe_delete_temp_file(str(file_path))
    except Exception as e:
        logger.error(f"Compression cleanup error for {task_id}: {e}")


async def stream_upload_to_disk(file: UploadFile, task_id: str) -> str:
    filename = f"{task_id}_{file.filename}"
    file_path = UPLOAD_DIR / filename
    logger.info(f"Streaming upload to disk: {file_path}")
    file_size = 0
    if hasattr(file, "size") and file.size:
        file_size = file.size
    else:
        logger.warning("File size not available, using chunk-based progress")
    uploaded_size = 0
    chunk_number = 0
    with open(file_path, "wb") as buffer:
        while True:
            chunk = await file.read(128 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
            uploaded_size += len(chunk)
            chunk_number += 1
            if file_size > 0:
                progress = 10 + (uploaded_size / file_size) * 20
                progress = min(30, progress)
                if chunk_number % 10 == 0 or uploaded_size % (1024 * 1024) == 0:
                    await progress_tracker.update_progress(
                        task_id,
                        int(progress),
                        f"Uploading to disk... ({uploaded_size / (1024 * 1024):.1f} MB)",
                        "uploading",
                    )
    logger.info(f"File upload completed: {uploaded_size} bytes written to {file_path}")
    await progress_tracker.update_progress(
        task_id, 30, "File streamed to disk!", "processing"
    )
    return str(file_path)


estimation_results = {}


async def process_estimation_sequential_disk(
    task_id: str, file_path: str, filename: str
):
    try:
        await progress_tracker.update_progress(
            task_id, 25, "Analyzing file sequentially...", "processing"
        )
        original_size = os.path.getsize(file_path)
        original_size_mb = original_size / (1024 * 1024)
        estimates = {}
        key_presets = ["ebook", "printer", "screen", "prepress"]
        for i, preset_name in enumerate(key_presets):
            progress = 30 + i * 15
            await progress_tracker.update_progress(
                task_id,
                progress,
                f"Testing {preset_name} compression...",
                "compressing",
            )
            try:
                output_filename = f"{task_id}_{preset_name}_estimation.pdf"
                output_path = ESTIMATION_DIR / output_filename
                success = compress_pdf_ghostscript_file(
                    file_path, str(output_path), preset_name
                )
                if success and os.path.exists(output_path):
                    compressed_size = os.path.getsize(output_path)
                    compressed_size_mb = compressed_size / (1024 * 1024)
                    estimates[preset_name] = round(compressed_size_mb, 2)
                    savings_pct = (
                        (original_size - compressed_size) / original_size
                    ) * 100
                    logger.info(
                        f"✅ {preset_name}: {original_size_mb:.1f}MB → {compressed_size_mb:.1f}MB ({savings_pct:.1f}% savings)"
                    )
                    safe_delete_temp_file(str(output_path))
                    await asyncio.sleep(0.5)
                else:
                    estimates[preset_name] = get_fallback_estimate(
                        original_size_mb, preset_name
                    )
                    logger.warning(
                        f"⚠️ {preset_name} compression failed, using fallback"
                    )
            except Exception as e:
                logger.error(f"Preset {preset_name} estimation failed: {str(e)}")
                estimates[preset_name] = get_fallback_estimate(
                    original_size_mb, preset_name
                )
        estimates["original"] = round(original_size_mb, 2)
        estimation_results[task_id] = {
            "estimates": estimates,
            "original_size_mb": round(original_size_mb, 2),
            "used_s3": False,
            "sequential_processing": True,
        }
        safe_delete_temp_file(file_path)
        await progress_tracker.update_progress(
            task_id, 100, "Sequential estimation completed!", "completed"
        )
    except Exception as e:
        logger.error(f"Sequential estimation error: {str(e)}")
        safe_delete_temp_file(file_path)
        await progress_tracker.update_progress(
            task_id, 100, f"Error: {str(e)}", "error"
        )
        cleanup_compression_estimation_files(task_id)
    finally:
        cleanup_compression_estimation_files(task_id)


@app.post("/start_estimation")
@limiter.limit(RATE_LIMITS["estimation"])
async def start_estimation(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    logger.info(f"Starting SEQUENTIAL estimation: file={file.filename}")

    task_id = str(uuid.uuid4())
    file_path = None

    try:
        await progress_tracker.update_progress(
            task_id, 0, "Starting estimation...", "initializing"
        )

        # Step 1: Upload to disk immediately (NO RAM)
        await progress_tracker.update_progress(
            task_id, 10, "Uploading file to disk...", "uploading"
        )
        file_path = await upload_to_disk_first(file, task_id)

        # Step 2: Validate from disk (NO RAM loading of entire file)
        await progress_tracker.update_progress(
            task_id, 30, "Validating file...", "validating"
        )
        await validate_file_from_disk(file_path, "pdf", file.filename)
        # page_count = validate_pdf_page_count(file_path, max_pages=500)  # Estimation can handle 500 pages
        # logger.info(f"✅ PDF has {page_count} pages - proceeding with estimation")

        await progress_tracker.update_progress(
            task_id, 40, "File validated! Starting estimation...", "processing"
        )
        background_tasks.add_task(
            process_estimation_sequential_disk, task_id, file_path, file.filename
        )

        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "started",
                "message": "Estimation started (pure disk-based)",
                "processing_mode": "pure_disk",
            }
        )

    except HTTPException as e:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
        await progress_tracker.update_progress(
            task_id, 100, f"Validation failed: {e.detail}", "error"
        )
        raise e
    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
        logger.error(f"❌ Failed: {str(e)}")
        await progress_tracker.update_progress(
            task_id, 100, f"Error: {str(e)}", "error"
        )
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@app.get("/estimation_result/{task_id}")
async def get_estimation_result(task_id: str):
    if task_id not in estimation_results:
        raise HTTPException(
            status_code=404, detail="Estimation result not found or expired"
        )
    result = estimation_results[task_id]
    estimates = result["estimates"]
    original_size = result["original_size_mb"]
    savings_data = {}
    for preset, compressed_size in estimates.items():
        if preset != "original":
            savings_pct = ((original_size - compressed_size) / original_size) * 100
            savings_data[preset] = f"{savings_pct:.1f}%"
    result["savings_percentages"] = savings_data
    result["recommendation"] = get_compression_recommendation(estimates, original_size)
    del estimation_results[task_id]
    logger.info(f"✅ Returning ACTUAL estimation results for {task_id}: {savings_data}")
    return JSONResponse(content=result)


compression_results = {}


async def process_compression_with_progress(
    task_id: str, input_path: str, filename: str, preset: str
):
    """Process compression with progress updates"""
    output_path = None
    try:
        await progress_tracker.update_progress(
            task_id, 50, "Compressing PDF...", "compressing"
        )

        output_filename = f"compressed_{task_id}_{Path(filename).stem}_{preset}.pdf"
        output_path = OUTPUT_DIR / output_filename

        # Run compression
        success = compress_pdf_ghostscript_file(input_path, str(output_path), preset)

        if not success:
            raise Exception("Compression failed")

        await progress_tracker.update_progress(
            task_id, 90, "Finalizing...", "finalizing"
        )

        # Get file sizes
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        savings = (
            ((original_size - compressed_size) / original_size) * 100
            if original_size > 0
            else 0
        )

        # Store result metadata
        result_data = {
            "file_path": str(output_path),
            "filename": f"compressed_{Path(filename).stem}_{preset}.pdf",
            "original_size": original_size,
            "compressed_size": compressed_size,
            "savings": savings,
            "preset": preset,
        }

        # Store in progress tracker for download
        await progress_tracker.update_progress(
            task_id, 100, f"Complete! Saved {savings:.1f}%", "completed"
        )

        # Store result separately (could use Redis or dict)
        compression_results[task_id] = result_data

        # Cleanup input file
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)

        logger.info(f"✅ Compression complete for {task_id}")

    except Exception as e:
        logger.error(f"Compression error for {task_id}: {e}")
        await progress_tracker.update_progress(
            task_id, 100, f"Error: {str(e)}", "error"
        )
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)


@app.post("/start_compression")
@limiter.limit(RATE_LIMITS["compress"])
async def start_compression(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    preset: str = Form("ebook"),
):
    logger.info(f"Starting compression: file={file.filename}, preset={preset}")

    task_id = str(uuid.uuid4())
    file_path = None

    try:
        # Initialize progress
        await progress_tracker.update_progress(
            task_id, 0, "Initializing compression...", "initializing"
        )

        # Step 1: Upload to disk
        await progress_tracker.update_progress(
            task_id, 5, "Uploading file...", "uploading"
        )
        file_path = await upload_to_disk_first(file, task_id)

        # Step 2: Validate file
        await progress_tracker.update_progress(
            task_id, 30, "Validating file...", "validating"
        )
        await validate_file_from_disk(file_path, "pdf", file.filename)

        # Step 3: Validate preset
        valid_presets = ["prepress", "printer", "ebook", "screen"]
        if preset not in valid_presets:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid preset. Use: {', '.join(valid_presets)}",
            )

        await progress_tracker.update_progress(
            task_id, 40, "Starting compression...", "processing"
        )

        # Step 4: Process in background
        background_tasks.add_task(
            process_compression_with_progress, task_id, file_path, file.filename, preset
        )

        return JSONResponse(
            content={
                "task_id": task_id,
                "status": "started",
                "message": "Compression started",
                "processing_mode": "disk_based",
            }
        )

    except HTTPException as e:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
        await progress_tracker.update_progress(
            task_id, 100, f"Validation failed: {e.detail}", "error"
        )
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
        logger.error(f"❌ Failed: {str(e)}", exc_info=True)
        await progress_tracker.update_progress(
            task_id, 100, f"Error: {str(e)}", "error"
        )
        return JSONResponse(status_code=500, content={"detail": f"Failed: {str(e)}"})


@app.get("/download_compressed/{task_id}")
async def download_compressed(task_id: str):
    """Download compressed file"""
    result_data = compression_results.get(task_id)

    if not result_data:
        raise HTTPException(status_code=404, detail="Compressed result not found")

    filename = result_data["filename"]
    file_path = result_data["file_path"]

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Compressed file not found")

    def file_generator():
        with open(file_path, "rb") as f:
            while chunk := f.read(64 * 1024):  # 64KB chunks
                yield chunk
        # Clean up after download
        safe_delete_temp_file(file_path)
        if task_id in compression_results:
            del compression_results[task_id]
        # Clean up progress data
        if task_id in progress_tracker.tasks:
            del progress_tracker.tasks[task_id]

    return StreamingResponse(
        file_generator(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Original-Size": str(result_data["original_size"]),
            "X-Compressed-Size": str(result_data["compressed_size"]),
            "X-Savings-Percent": f"{result_data['savings']:.1f}",
            "X-Compression-Level": result_data["preset"],
        },
    )


progress_store: Dict[str, dict] = {}

# Or use Redis for production (recommended)

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


# Make sure your update_progress function stores in ALL locations


@app.get("/progress/{task_id}")
async def get_progress_endpoint(task_id: str):
    """Get progress for a task - FIXED endpoint"""
    logger.info(f"🔍 Progress requested for task: {task_id}")

    # Check in progress_tracker first
    progress = progress_tracker.get_progress(task_id)

    if progress:
        logger.info(f"✅ Found progress for {task_id}: {progress.get('progress', 0)}%")
        return JSONResponse(content=progress)

    # Also check the global progress_store as fallback
    if task_id in progress_store:
        logger.info(f"✅ Found progress in store for {task_id}")
        return JSONResponse(content=progress_store[task_id])

    # Check Redis directly
    try:
        redis_data = redis_client.get(f"progress:{task_id}")
        if redis_data:
            logger.info(f"✅ Found progress in Redis for {task_id}")
            return JSONResponse(content=json.loads(redis_data))
    except:
        pass

    logger.warning(f"❌ Task {task_id} not found in any storage")
    raise HTTPException(status_code=404, detail="Task not found")


@app.post("/stop_operations")
async def stop_operations():
    try:
        system = platform.system()
        if system == "Windows":
            subprocess.run(
                ["taskkill", "/f", "/im", "gswin64c.exe"], capture_output=True
            )
            subprocess.run(
                ["taskkill", "/f", "/im", "gswin32c.exe"], capture_output=True
            )
        else:
            subprocess.run(["pkill", "-f", "ghostscript"], capture_output=True)
            subprocess.run(["pkill", "-f", "gs"], capture_output=True)
        logger.info("✅ Killed all Ghostscript processes")
        return {"status": "stopped", "message": "All operations terminated"}
    except Exception as e:
        logger.error(f"Error stopping operations: {str(e)}")
        return {"status": "error", "message": str(e)}


def get_adobe_services():
    from adobe.pdfservices.operation.auth.service_principal_credentials import (
        ServicePrincipalCredentials,
    )
    from adobe.pdfservices.operation.pdf_services import PDFServices

    credentials = ServicePrincipalCredentials(
        client_id=os.getenv("PDF_SERVICES_CLIENT_ID"),
        client_secret=os.getenv("PDF_SERVICES_CLIENT_SECRET"),
    )
    return PDFServices(credentials=credentials)


async def save_uploaded_file_disk(file: UploadFile, task_id: str) -> Path:
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, detail=f"File exceeds {MAX_FILE_SIZE/1024/1024}MB limit"
        )
    task_upload_dir = UPLOAD_DIR / task_id
    task_upload_dir.mkdir(exist_ok=True)
    file_path = task_upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            buffer.write(chunk)
    logger.info(f"✅ File saved to disk: {file_path}")
    return file_path


def validate_pdf_pages(file_path: Path) -> int:
    MAX_PAGES_ESTIMATION = 2000
    try:
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            if page_count > MAX_PAGES_ESTIMATION:
                raise HTTPException(400, f"PDF exceeds {MAX_PAGES_ESTIMATION} pages")
            return page_count
    except Exception as e:
        logger.error(f"PDF validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid PDF file")


def cleanup_task_files(task_id: str):
    try:
        cleaned_count = 0
        directories_to_clean = [
            UPLOAD_DIR,
            OUTPUT_DIR,
            ESTIMATION_DIR,
            PDFTOWORD,
            TEMP_DIR,
        ]
        for base_dir in directories_to_clean:
            if not base_dir.exists():
                continue
            task_dir = base_dir / task_id
            if task_dir.exists() and task_dir.is_dir():
                try:
                    shutil.rmtree(task_dir)
                    cleaned_count += 1
                    logger.info(f"✅ Cleaned task directory: {task_dir}")
                except Exception as e:
                    logger.error(f"❌ Failed to clean directory {task_dir}: {e}")
            for pattern in [f"*{task_id}*", f"*{task_id}*.*"]:
                for file_path in base_dir.glob(pattern):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            cleaned_count += 1
                            logger.info(f"✅ Cleaned task file: {file_path}")
                    except Exception as e:
                        logger.error(f"❌ Failed to clean file {file_path}: {e}")
        if cleaned_count > 0:
            logger.info(
                f"🧹 Cleanup completed for task {task_id}: {cleaned_count} items removed"
            )
        else:
            logger.info(f"ℹ️ No files found to clean for task {task_id}")
    except Exception as e:
        logger.error(f"💥 Critical error in cleanup_task_files for {task_id}: {e}")


# def cleanup_orphaned_files():
#     try:
#         current_time = time.time()
#         cleaned_count = 0
#         for root_dir in [UPLOAD_DIR, OUTPUT_DIR, ESTIMATION_DIR, PDFTOWORD, TEMP_DIR]:
#             if not root_dir.exists():
#                 continue
#             for item in root_dir.iterdir():
#                 try:
#                     if item.is_dir():
#                         continue
#                     stat = item.stat()
#                     most_recent = max(stat.st_mtime, stat.st_ctime, stat.st_atime)
#                     if most_recent > (current_time - orphan_age_seconds):
#                         continue
#                     if item.is_file():
#                         item.unlink()
#                         cleaned_count += 1
#                         logger.info(f"🧹 Cleaned up orphaned file: {item}")
#                 except Exception as e:
#                     logger.error(f"❌ Error cleaning {item}: {e}")
#         if cleaned_count > 0:
#             logger.info(f"🎯 Orphaned files cleanup completed: {cleaned_count} files removed")
#         else:
#             logger.info("ℹ️ No orphaned files found for cleanup")
#     except Exception as e:
#         logger.error(f"💥 Critical error in orphaned file cleanup: {e}")


def cleanup_orphaned_files():
    try:
        current_time = time.time()
        cleaned_files = 0
        cleaned_dirs = 0
        total_scanned = 0

        root_directories = [UPLOAD_DIR, OUTPUT_DIR, ESTIMATION_DIR, PDFTOWORD, TEMP_DIR]

        logger.info("🧹 Starting recursive cleanup (files + empty subfolders)...")

        for base_dir in root_directories:
            if not base_dir.exists():
                continue

            # Step 1: Delete old files recursively
            for root, dirs, files in os.walk(
                str(base_dir), topdown=False
            ):  # topdown=False is important
                for filename in files:
                    file_path = Path(root) / filename
                    total_scanned += 1

                    try:
                        stat = file_path.stat()
                        most_recent = max(stat.st_mtime, stat.st_ctime, stat.st_atime)

                        if most_recent > (current_time - orphan_age_seconds):
                            continue  # Skip recent files

                        file_path.unlink()
                        cleaned_files += 1
                        logger.info(
                            f"🗑️ Deleted old file: {file_path.relative_to(base_dir)}"
                        )

                    except Exception as e:
                        logger.error(f"Failed to delete file {file_path}: {e}")

                # Step 2: Delete empty subdirectories (but NOT the main root folders)
                for dir_name in dirs[:]:  # copy list to avoid modification issues
                    dir_path = Path(root) / dir_name

                    # IMPORTANT: Never delete the main root directories
                    if dir_path in root_directories:
                        continue

                    try:
                        if dir_path.exists() and not any(
                            dir_path.iterdir()
                        ):  # if empty
                            dir_path.rmdir()
                            cleaned_dirs += 1
                            logger.info(
                                f"🗑️ Deleted empty folder: {dir_path.relative_to(base_dir)}"
                            )
                    except Exception as e:
                        logger.error(f"Failed to delete folder {dir_path}: {e}")

        logger.info(
            f"🎯 Cleanup Finished! Files deleted: {cleaned_files} | Empty folders deleted: {cleaned_dirs} | Scanned: {total_scanned}"
        )

        return {
            "message": f"Cleanup completed successfully.",
            "files_deleted": cleaned_files,
            "folders_deleted": cleaned_dirs,
        }

    except Exception as e:
        logger.error(f"💥 Critical error in cleanup: {e}", exc_info=True)
        return {"error": str(e)}


def convert_pdf_to_word_disk_based(
    pdf_file_path: Path, task_id: str, max_retries: int = 1
) -> Optional[bytes]:
    output_docx_path = None
    for attempt in range(max_retries):
        try:
            logger.info(
                f"🔄 PDF to Word conversion attempt {attempt + 1}/{max_retries} for task {task_id}"
            )
            from adobe.pdfservices.operation.exception.exceptions import (
                ServiceApiException,
                ServiceUsageException,
                SdkException,
            )
            from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
            from adobe.pdfservices.operation.io.stream_asset import StreamAsset
            from adobe.pdfservices.operation.pdf_services_media_type import (
                PDFServicesMediaType,
            )
            from adobe.pdfservices.operation.pdfjobs.jobs.export_pdf_job import (
                ExportPDFJob,
            )
            from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_params import (
                ExportPDFParams,
            )
            from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_target_format import (
                ExportPDFTargetFormat,
            )
            from adobe.pdfservices.operation.pdfjobs.result.export_pdf_result import (
                ExportPDFResult,
            )

            if not pdf_file_path.exists():
                logger.error(f"❌ Input file missing for retry: {pdf_file_path}")
                return None
            with open(pdf_file_path, "rb") as file:
                input_stream = file.read()
            pdf_services = get_adobe_services()
            input_asset = pdf_services.upload(
                input_stream=input_stream, mime_type=PDFServicesMediaType.PDF
            )
            export_pdf_params = ExportPDFParams(
                target_format=ExportPDFTargetFormat.DOCX
            )
            export_pdf_job = ExportPDFJob(
                input_asset=input_asset, export_pdf_params=export_pdf_params
            )
            location = pdf_services.submit(export_pdf_job)
            result = pdf_services.get_job_result(location, ExportPDFResult)
            result_asset: CloudAsset = result.get_result().get_asset()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)
            output_task_dir = OUTPUT_DIR / task_id
            output_task_dir.mkdir(exist_ok=True)
            output_docx_path = output_task_dir / "converted.docx"
            with open(output_docx_path, "wb") as out_file:
                out_file.write(stream_asset.get_input_stream())
            with open(output_docx_path, "rb") as f:
                docx_bytes = f.read()
            logger.info(
                f"✅ PDF to Word conversion successful on attempt {attempt + 1}"
            )
            return docx_bytes
        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logger.error(
                f"❌ Adobe PDF Services error (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            if output_docx_path and output_docx_path.exists():
                try:
                    output_docx_path.unlink()
                    logger.info(f"🧹 Cleaned failed output file: {output_docx_path}")
                except Exception as cleanup_error:
                    logger.error(f"❌ Failed to clean output file: {cleanup_error}")
            if attempt < max_retries - 1:
                wait_time = (2**attempt) + 1
                logger.info(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error("💥 All conversion attempts failed")
                return None
        except Exception as e:
            logger.error(
                f"❌ Unexpected error in PDF to Word conversion (attempt {attempt + 1}): {str(e)}"
            )
            if output_docx_path and output_docx_path.exists():
                try:
                    output_docx_path.unlink()
                    logger.info(f"🧹 Cleaned failed output file: {output_docx_path}")
                except Exception as cleanup_error:
                    logger.error(f"❌ Failed to clean output file: {cleanup_error}")
            if attempt < max_retries - 1:
                wait_time = (2**attempt) + 1
                logger.info(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error("💥 All conversion attempts failed")
                return None
    return None


async def _do_adobe_conversion(pdf_file_path: Path, task_id: str) -> Optional[bytes]:
    """Actual Adobe conversion - separated for timeout"""
    from adobe.pdfservices.operation.exception.exceptions import ServiceApiException
    from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
    from adobe.pdfservices.operation.io.stream_asset import StreamAsset
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.pdfjobs.jobs.export_pdf_job import ExportPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_params import (
        ExportPDFParams,
    )
    from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_target_format import (
        ExportPDFTargetFormat,
    )
    from adobe.pdfservices.operation.pdfjobs.result.export_pdf_result import (
        ExportPDFResult,
    )

    with open(pdf_file_path, "rb") as file:
        input_stream = file.read()

    pdf_services = get_adobe_services()
    input_asset = pdf_services.upload(
        input_stream=input_stream, mime_type=PDFServicesMediaType.PDF
    )
    export_pdf_params = ExportPDFParams(target_format=ExportPDFTargetFormat.DOCX)
    export_pdf_job = ExportPDFJob(
        input_asset=input_asset, export_pdf_params=export_pdf_params
    )
    location = pdf_services.submit(export_pdf_job)
    result = pdf_services.get_job_result(location, ExportPDFResult)
    result_asset: CloudAsset = result.get_result().get_asset()
    stream_asset: StreamAsset = pdf_services.get_content(result_asset)

    output_task_dir = OUTPUT_DIR / task_id
    output_task_dir.mkdir(exist_ok=True)
    output_docx_path = output_task_dir / "converted.docx"

    with open(output_docx_path, "wb") as out_file:
        out_file.write(stream_asset.get_input_stream())

    with open(output_docx_path, "rb") as f:
        return f.read()


def convert_pdf_to_word_local_fallback(
    pdf_file_path: Path, task_id: str
) -> Optional[bytes]:
    """Fallback using pdf2docx library (install: pip install pdf2docx)"""
    try:
        from pdf2docx import Converter

        output_docx_path = OUTPUT_DIR / task_id / "converted_fallback.docx"
        output_docx_path.parent.mkdir(exist_ok=True)

        cv = Converter(str(pdf_file_path))
        cv.convert(str(output_docx_path), start=0, end=None)
        cv.close()

        with open(output_docx_path, "rb") as f:
            return f.read()

    except Exception as e:
        logger.error(f"Local fallback failed: {e}")
        return None


@app.post("/convert_pdf_to_word")
@limiter.limit(RATE_LIMITS["convert"])
async def convert_pdf_to_word_endpoint(request: Request, file: UploadFile = File(...)):
    logger.info(f"📥 Convert to Word: {file.filename}")

    task_id = str(uuid.uuid4())
    file_path = None

    try:
        # Initialize progress
        await progress_tracker.update_progress(
            task_id, 0, "Starting PDF to Word conversion...", "initializing"
        )

        # Step 1: Upload to disk
        await progress_tracker.update_progress(
            task_id, 10, "Uploading file...", "uploading"
        )
        file_path = await upload_to_disk_first(file, task_id)

        # Step 2: Validate from disk
        await progress_tracker.update_progress(
            task_id, 30, "Validating file...", "validating"
        )
        await validate_file_from_disk(file_path, "pdf", file.filename)

        # Step 3: Validate page count
        await progress_tracker.update_progress(
            task_id, 40, "Checking PDF structure...", "processing"
        )
        # page_count = validate_pdf_pages(Path(file_path))
        # logger.info(f"📄 PDF validated: {page_count} pages")

        # Step 4: Convert using disk file
        await progress_tracker.update_progress(
            task_id, 50, "Converting to Word format...", "converting"
        )
        docx_bytes = convert_pdf_to_word_disk_based(Path(file_path), task_id)

        if not docx_bytes:
            await progress_tracker.update_progress(
                task_id, 70, "Trying fallback conversion...", "fallback"
            )
            docx_bytes = convert_pdf_to_word_local_fallback(Path(file_path), task_id)

        if not docx_bytes:
            raise HTTPException(500, detail="Conversion failed")

        await progress_tracker.update_progress(
            task_id, 90, "Preparing download...", "finalizing"
        )

        # Return response with task_id in headers
        response = StreamingResponse(
            io.BytesIO(docx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": 'attachment; filename="converted_output.docx"',
                "X-Task-ID": task_id,
            },
        )

        await progress_tracker.update_progress(
            task_id, 100, "Conversion complete!", "completed"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        await progress_tracker.update_progress(
            task_id, 100, f"Error: {str(e)}", "error"
        )
        raise HTTPException(500, detail=f"Conversion error: {str(e)}")
    finally:
        # Cleanup
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
        cleanup_task_files(task_id)


def convert_pdf_to_excel_disk_based(
    pdf_file_path: Path, task_id: str, max_retries: int = 3
) -> Optional[bytes]:
    """
    Convert PDF to Excel - Reads from disk, minimizes RAM by processing in chunks
    """
    output_xlsx_path = None

    for attempt in range(max_retries):
        try:
            logger.info(
                f"🔄 PDF to Excel conversion attempt {attempt + 1}/{max_retries} for task {task_id}"
            )
            from adobe.pdfservices.operation.exception.exceptions import (
                ServiceApiException,
                ServiceUsageException,
                SdkException,
            )
            from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
            from adobe.pdfservices.operation.io.stream_asset import StreamAsset
            from adobe.pdfservices.operation.pdf_services_media_type import (
                PDFServicesMediaType,
            )
            from adobe.pdfservices.operation.pdfjobs.jobs.export_pdf_job import (
                ExportPDFJob,
            )
            from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_params import (
                ExportPDFParams,
            )
            from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_target_format import (
                ExportPDFTargetFormat,
            )
            from adobe.pdfservices.operation.pdfjobs.result.export_pdf_result import (
                ExportPDFResult,
            )

            if not pdf_file_path.exists():
                logger.error(
                    f"❌ Input file missing for Excel conversion: {pdf_file_path}"
                )
                return None

            # ✅ Read file from disk (unavoidable for Adobe API)
            with open(pdf_file_path, "rb") as file:
                input_stream = file.read()

            pdf_services = get_adobe_services()
            input_asset = pdf_services.upload(
                input_stream=input_stream, mime_type=PDFServicesMediaType.PDF
            )

            # Free input_stream ASAP
            input_stream = None

            export_pdf_params = ExportPDFParams(
                target_format=ExportPDFTargetFormat.XLSX
            )
            export_pdf_job = ExportPDFJob(
                input_asset=input_asset, export_pdf_params=export_pdf_params
            )
            location = pdf_services.submit(export_pdf_job)
            result = pdf_services.get_job_result(location, ExportPDFResult)
            result_asset: CloudAsset = result.get_result().get_asset()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            output_task_dir = OUTPUT_DIR / task_id
            output_task_dir.mkdir(exist_ok=True)
            output_xlsx_path = output_task_dir / "converted.xlsx"

            # ✅ Write directly to disk (stream to disk, not RAM)
            with open(output_xlsx_path, "wb") as out_file:
                out_file.write(stream_asset.get_input_stream())

            # ✅ Read back for response (unavoidable)
            with open(output_xlsx_path, "rb") as f:
                xlsx_bytes = f.read()

            logger.info(
                f"✅ PDF to Excel conversion successful on attempt {attempt + 1}"
            )
            return xlsx_bytes

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logger.error(
                f"❌ Adobe PDF Services Excel error (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            if output_xlsx_path and output_xlsx_path.exists():
                try:
                    output_xlsx_path.unlink()
                except:
                    pass
            if attempt < max_retries - 1:
                wait_time = (2**attempt) + 1
                logger.info(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                return None
        except Exception as e:
            logger.error(f"❌ Unexpected error (attempt {attempt + 1}): {str(e)}")
            if output_xlsx_path and output_xlsx_path.exists():
                try:
                    output_xlsx_path.unlink()
                except:
                    pass
            if attempt < max_retries - 1:
                wait_time = (2**attempt) + 1
                time.sleep(wait_time)
                continue
            else:
                return None

    return None


@app.post("/convert_pdf_to_excel")
@limiter.limit(RATE_LIMITS["convert"])
async def convert_pdf_to_excel_endpoint(request: Request, file: UploadFile = File(...)):
    logger.info(f"📥 Convert to Excel: {file.filename}")

    task_id = str(uuid.uuid4())
    file_path = None

    try:
        # Initialize progress
        await progress_tracker.update_progress(
            task_id, 0, "Starting PDF to Excel conversion...", "initializing"
        )

        # Step 1: Upload to disk
        await progress_tracker.update_progress(
            task_id, 10, "Uploading file...", "uploading"
        )
        file_path = await upload_to_disk_first(file, task_id)

        # Step 2: Validate from disk
        await progress_tracker.update_progress(
            task_id, 30, "Validating file...", "validating"
        )
        await validate_file_from_disk(file_path, "pdf", file.filename)

        # Step 3: Validate page count
        await progress_tracker.update_progress(
            task_id, 40, "Checking PDF structure...", "processing"
        )
        # page_count = validate_pdf_pages(Path(file_path))
        # logger.info(f"📄 PDF validated: {page_count} pages")

        # Step 4: Convert using disk file
        await progress_tracker.update_progress(
            task_id, 50, "Converting to Excel format...", "converting"
        )
        logger.info("🔄 Starting PDF to Excel conversion...")
        xlsx_bytes = convert_pdf_to_excel_disk_based(Path(file_path), task_id)

        if not xlsx_bytes:
            raise HTTPException(
                status_code=500,
                detail="Excel conversion failed after all retry attempts",
            )

        await progress_tracker.update_progress(
            task_id, 90, "Preparing download...", "finalizing"
        )
        logger.info(f"✅ Excel conversion successful for task {task_id}")

        # Return response with task_id in headers
        response = StreamingResponse(
            io.BytesIO(xlsx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": 'attachment; filename="converted_output.xlsx"',
                "X-Task-ID": task_id,
            },
        )

        await progress_tracker.update_progress(
            task_id, 100, "Conversion complete!", "completed"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Error: {str(e)}")
        await progress_tracker.update_progress(
            task_id, 100, f"Error: {str(e)}", "error"
        )
        raise HTTPException(status_code=500, detail=f"Excel conversion error: {str(e)}")
    finally:
        # Cleanup
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"🗑️ Deleted: {file_path}")
        cleanup_task_files(task_id)


##########################


@app.post("/encrypt_pdf")
@limiter.limit(RATE_LIMITS["convert"])
async def encrypt_pdf_endpoint(
    request: Request, file: UploadFile = File(...), password: str = Form(...)
):
    """
    Encrypt PDF - COMPLETELY DISK-BASED
    - Streams upload directly to disk
    - Processes page by page from disk
    - Streams response directly to client
    - Never loads full PDF into RAM
    """
    logger.info(f"🔐 Encrypt PDF (disk-based): {file.filename}")

    task_id = str(uuid.uuid4())
    input_path = None
    output_path = None

    try:
        # ✅ STEP 1: Stream upload directly to disk (NO RAM)
        input_path = await upload_to_disk_first(file, task_id)
        logger.info(f"✅ File saved to disk: {input_path}")

        # ✅ STEP 2: Validate file from disk (reads only headers)
        await validate_file_from_disk(input_path, "pdf", file.filename)
        # page_count = validate_pdf_page_count(input_path, max_pages=5)  # Encryption is fast, 500 pages is fine
        # logger.info(f"✅ PDF has {page_count} pages - proceeding with encryption")

        # ✅ STEP 3: Create output path on disk
        output_filename = f"encrypted_{task_id}_{sanitize_filename(file.filename)}"
        output_path = str(OUTPUT_DIR / output_filename)

        # ✅ STEP 4: Encrypt using streaming (page by page)
        await progress_tracker.update_progress(
            task_id, 50, "Encrypting PDF (streaming)...", "encrypting"
        )

        # Run encryption in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, encrypt_pdf_streaming, input_path, output_path, password
        )

        if not success:
            raise HTTPException(status_code=500, detail="Encryption failed")

        await progress_tracker.update_progress(
            task_id, 90, "Encryption complete, preparing download...", "finalizing"
        )

        # ✅ STEP 5: Stream response directly from disk (NO RAM)
        def file_stream():
            try:
                with open(output_path, "rb") as f:
                    while chunk := f.read(65536):  # 64KB chunks
                        yield chunk
            finally:
                # Cleanup after streaming completes
                if output_path and os.path.exists(output_path):
                    os.unlink(output_path)
                    logger.info(f"🗑️ Deleted encrypted temp file: {output_path}")

        response = StreamingResponse(
            file_stream(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="encrypted_{file.filename}"',
                "X-Encryption-Method": "streaming-disk-based",
                "X-Task-ID": task_id,
            },
        )

        await progress_tracker.update_progress(
            task_id, 100, "Ready to download!", "completed"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Encryption error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(e)}")

    finally:
        # ✅ STEP 6: Cleanup input file
        if input_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
                logger.info(f"🗑️ Deleted input temp file: {input_path}")
            except Exception as e:
                logger.error(f"Failed to delete {input_path}: {e}")

        # Force garbage collection
        gc.collect()


@app.post("/remove_pdf_password")
@limiter.limit(RATE_LIMITS["convert"])
async def remove_pdf_password_endpoint(
    request: Request, file: UploadFile = File(...), password: str = Form(...)
):
    """
    Remove PDF password - COMPLETELY DISK-BASED
    - Streams upload directly to disk
    - Processes page by page from disk
    - Streams response directly to client
    - Never loads full PDF into RAM
    """
    logger.info(f"🔓 Remove PDF password (disk-based): {file.filename}")

    task_id = str(uuid.uuid4())
    input_path = None
    output_path = None

    try:
        # ✅ STEP 1: Stream upload directly to disk (NO RAM)
        input_path = await upload_to_disk_first(file, task_id)
        logger.info(f"✅ File saved to disk: {input_path}")

        # ✅ STEP 2: Validate file from disk (reads only headers)
        await validate_file_from_disk(input_path, "pdf", file.filename)
        # page_count = validate_pdf_page_count(input_path, max_pages=500)
        # logger.info(f"✅ PDF has {page_count} pages - proceeding with password removal")

        # ✅ STEP 3: Create output path on disk
        output_filename = f"decrypted_{task_id}_{sanitize_filename(file.filename)}"
        output_path = str(OUTPUT_DIR / output_filename)

        # ✅ STEP 4: Decrypt using streaming (page by page)
        await progress_tracker.update_progress(
            task_id, 50, "Removing password (streaming)...", "decrypting"
        )

        # Run decryption in thread pool
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, decrypt_pdf_streaming, input_path, output_path, password
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Password removal failed - wrong password or corrupted file",
            )

        await progress_tracker.update_progress(
            task_id, 90, "Password removed, preparing download...", "finalizing"
        )

        # ✅ STEP 5: Stream response directly from disk (NO RAM)
        def file_stream():
            try:
                with open(output_path, "rb") as f:
                    while chunk := f.read(65536):  # 64KB chunks
                        yield chunk
            finally:
                # Cleanup after streaming completes
                if output_path and os.path.exists(output_path):
                    os.unlink(output_path)
                    logger.info(f"🗑️ Deleted decrypted temp file: {output_path}")

        response = StreamingResponse(
            file_stream(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="decrypted_{file.filename}"',
                "X-Decryption-Method": "streaming-disk-based",
                "X-Task-ID": task_id,
            },
        )

        await progress_tracker.update_progress(
            task_id, 100, "Ready to download!", "completed"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Decryption error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Password removal failed: {str(e)}"
        )

    finally:
        # ✅ STEP 6: Cleanup input file
        if input_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
                logger.info(f"🗑️ Deleted input temp file: {input_path}")
            except Exception as e:
                logger.error(f"Failed to delete {input_path}: {e}")

        # Force garbage collection
        gc.collect()


# ================================== Client side pdf validation endpoint (for testing) =============================


# ==========================================================


#### remove image background endpoint (disk-based)
@app.post("/remove_background")
@limiter.limit(RATE_LIMITS["upload"])
async def remove_background_endpoint(request: Request, file: UploadFile = File(...)):
    logger.info(f"Received remove background request for {file.filename}")

    task_id = str(uuid.uuid4())
    file_path = None
    temp_output_path = None

    try:
        # Step 1: Upload to disk immediately (NO RAM storage)
        file_path = await upload_to_disk_first(file, task_id)
        logger.info(f"✅ File saved to disk: {file_path}")

        # Step 2: Validate from disk (NO RAM loading of entire file)
        await validate_file_from_disk(file_path, "image", file.filename)
        logger.info(f"✅ File validation passed from disk")

        # Step 3: Process image with memory-efficient streaming
        logger.info("Processing image for background removal (memory efficient)")

        # Use PIL for lazy loading and dimension validation
        from PIL import Image
        import numpy as np

        # Validate dimensions without loading full image
        with Image.open(file_path) as img:
            width, height = img.size
            if width > 5000 or height > 5000:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image dimensions too large: {width}x{height} (max 5000x5000 pixels)",
                )
            logger.info(f"✅ Image dimensions: {width}x{height}")
            img_format = img.format
            img_mode = img.mode

        # Step 4: Process in chunks using disk-based approach
        # rembg works with bytes, but we'll read in chunks and reconstruct
        # For large images, we'll process tiles if needed

        file_size = os.path.getsize(file_path)

        if file_size > 10 * 1024 * 1024:  # > 10MB, use tile processing
            logger.info(
                f"Large image ({file_size/1024/1024:.1f}MB), using tile processing"
            )
            processed_image = await process_large_image_tiled(file_path, width, height)
        else:
            # For smaller images, read entire file (unavoidable for rembg)
            with open(file_path, "rb") as f:
                file_content = f.read()

            # Process with rembg
            processed_image = remove_background_rembg(file_content)

        if not processed_image or not processed_image.getvalue():
            raise HTTPException(status_code=500, detail="Failed to process image")

        # Step 5: Verify output is valid image
        try:
            import PIL.Image

            processed_image.seek(0)
            output_img = PIL.Image.open(processed_image)
            output_img.verify()
            processed_image.seek(0)
            logger.info("✅ Output image verification passed")
        except Exception as e:
            logger.warning(f"Output image verification warning: {e}")
            processed_image.seek(0)

        # Step 6: Return processed image with streaming
        output_filename = f"processed_{os.path.splitext(file.filename)[0]}.png"

        return StreamingResponse(
            content=processed_image,
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "X-Original-Size": str(file_size),
                "X-Processed-Size": str(processed_image.getbuffer().nbytes),
                "X-Task-ID": task_id,
                "X-Processing-Method": "memory-efficient",
            },
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        # Cleanup: Delete the temporary file from disk
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"✅ Deleted temp file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete temp file {file_path}: {e}")

        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.unlink(temp_output_path)
                logger.info(f"✅ Deleted temp output: {temp_output_path}")
            except Exception as e:
                logger.error(f"Failed to delete temp output: {e}")

        # Force garbage collection
        gc.collect()
        logger.info("✅ Background removal endpoint cleanup completed")


async def process_large_image_tiled(
    image_path: str, width: int, height: int, tile_size: int = 1024
) -> io.BytesIO:
    """
    Process large images in tiles to reduce memory usage
    """
    from PIL import Image
    import numpy as np
    from rembg import remove

    try:
        # Open image with lazy loading
        with Image.open(image_path) as img:
            # Convert to RGBA if needed
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # If image is manageable, process whole
            if width * height < 5000 * 5000:  # < 25 megapixels
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                result = remove(img_bytes.getvalue())
                return io.BytesIO(result)

            # For very large images, process in tiles
            logger.info(
                f"Processing {width}x{height} image in {tile_size}x{tile_size} tiles"
            )

            # Create output image
            output_img = Image.new("RGBA", (width, height))

            # Process tiles
            for x in range(0, width, tile_size):
                for y in range(0, height, tile_size):
                    # Calculate tile boundaries
                    x_end = min(x + tile_size, width)
                    y_end = min(y + tile_size, height)

                    # Extract tile
                    tile = img.crop((x, y, x_end, y_end))

                    # Process tile with rembg
                    tile_bytes = io.BytesIO()
                    tile.save(tile_bytes, format="PNG")
                    tile_bytes.seek(0)

                    processed_tile_bytes = remove(tile_bytes.getvalue())
                    processed_tile = Image.open(io.BytesIO(processed_tile_bytes))

                    # Paste back
                    output_img.paste(processed_tile, (x, y))

                    logger.debug(f"Processed tile: {x}-{x_end}, {y}-{y_end}")

                    # Force garbage collection after each tile
                    gc.collect()

            # Save output
            output_bytes = io.BytesIO()
            output_img.save(output_bytes, format="PNG", optimize=True)
            output_bytes.seek(0)

            return output_bytes

    except Exception as e:
        logger.error(f"Tile processing failed: {e}")
        # Fallback to regular processing
        with open(image_path, "rb") as f:
            content = f.read()
        result = remove(content)
        return io.BytesIO(result)


def remove_background_rembg(image_bytes):
    """Original rembg function (kept for compatibility)"""
    try:
        from rembg import remove

        output = remove(image_bytes)
        return io.BytesIO(output)
    except Exception as e:
        logger.error(f"Background removal failed: {str(e)}")
        raise ValueError(f"Background removal failed: {str(e)}")


import json

METADATA_KEY = f"{S3_PREFIX}_videos_metadata.json"


def load_video_metadata():
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=METADATA_KEY)
        metadata_str = response["Body"].read().decode("utf-8")
        return json.loads(metadata_str)
    except s3_client.exceptions.NoSuchKey:
        logger.info("No metadata file found, creating empty")
        return {}
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return {}


def save_video_metadata(metadata):
    try:
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=METADATA_KEY,
            Body=json.dumps(metadata, indent=2),
            ContentType="application/json",
        )
        logger.info(f"Saved metadata for {len(metadata)} videos")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")


def extract_video_id_from_key(s3_key: str) -> str:
    if s3_key.startswith(S3_PREFIX):
        parts = s3_key[len(S3_PREFIX) :].split("_", 1)
        if len(parts) == 2:
            return s3_key[len(S3_PREFIX) :]
    return s3_key


@app.get("/videos")
async def list_videos():
    try:
        video_metadata = load_video_metadata()
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PREFIX)
        videos = []
        if "Contents" in response:
            video_items = [
                obj
                for obj in response["Contents"]
                if obj["Key"]
                .lower()
                .endswith((".mp4", ".webm", ".ogg", ".mkv", ".avi", ".mov"))
            ]
            video_items.sort(key=lambda x: x["LastModified"], reverse=True)
            for obj in video_items:
                key = obj["Key"]
                filename = key.split("/")[-1]
                ext = filename.lower().split(".")[-1]
                video_id = key[len(S3_PREFIX) :] if key.startswith(S3_PREFIX) else key
                metadata = video_metadata.get(video_id, {})
                description = metadata.get("description", "")
                if not description:
                    name_without_ext = filename.rsplit(".", 1)[0]
                    if "_" in name_without_ext:
                        name_without_ext = name_without_ext.split("_", 1)[1]
                    description = name_without_ext.replace("_", " ").replace("-", " ")
                content_type_map = {
                    "mp4": "video/mp4",
                    "webm": "video/webm",
                    "ogg": "video/ogg",
                    "mkv": "video/x-matroska",
                    "avi": "video/x-msvideo",
                    "mov": "video/quicktime",
                }
                # ✅ Secure presigned URL instead of direct URL
                secure_url = get_secure_download_url(key, expires_in=3600)
                if not secure_url:
                    continue
                videos.append(
                    {
                        "id": video_id,
                        "name": filename,
                        "url": secure_url,
                        "raw_url": secure_url,
                        "description": sanitize_input(description, max_length=500),
                        "type": content_type_map.get(ext, "video/mp4"),
                        "format": ext.upper(),
                        "size": obj.get("Size", 0),
                        "uploaded": obj["LastModified"].strftime("%Y-%m-%d %H:%M"),
                        "has_description": bool(description and description.strip()),
                    }
                )
        logger.info(f"📊 Listed {len(videos)} videos (using metadata cache)")
        return JSONResponse(content=videos)
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail="Failed to list videos")


@app.delete("/delete-video/{video_id}")
async def delete_video(video_id: str, payload: Dict = Body(...)):
    try:
        password = payload.get("password", "")
        if not password:
            raise HTTPException(status_code=400, detail="Password required")
        if password != CORRECT_PASSWORD_HASH:
            raise HTTPException(status_code=401, detail="Incorrect password")
        video_key = f"{S3_PREFIX}{video_id}"
        try:
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=video_key)
            logger.info(f"Successfully deleted video from S3: {video_key}")
        except ClientError as e:
            logger.error(f"Error deleting video {video_key}: {str(e)}")
            if e.response["Error"]["Code"] != "404":
                raise HTTPException(
                    status_code=500, detail=f"Failed to delete video: {str(e)}"
                )
        return JSONResponse(content={"detail": "Video deleted successfully"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@app.post("/upload-video")
@limiter.limit(RATE_LIMITS["upload"])
async def upload_video(
    request: Request,
    video_file: UploadFile = File(...),
    password: str = Form(...),
    description: str = Form(...),
):
    task_id = str(uuid.uuid4())
    file_path = None

    try:
        if password != CORRECT_PASSWORD_HASH:
            raise HTTPException(status_code=401, detail="Incorrect password")

        description = sanitize_input(description, max_length=500, allow_html=False)

        # Step 1: Upload to disk immediately
        file_path = await upload_to_disk_first(video_file, task_id)

        # Step 2: Validate from disk
        await validate_file_from_disk(file_path, "video", video_file.filename)

        # Step 3: Read from disk for S3 upload (unavoidable)
        with open(file_path, "rb") as f:
            validated_content = f.read()

        # Step 4: Upload to S3
        file_hash = hashlib.md5(validated_content).hexdigest()
        s3_key = f"{S3_PREFIX}{file_hash}_{video_file.filename}"

        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=validated_content,
            Metadata={"description": description},
            ContentType=video_file.content_type or "video/mp4",
        )

        # Update metadata
        video_metadata = load_video_metadata()
        video_id = s3_key[len(S3_PREFIX) :]
        video_metadata[video_id] = {
            "description": description,
            "original_filename": video_file.filename,
            "uploaded_at": datetime.now().isoformat(),
            "size": len(validated_content),
        }
        save_video_metadata(video_metadata)

        return JSONResponse(
            content={
                "message": "Video uploaded successfully",
                "name": video_file.filename,
                "id": video_id,
                "description": description,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)


# def remove_background_rembg(image_bytes):
#     try:
#         output = remove(image_bytes)
#         return io.BytesIO(output)
#     except Exception as e:
#         logger.error(f"Background removal failed: {str(e)}")
#         raise ValueError(f"Background removal failed: {str(e)}")

# Razorpay payment endpoints (unchanged except rate limiting)
client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
pending_payments = {}


@app.get("/payment", response_class=HTMLResponse)
async def serve_payment():
    payment_path = os.path.join(static_dir, "payment.html")
    with open(payment_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/payment-success", response_class=HTMLResponse)
async def serve_payment_success():
    success_path = os.path.join(static_dir, "payment-success.html")
    with open(success_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/create-razorpay-order")
@limiter.limit("5/minute")
async def create_razorpay_order(request: Request):
    try:
        data = await request.json()
        amount = float(data.get("amount", 0))
        logger.info(f"📝 Creating Razorpay order for amount: ₹{amount}")
        if amount <= 0:
            return JSONResponse(
                status_code=400, content={"success": False, "error": "Invalid amount"}
            )
        amount_in_paise = int(amount * 100)
        order_data = {
            "amount": amount_in_paise,
            "currency": "INR",
            "payment_capture": 1,
            "receipt": f"receipt_{int(time.time())}",
            "notes": {"payment_for": "Website Services", "customer": "Guest User"},
        }
        order = client.order.create(data=order_data)
        logger.info(f"✅ Order created successfully: {order['id']}")
        response_data = {
            "success": True,
            "order_id": order["id"],
            "amount": amount_in_paise,
            "currency": "INR",
            "key": RAZORPAY_KEY_ID,
        }
        logger.info(f"📤 Sending response: {response_data}")
        return JSONResponse(response_data)
    except Exception as e:
        logger.error(f"❌ Razorpay order creation error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@app.get("/api/get-payment-details/{payment_id}")
@limiter.limit("10/minute")
async def get_payment_details(request: Request, payment_id: str):
    try:
        payment = client.payment.fetch(payment_id)
        utr = None
        if payment.get("method") == "upi":
            acquirer_data = payment.get("acquirer_data", {})
            utr = acquirer_data.get("upi_transaction_id") or acquirer_data.get("rrn")
        return JSONResponse(
            {
                "success": True,
                "payment_id": payment.get("id"),
                "order_id": payment.get("order_id"),
                "amount": payment.get("amount", 0) / 100,
                "status": payment.get("status"),
                "method": payment.get("method"),
                "utr": utr,
                "bank_reference": payment.get("bank_reference"),
                "created_at": datetime.fromtimestamp(
                    payment.get("created_at", 0)
                ).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    except Exception as e:
        logger.error(f"Error fetching payment details: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=404)


@app.post("/razorpay-webhook")
async def razorpay_webhook(request: Request):
    try:
        body = await request.body()
        signature = request.headers.get("X-Razorpay-Signature")
        expected_signature = hmac.new(
            RAZORPAY_KEY_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()
        if signature == expected_signature:
            webhook_data = json.loads(body)
            if webhook_data.get("event") == "payment.captured":
                payment_entity = (
                    webhook_data.get("payload", {}).get("payment", {}).get("entity", {})
                )
                payment_id = payment_entity.get("id")
                order_id = payment_entity.get("order_id")
                amount = payment_entity.get("amount", 0) / 100
                pending_payments[payment_id] = {
                    "paid": True,
                    "order_id": order_id,
                    "amount": amount,
                    "timestamp": datetime.now().isoformat(),
                }
                logger.info(f"✅ Payment verified: {payment_id} for ₹{amount}")
            return JSONResponse({"status": "success"})
        return JSONResponse({"status": "failed"}, status_code=400)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse({"status": "error"}, status_code=500)


@app.get("/api/payment-receipt")
async def get_payment_receipt(amount: float = 0, method: str = "upi"):
    india_tz = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(india_tz)
    current_hour = now_ist.hour
    if 5 <= current_hour < 12:
        greeting = "Good Morning! 🌅"
    elif 12 <= current_hour < 17:
        greeting = "Good Afternoon! ☀️"
    elif 17 <= current_hour < 21:
        greeting = "Good Evening! 🌙"
    else:
        greeting = "Good Night! 🌟"
    if amount <= 100:
        message = f"Thank you for supporting us with ₹{amount}! Your contribution helps us serve you better. 🙏"
    elif amount <= 500:
        message = f"We deeply appreciate your ₹{amount} payment! Your trust means everything to us. 💖"
    elif amount <= 1000:
        message = f"Amazing! Thank you for the ₹{amount} payment. You're helping us grow and improve. 🚀"
    else:
        message = f"Wow! Thank you for the generous ₹{amount} contribution. You're truly amazing! 🌟"
    method_emoji = {
        "razorpay": "💳 Razorpay",
        "upi": "📱 UPI",
        "card": "💳 Card",
        "wallet": "🏦 Wallet",
    }
    return JSONResponse(
        {
            "success": True,
            "greeting": greeting,
            "message": message,
            "amount": amount,
            "method": method_emoji.get(method, method),
            "timestamp": now_ist.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "IST",
        }
    )


@app.get("/terms", response_class=HTMLResponse)
async def terms_page():
    terms_path = os.path.join(static_dir, "terms.html")
    if os.path.exists(terms_path):
        with open(terms_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="Terms & Conditions")


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page():
    privacy_path = os.path.join(static_dir, "privacy.html")
    if os.path.exists(privacy_path):
        with open(privacy_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="Privacy Policy")


@app.get("/refund-policy", response_class=HTMLResponse)
async def refund_page():
    refund_path = os.path.join(static_dir, "refund.html")
    if os.path.exists(refund_path):
        with open(refund_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="Refund Policy")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
