from fastapi import FastAPI,Request, File, UploadFile, HTTPException, Form,Body,BackgroundTasks,Depends,status,Response
from fastapi.responses import HTMLResponse, StreamingResponse,JSONResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import platform
import subprocess
import uuid
import os
import boto3
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
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware

from botocore.exceptions import ClientError
import pathlib
import shutil

from pypdf import PdfReader
import gc

from typing import Any, Dict, List, Optional
from pydantic import Field,BaseModel
from langchain_core.retrievers import BaseRetriever

from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor



#langchain
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from fireworks.client import Fireworks
from langchain_openai import ChatOpenAI  # For OpenAI-compatible LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain    #### for aws


from langchain_core.prompts import ChatPromptTemplate

#### for openai
# from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import hashlib
import numpy as np  

###
import secrets
from starlette.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


from app.pdf_operations import  (
    encrypt_pdf,
    remove_pdf_password,
    remove_background_rembg
    
)

from rembg import remove
from dotenv import load_dotenv
from colorama import Fore, Style, init
init() 
load_dotenv()
from fastapi.middleware.gzip import GZipMiddleware


from datetime import datetime

app = FastAPI()

# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=[
#         "recallmind.online",
#         "www.recallmind.online",
#         "*.ap-south-1.compute.amazonaws.com", 

#         "localhost",
#         "127.0.0.1",
#         "::1"
#     ]
# )

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
        "Pragma"
    ],
    expose_headers=["X-Auth-Error", "WWW-Authenticate"],
    max_age=600,
)

# Other middleware
app.add_middleware(GZipMiddleware, minimum_size=500)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


BLOCKED_PATTERNS = [
    re.compile(r'wp-admin', re.IGNORECASE),
    re.compile(r'wordpress', re.IGNORECASE),
    re.compile(r'phpmyadmin', re.IGNORECASE),
    re.compile(r'administrator', re.IGNORECASE),
    re.compile(r'mysql', re.IGNORECASE),
    re.compile(r'sql', re.IGNORECASE),
    re.compile(r'\.env', re.IGNORECASE),
    re.compile(r'config\.json', re.IGNORECASE),
    re.compile(r'backup', re.IGNORECASE),
    re.compile(r'\.git', re.IGNORECASE),
    re.compile(r'\.svn', re.IGNORECASE),
    re.compile(r'\.bak$', re.IGNORECASE),
    re.compile(r'\.log$', re.IGNORECASE),
    re.compile(r'db_dump', re.IGNORECASE),
    re.compile(r'adminer', re.IGNORECASE),
]

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    path = request.url.path
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(path):
            return JSONResponse(
                status_code=403,
                content={"detail": "Access forbidden"}
            )
    return await call_next(request)
@app.middleware("http")
async def mobile_compatibility_middleware(request: Request, call_next):
    """Add mobile compatibility headers"""
    response = await call_next(request)
    
    # Check if mobile
    user_agent = request.headers.get("user-agent", "").lower()
    is_mobile = any(term in user_agent for term in ['mobile', 'android', 'iphone', 'ipad'])
    
    if is_mobile:
        # Add mobile-specific headers
        response.headers["X-Mobile-Compatible"] = "true"
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    
    return response

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# app.add_middleware(
#     SessionMiddleware,
#     secret_key=os.getenv("SESSION_SECRET", secrets.token_urlsafe(32)),
#     session_cookie="session",
#     max_age=3600,
#     same_site="none",  # ← CRITICAL: Allows cross-origin on mobile
#     https_only=False,   # ← Set to False for HTTP testing
#     domain="recallmind.online"  # ← Remove the dot for main domain
# )
##### ehuggingface

from sentence_transformers import SentenceTransformer
import threading


# ------------------------------------------------------------------
# 1. Load the HF model once (thread-safe, CPU-only)
# ------------------------------------------------------------------
_HF_MODEL_LOCK = threading.Lock()
_HF_MODEL: Optional[SentenceTransformer] = None

def get_hf_model() -> SentenceTransformer:
    """Lazy-load the model the first time it is needed."""
    global _HF_MODEL
    with _HF_MODEL_LOCK:
        if _HF_MODEL is None:
            logger.info("Loading HuggingFace model sentence-transformers/all-MiniLM-L6-v2…")
            _HF_MODEL = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
            logger.info("HF model loaded (384-dim).")
        return _HF_MODEL

# ------------------------------------------------------------------
# 2. LangChain-compatible wrapper
# ------------------------------------------------------------------
class HFEmbeddings:
    """Simple LangChain-compatible wrapper"""
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
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# Mount static files
current_dir = pathlib.Path(__file__).parent.resolve()  # Resolves to E:\AI\RAG\MyProject\FASTAPI\vishnu_ai without docker\fastapi\app
static_dir = os.path.join(current_dir, "static")  # Points to E:\AI\RAG\MyProject\FASTAPI\vishnu_ai without docker\fastapi\app\static

app.mount("/static", StaticFiles(directory=static_dir), name="static")


# AWS S3 Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")             
GROQ_API_KEY= os.getenv("GROQ_API_KEY")             
FIREWORKS_API_KEY= os.getenv("FIREWORKS_API_KEY")             
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")   
USE_S3 = all([BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY])
CORRECT_PASSWORD_HASH = os.getenv("CORRECT_PASSWORD_HASH")
CLEANUP_DASHBOARD_PASSWORD = os.getenv("CLEANUP_DASHBOARD_PASSWORD")





s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
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
            region_name=AWS_REGION 
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



# PDF_S3_KEY = "https://vishnufastapi.s3.ap-south-1.amazonaws.com/daily_pdfs/resume.pdf"  # Ensure case consistency
# PDF_URL = "https://vishnufastapi.s3.ap-south-1.amazonaws.com/daily_pdfs/resume.pdf"
PDF_URL_TABULAR = "https://vishnufastapi.s3.ap-south-1.amazonaws.com/daily_pdfs/3.pdf"  # Your main PDF with tables
PDF_URL_NONTABULAR_1= "https://vishnufastapi.s3.ap-south-1.amazonaws.com/daily_pdfs/1.pdf"
PDF_URL_NONTABULAR_2 = "https://vishnufastapi.s3.ap-south-1.amazonaws.com/daily_pdfs/2.pdf"

           

            # List of all non-tabular PDFs
NONTABULAR_PDFS = [PDF_URL_NONTABULAR_1, PDF_URL_NONTABULAR_2]
# NONTABULAR_PDFS = [PDF_URL_NONTABULAR_1]


S3_PREFIX = "Amazingvideo/"



# ========== CENTRALIZED DIRECTORY CONFIGURATION ==========
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp_processing"
UPLOAD_DIR = TEMP_DIR / "uploads"
OUTPUT_DIR = TEMP_DIR / "output"
ESTIMATION_DIR = TEMP_DIR / "estimation"
PDFTOWORD = TEMP_DIR / "word"

# File operation limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PAGES = 5
orphan_age_seconds = 600  # 10 minutes

# Add these 2 lines with your other global variables
# current_ghostscript_process = None
# current_task_id = None

def setup_directories():
    """Create necessary directories on startup - EC2 COMPATIBLE"""
    directories = [TEMP_DIR, UPLOAD_DIR, OUTPUT_DIR, ESTIMATION_DIR, PDFTOWORD]
    
    for directory in directories:
        try:
            # ✅ Use os.makedirs for EC2 compatibility
            os.makedirs(str(directory), exist_ok=True)
            # ✅ Set proper permissions
            os.chmod(str(directory), 0o755)
            logger.info(f"✅ Directory ready: {directory}")
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {e}")
            # ✅ Critical - fail fast if directories can't be created
            raise





def download_from_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/pdf, */*"
        }
        
        # Increased timeout and added retry strategy
        session = requests.Session()
        retry_strategy = requests.packages.urllib3.util.retry.Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        response = session.get(url, headers=headers, timeout=15)  # Increased timeout
        response.raise_for_status()
        
        # Verify it's actually a PDF
        content_type = response.headers.get('content-type', '')
        if 'pdf' not in content_type.lower() and not response.content.startswith(b'%PDF'):
            logger.warning(f"⚠️ URL {url} might not be a PDF. Content-Type: {content_type}")
            
        return response.content
        
    except requests.exceptions.Timeout:
        logger.error(f"⏰ Timeout downloading PDF from {url}")
        raise HTTPException(status_code=500, detail=f"Timeout downloading PDF: {url}")
    except requests.exceptions.ConnectionError:
        logger.error(f"🔌 Connection error downloading PDF from {url}")
        raise HTTPException(status_code=500, detail=f"Connection error - cannot reach {url}")
    except Exception as e:
        logger.error(f"❌ Failed to download PDF from {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")
######################################################### NEW IMPLEMENTATION FOR RAG #########################################################


# Configurable lists for table detection
HEADER_KEYWORDS = [
    "Company", "Duration", 
]

TITLE_KEYWORDS = [
    "Work Experience", "Project or Work Experience"
]




def is_raw_table_text(text):
    """Detect table data rows - LESS AGGRESSIVE VERSION"""
    line = text.strip()
    if not line:
        return False

    # Split into words
    words = line.split()
    
    # REMOVED: starts_with_number check (too restrictive)
    # starts_with_number = bool(re.match(r'^\d+[\s\.]', line))  # REMOVE THIS
    
    # Count numbers (integers or decimals)
    number_count = len(re.findall(r'\b\d+\b|\d+\.\d+', line))
    
    # Check for tabular structure (4 or more words - increased from 3)
    has_tabular = len(words) >= 4  # Increased threshold
    
    # Exclude header-like keywords
    has_header_keywords = any(h.lower() in line.lower() for h in HEADER_KEYWORDS)
    
    # Check for table-specific patterns
    # has_table_patterns = bool(re.search(r'\b(NO\.|KM|NOS|Mtr)\b|\d{6,}', line))

    # MORE RELAXED table detection
    return (
        (number_count >= 3 and has_tabular and not has_header_keywords) or  # Increased to 2 numbers
        (has_tabular and not has_header_keywords)
    )



def is_table_title(line, title_keywords=TITLE_KEYWORDS):
    """Detect if a line is a potential table title."""
    if not line.strip() or len(line) > 100:
        return False
    if re.search(r'^\d+\s', line) or re.search(r'\s\d+\.\d+\s', line):
        return False
    return any(k.lower() in line.lower() for k in title_keywords)

def is_header_row(row_str, header_keywords=HEADER_KEYWORDS, min_matches=1):
    """Determine if a row is a header row based on keyword matching."""
    if not row_str.strip():
        return False

    lower_row = row_str.lower()
    matches = sum(1 for kw in header_keywords if kw.lower() in lower_row)

    if "project name" in lower_row or "duration" in lower_row or "company" in lower_row:
        return True
    return matches >= min_matches




def normalize_text(text):
    """Normalize text by removing extra spaces and standardizing."""
    return ' '.join(text.strip().split())

def is_substring_match(line, table_content_set):
    """Check if a line is an exact or partial match of table content."""
    norm_line = normalize_text(line)
    for table_content in table_content_set:
        if norm_line == table_content:
            return True
        # Strict substring match for longer lines
        if len(norm_line) > 15 and len(table_content) > 15:
            if norm_line.lower() in table_content.lower():
                return True
    return False


def extract_text_with_tables(pdf_bytes):
    """Enhanced PDF extraction with robust table detection"""
    full_text = []
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                table_content = set()
                filtered_lines = []
                cleaned_tables = []

                # 🆕 NEW: Extract table title from page text WITHOUT affecting table processing
                table_title = None
                if text:
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        # Look for the specific title pattern in page text
                        if ("project or work experience" in line.lower() and "year" in line.lower()):
                            table_title = line
                            break

                # Extract tables - KEEP ORIGINAL LOGIC UNCHANGED
                tables = page.extract_tables() or []
                for table in tables:
                    cleaned_table = []
                    header_row = None
                    title_row = None

                    for row in table:
                        row = ["" if cell is None else str(cell).strip() for cell in row]
                        row_str = ' '.join(cell for cell in row if cell)
                        if not row_str.strip():
                            continue

                        norm_row_str = normalize_text(row_str)
                        table_content.add(norm_row_str)

                        for cell in row:
                            if cell:
                                cell_lines = cell.split('\n')
                                for cell_line in cell_lines:
                                    if cell_line.strip():
                                        table_content.add(normalize_text(cell_line))

                        # KEEP ORIGINAL table title detection (but it won't find the main title)
                        if is_table_title(row_str, TITLE_KEYWORDS):
                            title_row = row_str
                            continue

                        if header_row is None and is_header_row(row_str, HEADER_KEYWORDS):
                            header_row = row
                            continue

                        cleaned_table.append(row)

                    if cleaned_table:
                        cleaned_tables.append((title_row, header_row, cleaned_table))

                # Process text lines - KEEP ORIGINAL LOGIC
                if text:
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        norm_line = normalize_text(line)
                        
                        # Skip lines that are table-related
                        if (norm_line in table_content or
                            is_substring_match(line, table_content) or
                            is_table_title(line, TITLE_KEYWORDS) or
                            is_header_row(line, HEADER_KEYWORDS) or
                            is_raw_table_text(line) or
                            line == table_title):  # 🆕 ONLY ADD: Skip the extracted title
                            continue

                        filtered_lines.append(line)

                # Add filtered text
                if filtered_lines:
                    full_text.append(f"--- PAGE {i} ---\n" + "\n".join(filtered_lines))

                # Add tables to full_text - 🆕 MINIMAL CHANGE: Use extracted title if available
                for title_row, header_row, cleaned_table in cleaned_tables:
                    try:
                        # KEEP ORIGINAL table processing logic
                        max_cols = max(len(row) for row in cleaned_table)
                        cleaned_table = [row + [""] * (max_cols - len(row)) for row in cleaned_table]

                        if header_row and cleaned_table:
                            df = pd.DataFrame(cleaned_table, columns=header_row)
                        else:
                            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0]) if len(cleaned_table) > 1 else pd.DataFrame(cleaned_table)

                        markdown_table = df.to_markdown(index=False)
                        
                        # 🆕 ONLY CHANGE: Prefer the extracted page title over table row title
                        final_title = table_title if table_title else title_row
                        if final_title:
                            full_text.append(f"\n{final_title}\n{markdown_table}\n")
                        else:
                            full_text.append(f"\nTABLE (Page {i}):\n{markdown_table}\n")
                        
                    except Exception as e:
                        full_text.append(f"\nTABLE_RAW (Page {i}):\n{str(cleaned_table)}\n")

        return "\n".join(full_text)

    except pdfplumber.exceptions.PDFSyntaxError as e:
        raise Exception(f"Invalid PDF format: {e}")
    except Exception as e:
        raise Exception(f"Error processing PDF: {e}")


# ====================== Enhanced Logging Functions ======================


def log_embedding_process(documents: List[Document], source: str):
    """Log the embedding process with chunk statistics"""
    logger.info(f"\n{'='*80}")
    logger.info(f"EMBEDDING PROCESS FOR: {source}")
    logger.info(f"Total documents to embed: {len(documents)}")
    
    # Calculate chunk statistics
    chunk_sizes = [len(doc.page_content) for doc in documents]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    min_size = min(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0
    
    logger.info(f"📊 CHUNK STATISTICS:")
    logger.info(f"  • Average size: {avg_size:.0f} chars")
    logger.info(f"  • Min size: {min_size} chars")
    logger.info(f"  • Max size: {max_size} chars")
    logger.info(f"  • Size distribution: {dict(pd.Series(chunk_sizes).describe().to_dict())}")
    
    logger.info(f"{'='*80}")
    
    for i, doc in enumerate(documents, 1):
        logger.info(f"\n🔤 DOCUMENT #{i} TO BE EMBEDDED:")
        logger.info(f"📁 Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"📄 Page: {doc.metadata.get('page_num', 'N/A')}")
        logger.info(f"🔢 Chunk: {doc.metadata.get('chunk_num', 'N/A')}/{doc.metadata.get('total_chunks_page', 'N/A')}")
        logger.info(f"📏 Content Length: {len(doc.page_content)} characters")
        logger.info(f"📝 Content Preview (full contents):\n{doc.page_content}...")
        logger.info(f"🏷️ Metadata: {doc.metadata}")
        logger.info(f"{'-'*60}")





# ====================== Enhanced Document Processing ======================
def post_process_retrieved_docs(docs, query):
    """Enhanced post-processing for work experience tables"""
    processed = []
    table_headers = ["Project Name", "Duration", "Company", "Project Description"]
    
    query_lower = query.lower().strip()


    for doc in docs:
        content = doc.page_content
        source = doc.metadata.get("source", "unknown")

        page_num = doc.metadata.get("page_num")  # ✅ CHANGED from "page"
        if page_num and isinstance(page_num, float):
            doc.metadata["page_num"] = int(page_num) 
        
        # Check if it's a work experience table from PDF 3
        is_work_table = (
            "3.pdf" in source and
            content.count("|") > 3 and 
            any(header in content for header in table_headers)
        )

        if is_work_table:
            # ENHANCE the table content for better LLM understanding
            doc.metadata["content_type"] = "work_experience_table"
            doc.metadata["table_type"] = "work_experience"
            doc.metadata["priority"] = "high"  # Mark as high priority
            
            # Add table markers for better LLM processing
            enhanced_content = f"WORK_EXPERIENCE_TABLE_START\n{content}\nWORK_EXPERIENCE_TABLE_END"
            doc.page_content = enhanced_content
            
            # logger.info(f"🎯 ENHANCED WORK EXPERIENCE TABLE FROM: {source}")
            
        else:
            doc.metadata["content_type"] = "text"

        processed.append(doc)

    return processed




def ensure_tabular_inclusion(docs, query, min_tabular=2):
    """Ensure relevant content is included based on query type"""

    query_lower = query.lower()
    
    # Check if query is about work/companies/experience
    is_work_query = any(keyword in query_lower for keyword in [
        'company', 'work', 'experience', 'job', 'project', 'started working',
        'career', 'professional', 'employment', 'when did', 'start date',
        'kei', 'larsen', 'toubro', 'vindhya', 'punj', 'gng', 'l&t'
    ])
    
    if is_work_query:
        # Get ALL tabular docs (work experience tables)
        tabular_docs = [d for d in docs if "3.pdf" in d.metadata.get("source", "")]
        other_docs = [d for d in docs if "3.pdf" not in d.metadata.get("source", "")]
        
        logger.info(f"🔍 WORK QUERY DETECTED: Found {len(tabular_docs)} tabular documents")
        
        # FORCE include ALL tabular docs for work queries
        final_docs = tabular_docs  # Include all work experience tables
        
        # Add top-scoring other docs to reach desired count
        remaining_slots = 5 - len(final_docs)
        if remaining_slots > 0:
            final_docs.extend(other_docs[:remaining_slots])
        
        logger.info(f"📊 Final docs for work query: {len(final_docs)} total, {len(tabular_docs)} from work experience table")
        
        return final_docs
    
    # For website queries (keep existing logic)
    elif any(keyword in query_lower for keyword in ['website', 'site', 'url', 'link', 'web']):
        website_docs = [d for d in docs if any(keyword in d.page_content.lower() for keyword in [
            'recallmind', 'parcelfile', 'vishnuji.com', 'website', 'file transfer'
        ])]
        other_docs = [d for d in docs if d not in website_docs]
        final_docs = website_docs[:3]
        remaining_slots = 5 - len(final_docs)
        if remaining_slots > 0:
            final_docs.extend(other_docs[:remaining_slots])
        return final_docs
    
    else:
        # For general queries, return top 5 but ensure table inclusion if present
        final_docs = docs[:5]
        # If there are tabular docs in top 10 but not in top 5, include at least one
        tabular_in_top_10 = [d for d in docs[:10] if "3.pdf" in d.metadata.get("source", "")]
        if tabular_in_top_10 and not any("3.pdf" in d.metadata.get("source", "") for d in final_docs):
            # Replace the lowest scoring doc with the highest scoring tabular doc
            final_docs[-1] = tabular_in_top_10[0]
        
        return final_docs




##### chromadb
class ChromaDBRetriever(BaseRetriever):
    vectorstore: Any = Field(...)
    search_kwargs: Dict = Field(default_factory=lambda: {"k": 10})
    
    def _get_relevant_documents(self, query: str, *, run_manager = None) -> List[Document]:
        try:
            start_time = time.time()
            
            # ✅ ONLY RETRIEVAL - no business logic
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, 
                k=self.search_kwargs["k"]
            )
            
            # Add scores to metadata
            docs = []
            for doc, score in docs_with_scores:
                doc.metadata["score"] = float(score)
                docs.append(doc)
            
            retrieval_time = time.time() - start_time
            logger.info(f"⚡ ChromaDB retrieval: {retrieval_time:.3f}s for {len(docs)} docs")
            
            
            return docs  # ✅ Return raw retrieved docs
            
        except Exception as e:
            logger.error(f"❌ ChromaDB retrieval failed: {e}")
            return []


##########NEW APPROACH #############
def process_non_tabular_pdf_complete(pdf_bytes, pdf_url, max_chunks_per_page=3, target_chunk_size=2500):
    """Process non-tabular PDF with 100% content preservation and strict page boundaries"""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            all_page_chunks = []
            total_pages = len(pdf.pages)
            
            logger.info(f"📄 Processing PDF: {pdf_url.split('/')[-1]} - {total_pages} pages")
            
            for page_num in range(1, total_pages + 1):
                page = pdf.pages[page_num - 1]
                
                # Extract COMPLETE raw text from page
                original_text = page.extract_text() or ""
                if not original_text.strip():
                    # Empty page - create placeholder
                    all_page_chunks.append({
                        'content': f"Page {page_num} - No extractable content",
                        'page_num': page_num,
                        'chunk_num': 1,
                        'total_chunks_page': 1,
                        'content_hash': hashlib.md5("empty".encode()).hexdigest()
                    })
                    continue
                
                logger.info(f"   📝 Page {page_num}: {len(original_text)} characters")
                
                # Process this page with guaranteed content preservation
                page_chunks = process_single_page_complete(
                    original_text, 
                    page_num, 
                    pdf_url,
                    max_chunks_per_page, 
                    target_chunk_size
                )
                
                # Verify 100% content preservation
                verify_page_content_preservation(original_text, page_chunks, page_num, pdf_url)
                
                # Add to results
                all_page_chunks.extend(page_chunks)
            
            # Final validation
            total_original_chars = sum(len(chunk['content']) for chunk in all_page_chunks)
            logger.info(f"✅ FINAL: {len(all_page_chunks)} chunks created from {total_pages} pages")
            
            return all_page_chunks
            
    except Exception as e:
        logger.error(f"❌ Error processing PDF {pdf_url}: {e}")
        return []

def process_single_page_complete(original_text, page_num, pdf_url, max_chunks, target_size):
    """Process a single page with 100% content preservation"""
    
    # Step 1: Clean but preserve ALL content
    cleaned_text = clean_text_preserve_all(original_text)
    
    # Step 2: Split into logical sections while preserving order
    sections = split_into_preserved_sections(cleaned_text)
    
    logger.info(f"      📑 Page {page_num} split into {len(sections)} sections")
    
    # Step 3: Create chunks that preserve ALL content
    chunks = create_content_preserving_chunks(
        sections, 
        max_chunks, 
        target_size,
        page_num
    )
    
    # Step 4: Create chunk objects with proper metadata
    page_chunks = []
    for chunk_num, chunk_content in enumerate(chunks, 1):
        page_chunks.append({
            'content': chunk_content,
            'page_num': page_num,
            'chunk_num': chunk_num,
            'total_chunks_page': len(chunks),
            'pdf_source': pdf_url.split('/')[-1],
            'content_hash': hashlib.md5(chunk_content.encode()).hexdigest()
        })
    
    return page_chunks

def clean_text_preserve_all(text):
    """Clean text while preserving 100% of the content"""
    if not text:
        return ""
    
    # Preserve ALL content - minimal cleaning only
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
    text = re.sub(r'\n+', '\n', text)        # Normalize line breaks
    
    return text.strip()

def split_into_preserved_sections(text):
    """Split text into sections while preserving ALL content - MINIMAL FIX"""
    if not text:
        return []
    
    sections = []
    
    # First, split by major section breaks (double newlines) BUT be smarter about it
    lines = text.split('\n')
    current_section = []
    
    for line in lines:
        line = line.strip()
        if not line:
            # Empty line - potential section break
            if current_section:
                # Check if we should preserve current section as is
                section_text = ' '.join(current_section)
                if should_preserve_section(current_section):
                    sections.append(section_text)
                    current_section = []
                else:
                    # Continue accumulating if it's a coherent paragraph
                    current_section.append("")  # Add empty line as separator
            continue
        
        # Add line to current section
        current_section.append(line)
    
    # Add the last section
    if current_section:
        section_text = ' '.join(current_section)
        sections.append(section_text)
    
    # If we have no sections (shouldn't happen), return the original text
    if not sections:
        return [text]
    
    return sections
def should_preserve_section(lines):
    """MINIMAL: Check if current lines form a section that should be preserved together"""
    if len(lines) < 2:
        return True  # Single line sections are fine
    
    # Check if first line looks like a heading and subsequent lines are content
    first_line = lines[0]
    is_heading_like = (
        first_line.endswith(':') or 
        (len(first_line) < 100 and first_line and first_line[0].isalnum())
    )
    
    # Check if we have bullet points or short content lines after heading
    if is_heading_like:
        subsequent_content = lines[1:]
        has_bullet_points = any(line.startswith('•') for line in subsequent_content)
        has_short_content = all(len(line) < 200 for line in subsequent_content)
        
        if has_bullet_points or (has_short_content and len(subsequent_content) <= 5):
            return False  # Don't break this section yet
    
    return True  # Default to breaking at empty lines

def create_content_preserving_chunks(sections, max_chunks, target_size, page_num):
    """Create chunks that guarantee 100% content preservation"""
    if not sections:
        return []
    
    # Calculate total content size
    total_chars = sum(len(section) for section in sections)
    
    # If total content is small, just return as one chunk
    if total_chars <= target_size:
        return [' '.join(sections)]
    
    # Calculate ideal chunk size
    ideal_chunk_size = max(target_size, total_chars // max_chunks)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for section in sections:
        section_size = len(section)
        
        # If this single section is larger than ideal size, we have to split it
        if section_size > ideal_chunk_size:
            # If we have current content, save it first
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split the large section
            large_chunks = split_large_section(section, ideal_chunk_size)
            chunks.extend(large_chunks)
            continue
        
        # If adding this section would exceed ideal size and we have content, start new chunk
        if (current_size + section_size > ideal_chunk_size and 
            current_chunk and 
            len(chunks) < max_chunks - 1):
            chunks.append(' '.join(current_chunk))
            current_chunk = [section]
            current_size = section_size
        else:
            current_chunk.append(section)
            current_size += section_size
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # If we have too many chunks, merge the smallest ones
    if len(chunks) > max_chunks:
        chunks = merge_small_chunks(chunks, max_chunks)
    
    # Final verification - ensure we didn't lose any content
    original_content = ' '.join(sections)
    chunked_content = ' '.join(chunks)
    
    if original_content != chunked_content:
        logger.warning(f"🚨 CONTENT LOSS DETECTED on page {page_num}!")
        # Emergency fallback - return the original text as single chunk
        return [original_content]
    
    return chunks[:max_chunks]  # Ensure we don't exceed max chunks

def split_large_section(section, max_size):
    """Split a very large section into smaller chunks"""
    if len(section) <= max_size:
        return [section]
    
    chunks = []
    words = section.split()
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        
        if current_size + word_size > max_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def merge_small_chunks(chunks, max_chunks):
    """Merge small chunks to stay within max_chunks limit"""
    if len(chunks) <= max_chunks:
        return chunks
    
    # Sort chunks by size and merge smallest ones first
    chunk_sizes = [(i, len(chunk)) for i, chunk in enumerate(chunks)]
    chunk_sizes.sort(key=lambda x: x[1])
    
    while len(chunks) > max_chunks and len(chunk_sizes) > 1:
        # Find two smallest chunks to merge
        smallest_idx1, size1 = chunk_sizes[0]
        smallest_idx2, size2 = chunk_sizes[1]
        
        # Merge them
        merged = chunks[smallest_idx1] + " " + chunks[smallest_idx2]
        
        # Replace first chunk with merged content, remove second
        chunks[smallest_idx1] = merged
        chunks.pop(smallest_idx2)
        
        # Recalculate sizes
        chunk_sizes = [(i, len(chunk)) for i, chunk in enumerate(chunks)]
        chunk_sizes.sort(key=lambda x: x[1])
    
    return chunks

def verify_page_content_preservation(original_text, page_chunks, page_num, pdf_url):
    """Verify that 100% of original content is preserved"""
    original_normalized = ' '.join(original_text.split())
    all_chunks_normalized = ' '.join(' '.join(chunk['content'].split()) for chunk in page_chunks)
    
    if original_normalized != all_chunks_normalized:
        # Find exactly what's missing
        original_words = set(original_normalized.split())
        chunk_words = set(all_chunks_normalized.split())
        missing_words = original_words - chunk_words
        
        if missing_words:
            logger.error(f"🚨 CONTENT LOSS on {pdf_url.split('/')[-1]} Page {page_num}:")
            logger.error(f"   Missing {len(missing_words)} words: {list(missing_words)[:10]}...")
            
            # Find missing sentences
            original_sentences = re.split(r'[.!?]+', original_normalized)
            for sentence in original_sentences:
                sentence = sentence.strip()
                if (sentence and 
                    len(sentence) > 20 and 
                    sentence not in all_chunks_normalized):
                    logger.error(f"   Missing sentence: {sentence[:100]}...")
        
        coverage = len(chunk_words.intersection(original_words)) / len(original_words) * 100
        logger.error(f"   Coverage: {coverage:.1f}% - NEEDS FIXING!")
        return False
    else:
        logger.info(f"      ✅ Page {page_num}: 100% content preserved")
        return True



def run_comprehensive_content_audit():
    """Run a comprehensive audit of all PDF content coverage"""
    logger.info("\n" + "="*80)
    logger.info("🔍 COMPREHENSIVE CONTENT COVERAGE AUDIT")
    logger.info("="*80)
    
    all_pdfs = NONTABULAR_PDFS + [PDF_URL_TABULAR]
    
    for pdf_url in all_pdfs:
        try:
            pdf_name = pdf_url.split('/')[-1]
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
                    
                    # Process this page individually
                    page_chunks = process_single_page_complete(
                        original_text, page_num, pdf_url, 3, 2000
                    )
                    
                    page_processed_chars = sum(len(chunk['content']) for chunk in page_chunks)
                    total_processed_chars += page_processed_chars
                    
                    coverage = (page_processed_chars / len(original_text)) * 100 if original_text else 100
                    
                    if coverage < 99.9:
                        missing_pages.append(page_num)
                        logger.warning(f"   Page {page_num}: {len(original_text)} → {page_processed_chars} chars ({coverage:.1f}%) ❌")
                    else:
                        logger.info(f"   Page {page_num}: {len(original_text)} → {page_processed_chars} chars ({coverage:.1f}%) ✅")
                
                overall_coverage = (total_processed_chars / total_original_chars) * 100 if total_original_chars else 100
                
                if missing_pages:
                    logger.error(f"🚨 {pdf_name}: {overall_coverage:.1f}% overall - MISSING PAGES: {missing_pages}")
                else:
                    logger.info(f"✅ {pdf_name}: {overall_coverage:.1f}% overall - ALL CONTENT PRESERVED")
                
        except Exception as e:
            logger.error(f"❌ Failed to audit {pdf_url}: {e}")

# Run this after initialization

######################

def initialize_vectorstore():
    try:
        logger.info("🚀 Initializing ChromaDB with HuggingFace embeddings...")
        
        # Use HuggingFace embeddings instead of OpenAI
        embeddings = HFEmbeddings()
        
        # Define persist directory
        persist_dir = "./chroma_db"
        os.makedirs(persist_dir, exist_ok=True)
        
        # Check if ChromaDB already exists and has data
        collection_exists = False
        documents_count = 0
        
        try:
            existing_vectorstore = Chroma(
                collection_name="vishnu_ai_docs",
                embedding_function=embeddings,
                persist_directory=persist_dir
            )
            
            documents_count = existing_vectorstore._collection.count()
            collection_exists = documents_count > 0
            
            if collection_exists:
                logger.info(f"📚 Existing ChromaDB found with {documents_count} documents")
                return existing_vectorstore, embeddings
            else:
                logger.info("📭 ChromaDB exists but is empty, will recreate...")
                
        except Exception as e:
            logger.info(f"🆕 No existing ChromaDB found: {e}")
            collection_exists = False
        
        # If we reach here, need to create new ChromaDB
        logger.info("📦 Creating new ChromaDB collection...")
        
        vectorstore = Chroma(
            collection_name="vishnu_ai_docs",
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        
        # Process documents only if collection is empty
        all_documents = []
        
        # Try to download and process PDFs with graceful fallback
        pdf_urls_to_try = [PDF_URL_TABULAR] + NONTABULAR_PDFS
        successful_downloads = 0
        
        for pdf_url in pdf_urls_to_try:
            try:
                logger.info(f"📥 Attempting to download: {pdf_url}")
                pdf_bytes = download_from_url(pdf_url)
                pdf_name = pdf_url.split('/')[-1]
                successful_downloads += 1
                
                if pdf_url == PDF_URL_TABULAR:
                    # Process tabular PDF
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
                            "total_chunks_page": 1
                        }
                    )
                    all_documents.append(tabular_doc)
                    logger.info(f"✅ Tabular PDF processed: {len(tabular_text)} characters")
                    
                else:
                    # Process non-tabular PDF
                    logger.info(f"📝 Processing: {pdf_name}")
                    page_chunks = process_non_tabular_pdf_complete(
                        pdf_bytes, pdf_url, max_chunks_per_page=3, target_chunk_size=2500
                    )
                    
                    logger.info(f"📑 Created {len(page_chunks)} chunks from {pdf_name}")
                    
                    for chunk_info in page_chunks:
                        chunk_doc = Document(
                            page_content=chunk_info['content'],
                            metadata={
                                "source": pdf_url,
                                "content_type": "text_heavy",
                                "document_type": "nontabular",
                                "page_num": chunk_info['page_num'],
                                "chunk_num": chunk_info['chunk_num'],
                                "total_chunks_page": chunk_info['total_chunks_page'],
                                "pdf_source": chunk_info['pdf_source'],
                                "content_hash": chunk_info['content_hash']
                            }
                        )
                        all_documents.append(chunk_doc)
                    
                    pages_covered = set(chunk['page_num'] for chunk in page_chunks)
                    logger.info(f"📊 {pdf_name}: {len(pages_covered)} pages → {len(page_chunks)} chunks")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to download/process {pdf_url}: {e}")
                continue
        
        # Check if we have any documents to add
        if not all_documents:
            logger.error("❌ No PDFs could be downloaded. Creating empty ChromaDB with fallback data.")
            
            # Add minimal fallback document
            fallback_doc = Document(
                page_content="Vishnu Kumar - Electrical Engineer with 12 years experience. Worked at L&T, KEI Industries, Punj Lloyd. Skills: substation execution, project management, quality assurance.",
                metadata={
                    "source": "fallback",
                    "content_type": "text",
                    "document_type": "fallback", 
                    "page_num": 1,
                    "chunk_num": 1
                }
            )
            all_documents.append(fallback_doc)
        
        # Add documents to ChromaDB
        logger.info(f"📤 Adding {len(all_documents)} documents to ChromaDB ({successful_downloads}/{len(pdf_urls_to_try)} PDFs successful)")
        
        if all_documents:
            # HuggingFace embeddings work better with smaller batches
            batch_size = 20  # Reduced from 50 for better memory management
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]
                vectorstore.add_documents(batch)
                logger.info(f"✅ Added batch {i//batch_size + 1}/{(len(all_documents)-1)//batch_size + 1}")
                
                # Small delay between batches to prevent memory issues
                if i + batch_size < len(all_documents):
                    time.sleep(0.5)
        
        final_count = vectorstore._collection.count()
        logger.info(f"🎉 ChromaDB initialization completed with {final_count} documents")
        
        return vectorstore, embeddings
        
    except Exception as e:
        logger.error(f"❌ ChromaDB initialization failed: {e}", exc_info=True)
        
        # Emergency fallback - return empty ChromaDB
        logger.info("🆘 Creating emergency fallback ChromaDB...")
        try:
            embeddings = HFEmbeddings()
            
            vectorstore = Chroma(
                collection_name="vishnu_ai_docs_fallback",
                embedding_function=embeddings,
                persist_directory="./chroma_db_fallback"
            )
            
            # Add minimal document
            fallback_doc = Document(
                page_content="System is initializing. Please try chat functionality.",
                metadata={"source": "fallback", "error": True}
            )
            vectorstore.add_documents([fallback_doc])
            
            return vectorstore, embeddings
            
        except Exception as fallback_error:
            logger.critical(f"💥 Even fallback failed: {fallback_error}")
            raise





def verify_embeddings(embeddings_list):
    """Verify embeddings are valid"""
    if not embeddings_list:
        raise ValueError("No embeddings generated")
    
    for i, embedding in enumerate(embeddings_list):
        if len(embedding) != 384:
            raise ValueError(f"Embedding {i} has wrong dimension: {len(embedding)}")
        
        # Check if embedding is all zeros or contains NaN
        if all(v == 0 for v in embedding) or any(np.isnan(v) for v in embedding):
            raise ValueError(f"Invalid embedding at index {i}")
    
    return True

# def get_llm():
#     return ChatOpenAI(
#         model="accounts/fireworks/models/llama-v3p3-70b-instruct",  # Much faster 7B model
#         base_url="https://api.fireworks.ai/inference/v1",
#         api_key=os.getenv("FIREWORKS_API_KEY"),
#         temperature=0.1,
#         max_tokens=1256,  # Short responses for simple questions
#         timeout=10,
#     )

# def get_llm():
#     return ChatOpenAI(
#         model="accounts/fireworks/models/gpt-oss-20b",  # Direct Fireworks model
#         base_url="https://api.fireworks.ai/inference/v1",
#         api_key=os.getenv("FIREWORKS_API_KEY"),  # Use Fireworks key, not HF token
#         temperature=0.2,
#         max_tokens=2024,
#         timeout=10,
#     )

# def get_llm():
#     return ChatOpenAI(
#         model="openai/gpt-oss-20b:fireworks-ai",  # Use :fireworks-ai for speed; alternatives: :cerebras
#         base_url="https://router.huggingface.co/v1",
#         api_key=os.getenv("HF_TOKEN"),
#         temperature=0.1,
#         max_tokens=2024,
#         timeout=10,
#     )



def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
       # model="gemini-2.5-pro",
        temperature=0.2,
        max_tokens=2024,
        timeout=None,
        api_key=GOOGLE_API_KEY
    )

# def get_llm():
#     # Try Groq first (fastest), fallback to Gemini
    
#     return ChatGroq(
#         # model="llama-3.1-8b-instant",  # Fastest Groq model
#         model="openai/gpt-oss-20b",
#         temperature=0.1,
#         max_tokens=2024,
#         timeout=10,
#         groq_api_key=os.getenv("GROQ_API_KEY")  # Add to your .env
#     )



retriever = None
llm = None
thread_pool = ThreadPoolExecutor(max_workers=4)



# ====================== Startup Event ======================
@app.on_event("startup")
async def startup_event():

    logger.info("🚀 Startup initiated")
    # Disable ChromaDB telemetry
    # import os
    # os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    #     # Disable pdfplumber warnings if needed
    # import logging
    # pdfplumber_logger = logging.getLogger('pdfplumber')
    # pdfplumber_logger.setLevel(logging.ERROR)

    global retriever, llm, vectorstore
    logger.info("🚀 Starting AI services...")
    setup_directories()
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            vectorstore, embeddings = initialize_vectorstore()
            
            retriever = ChromaDBRetriever(
                vectorstore=vectorstore,
                search_kwargs={"k": 10}
            )
            
            llm = get_llm()
            
            # Test retrieval
            test_start = time.time()
            test_docs = retriever.invoke("test")
            test_time = time.time() - test_start
            
            logger.info(f"✅ AI services initialized successfully! Test retrieval: {test_time:.3f}s")
            break  # Success, exit retry loop
            
        except Exception as e:
            logger.error(f"❌ Startup attempt {attempt + 1}/{max_retries + 1} failed: {e}")
            
            if attempt < max_retries:
                logger.info(f"🔄 Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.critical("💥 All startup attempts failed. Using fallback mode.")
                # Ultimate fallback
                class FallbackRetriever(BaseRetriever):
                            def _get_relevant_documents(self, query: str, *, run_manager = None) -> List[Document]:
                                return [Document(
                                    page_content="System is experiencing technical difficulties. Please try again later.",
                                    metadata={"source": "fallback", "error": True}
                                )]
                        
                retriever = FallbackRetriever()
                llm = get_llm()
                # Ensure vectorstore exists to prevent None errors
                vectorstore = None



@app.get("/", response_class=HTMLResponse)
async def serve_index():
   
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


#######################################   login

CLEANUP_DASHBOARD_PASSWORD= os.getenv("CLEANUP_DASHBOARD_PASSWORD")
# if not CLEANUP_DASHBOARD_PASSWORD:
#     raise RuntimeError("CLEANUP_DASHBOARD_PASSWORD is required in .env")

security = HTTPBasic()

active_sessions = {}

def create_session_token(username: str) -> str:
    """Create a secure session token"""
    token_data = f"{username}{time.time()}{secrets.token_urlsafe(16)}"
    return hashlib.sha256(token_data.encode()).hexdigest()

def verify_session_token(token: str) -> bool:
    """Verify if session token is valid"""
    if token in active_sessions:
        session_data = active_sessions[token]
        # Check if session is not expired (1 hour)
        if time.time() - session_data.get("login_time", 0) < 3600:
            return True
        else:
            # Remove expired session
            del active_sessions[token]
    return False

def check_auth(request: Request):
    """Mobile-optimized authentication"""
    # Try multiple ways to get the token
    token = None
    
    # 1. Check query parameter (PRIMARY for mobile)
    token = request.query_params.get("token")
    
    # 2. Check Authorization header
    if not token:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
    
    # 3. Check cookies (least reliable on mobile)
    if not token:
        token = request.cookies.get("session_token")
    
    # 4. Special mobile handling - if we have a mobile flag but no token yet
    user_agent = request.headers.get("user-agent", "").lower()
    is_mobile = any(term in user_agent for term in ['mobile', 'android', 'iphone', 'ipad'])
    
    if is_mobile and not token:
        # For mobile, we might be in a redirect chain - allow access to cleanup page
        # The JavaScript will handle authentication
        logger.info(f"Mobile access to {request.url.path} - allowing page load")
        return True
    
    if not token or not verify_session_token(token):
        raise HTTPException(
            status_code=401, 
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return True
@app.get("/mobile-test")
async def mobile_test(request: Request):
    """Debug endpoint to check mobile authentication"""
    user_agent = request.headers.get("user-agent", "")
    is_mobile = any(term in user_agent.lower() for term in ['mobile', 'android', 'iphone', 'ipad'])
    
    token = request.cookies.get("session_token") or request.query_params.get("token")
    
    return {
        "is_mobile": is_mobile,
        "user_agent": user_agent,
        "has_token": bool(token),
        "token_valid": verify_session_token(token) if token else False,
        "cookies_received": dict(request.cookies),
        "query_params": dict(request.query_params)
    }
@app.post("/api/login")
async def api_login(
    credentials: dict, 
    response: Response, 
    request: Request,
    background_tasks: BackgroundTasks
):
    username = credentials.get("username", "").strip()
    password = credentials.get("password", "").strip()

    if username == "admin" and password == os.getenv("CLEANUP_DASHBOARD_PASSWORD"):
        # Create session token
        session_token = create_session_token(username)
        active_sessions[session_token] = {
            "username": username,
            "login_time": time.time(),
            "authenticated": True
        }
        
        # Detect mobile
        user_agent = request.headers.get("user-agent", "").lower()
        is_mobile = any(term in user_agent for term in ['mobile', 'android', 'iphone', 'ipad'])
        
        # For MOBILE: Server-side redirect with token in URL
        if is_mobile:
            redirect_url = f"/cleanup?token={session_token}&mobile=true&ts={int(time.time())}"
            
            # Set cookie anyway (might work on some mobile browsers)
            response = RedirectResponse(url=redirect_url, status_code=303)
            response.set_cookie(
                key="session_token",
                value=session_token,
                max_age=3600,
                httponly=False,
                samesite="lax",
                secure=False
            )
            return response
        else:
            # For DESKTOP: Return JSON response as before
            response_data = {
                "status": "success", 
                "token": session_token,
                "message": "Login successful"
            }
            
            response = JSONResponse(response_data)
            response.set_cookie(
                key="session_token",
                value=session_token,
                max_age=3600,
                httponly=False,
                samesite="lax", 
                secure=False
            )
            return response
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
@app.get("/cleanup", response_class=HTMLResponse)
async def cleanup_dashboard(request: Request):
    try:
        check_auth(request)
    except HTTPException as e:
        # If not authenticated, still serve the page but it will handle auth via JavaScript
        logger.info(f"Serving cleanup page without auth - JavaScript will handle authentication")
    
    # Always serve the page - let frontend handle the auth state
    cleanup_path = os.path.join(static_dir, "cleanup.html")
    with open(cleanup_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/cookie-test")
async def cookie_test(request: Request, response: Response):
    """Test if cookies work on mobile"""
    # Set a test cookie
    response = JSONResponse({
        "message": "Cookie test",
        "cookies_received": request.cookies,
        "headers": dict(request.headers)
    })
    
    response.set_cookie(
        key="test_cookie",
        value="working_" + str(time.time()),
        max_age=3600,
        httponly=False,  # Make it accessible to JavaScript
        samesite="none",
        secure=False,

    )
    
    return response
@app.get("/cleanup-logs")
async def get_cleanup_logs(request: Request):
    check_auth(request)
    
    try:
        log_path = "/home/ubuntu/cron_cleanup.log"
        if not os.path.exists(log_path):
            return {
                "logs": "No logs found yet. Cronjob may not have run.",
                "last_updated": None,
                "file_size": "0 KB",
                "line_count": 0,
                "recent_entries": []
            }
        
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        stat = os.stat(log_path)
        last_updated = stat.st_mtime
        
        # Get recent entries (last 10 lines)
        lines = log_content.strip().split('\n')
        recent_entries = lines[-10:] if lines else []
        
        return {
            "logs": log_content,
            "last_updated": last_updated,
            "file_size": f"{stat.st_size / 1024:.2f} KB",
            "line_count": len(lines),
            "recent_entries": recent_entries,
            "total_runs": len([line for line in lines if "Starting scheduled cleanup" in line])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")

@app.get("/cleanup-logs-page")
async def get_cleanup_logs_page():
    """Serve a dedicated page for viewing cleanup logs"""
    log_path = "/home/ubuntu/cron_cleanup.log"
    
    if not os.path.exists(log_path):
        log_content = "No logs found yet. Cronjob may not have run."
    else:
        with open(log_path, 'r') as f:
            log_content = f.read()
    

    lines = log_content.strip().split('\n')
    lines.reverse()  # This puts latest entries at top
    log_content = '\n'.join(lines)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cleanup Logs</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1200px; margin: 0 auto; 
                background: white; padding: 30px; 
                border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            h1 {{ color: #333; margin-bottom: 20px; text-align: center; }}
            .controls {{ 
                display: flex; gap: 10px; margin-bottom: 20px; 
                justify-content: center; flex-wrap: wrap;
            }}
            .btn {{ 
                padding: 10px 20px; border: none; border-radius: 8px;
                font-weight: 600; cursor: pointer; transition: all 0.3s ease;
            }}
            .btn-primary {{ background: #007bff; color: white; }}
            .btn-primary:hover {{ background: #0056b3; transform: translateY(-2px); }}
            .btn-secondary {{ background: #6c757d; color: white; }}
            .btn-secondary:hover {{ background: #545b62; transform: translateY(-2px); }}
            .log-container {{ 
                background: #1e1e1e; color: #00ff00; padding: 20px;
                border-radius: 8px; font-family: 'Courier New', monospace;
                max-height: 600px; overflow-y: auto; white-space: pre-wrap;
            }}
            .log-entry {{ 
                margin: 5px 0; padding: 5px; 
                border-left: 3px solid transparent;
            }}
            .log-info {{ border-left-color: #17a2b8; }}
            .log-success {{ border-left-color: #28a745; }}
            .log-error {{ border-left-color: #dc3545; }}
            .log-warning {{ border-left-color: #ffc107; }}
            .stats {{ 
                display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px; margin-bottom: 20px;
            }}
            .stat-card {{ 
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white; padding: 15px; border-radius: 10px; text-align: center;
            }}
            .stat-number {{ font-size: 1.5em; font-weight: bold; margin-bottom: 5px; }}
            .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧹 Cron Cleanup Logs</h1>
            
            <div class="controls">
                <button class="btn btn-primary" onclick="refreshLogs()">🔄 Refresh</button>
                <button class="btn btn-secondary" onclick="goToDashboard()">📊 Dashboard</button>
                <button class="btn btn-secondary" onclick="clearLogs()">🗑️ Clear Logs</button>
                <button class="btn btn-secondary" onclick="downloadLogs()">📥 Download Logs</button>
            </div>
            
            <div id="logStats" class="stats">
                <!-- Stats will be populated by JavaScript -->
            </div>
            
            <div class="log-container" id="logContent">
                {log_content}
            </div>
        </div>
        
        <script>
            function refreshLogs() {{
                location.reload();
            }}
            
            function goToDashboard() {{
                window.location.href = '/cleanup';
            }}
            
            async function clearLogs() {{
                if (confirm('Are you sure you want to clear all logs?')) {{
                    const response = await fetch('/clear-cleanup-logs', {{ method: 'POST' }});
                    if (response.ok) {{
                        location.reload();
                    }} else {{
                        alert('Failed to clear logs');
                    }}
                }}
            }}
            
            function downloadLogs() {{
                const logContent = document.getElementById('logContent').textContent;
                const blob = new Blob([logContent], {{ type: 'text/plain' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'cleanup_logs.txt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }}
            
            // Color code logs
            function colorCodeLogs() {{
                const logContainer = document.getElementById('logContent');
                const lines = logContainer.innerHTML.split('\\n');
                const coloredLines = lines.map(line => {{
                    let className = 'log-entry log-info';
                    if (line.includes('ERROR') || line.includes('❌') || line.includes('💥')) 
                        className = 'log-entry log-error';
                    else if (line.includes('SUCCESS') || line.includes('✅') || line.includes('🎯')) 
                        className = 'log-entry log-success';
                    else if (line.includes('WARNING') || line.includes('⚠️')) 
                        className = 'log-entry log-warning';
                    
                    return `<div class="${{className}}">${{line}}</div>`;
                }});
                logContainer.innerHTML = coloredLines.join('');
            }}
            
            // Load stats
            async function loadStats() {{
                try {{
                    const response = await fetch('/cleanup-logs');
                    const data = await response.json();
                    
                    const statsHtml = `
                        <div class="stat-card">
                            <div class="stat-number">${{data.total_runs || 0}}</div>
                            <div class="stat-label">Total Runs</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${{data.line_count || 0}}</div>
                            <div class="stat-label">Log Entries</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${{data.file_size || '0 KB'}}</div>
                            <div class="stat-label">Log Size</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${{data.last_updated ? new Date(data.last_updated * 1000).toLocaleDateString() : 'Never'}}</div>
                            <div class="stat-label">Last Updated</div>
                        </div>
                    `;
                    
                    document.getElementById('logStats').innerHTML = statsHtml;
                }} catch (error) {{
                    console.error('Error loading stats:', error);
                }}
            }}
            
            document.addEventListener('DOMContentLoaded', function() {{
                colorCodeLogs();
                loadStats();
            }});
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.post("/test-cleanup")
async def test_cleanup(request: Request):
    check_auth(request)  # ← Use the new auth
    
    try:
        logger.info("🧹 Manual cleanup triggered via /test-cleanup")
        cleanup_orphaned_files()
        return {
            "message": "Cleanup completed successfully", 
            "timestamp": time.time(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"❌ Manual cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
@app.get("/session-debug")
async def session_debug(request: Request):
    """Debug endpoint to check cookie status"""
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
        "is_mobile": "Mobile" in request.headers.get("user-agent", "")
    }

@app.get("/cleanup-status")
async def get_cleanup_status(request: Request):
    check_auth(request)
    
    try:
        current_time = time.time()
        directories = {
            'uploads': BASE_DIR / "app" / "temp_processing" / "uploads",
            'output': BASE_DIR / "app" / "temp_processing" / "output", 
            'estimation': BASE_DIR / "app" / "temp_processing" / "estimation",
            'word': BASE_DIR / "app" / "temp_processing" / "word",
            'temp_processing': BASE_DIR / "app" / "temp_processing"
        }
        
        stats = {
            "last_cleanup_time": 0,
            "current_time": current_time,
            "next_cleanup_in": 900,
            "directories": {},
            "total_files": 0,
            "old_files": 0,
            "total_size": 0
        }
        
        for dir_name, dir_path in directories.items():
            if dir_path.exists():
                total_files = 0
                old_files = 0
                total_size = 0
                
                try:
                    for file_path in dir_path.iterdir():
                        if file_path.is_file():
                            total_files += 1
                            file_stat = file_path.stat()
                            total_size += file_stat.st_size
                            if file_stat.st_mtime < (current_time - 900):
                                old_files += 1
                except Exception as e:
                    logger.error(f"Error scanning directory {dir_name}: {str(e)}")
                
                stats["directories"][dir_name] = {
                    "total_files": total_files,
                    "old_files": old_files,
                    "total_size": total_size,
                    "exists": True
                }
                
                stats["total_files"] += total_files
                stats["old_files"] += old_files
                stats["total_size"] += total_size
            else:
                stats["directories"][dir_name] = {
                    "total_files": 0,
                    "old_files": 0,
                    "total_size": 0,
                    "exists": False
                }
        
        return stats
    except Exception as e:
        logger.error(f"Cleanup status check failed: {e}")
        return {"error": str(e)}








@app.post("/api/logout")
async def api_logout(request: Request, response: Response):
    token = request.cookies.get("session_token") or request.headers.get("authorization", "").replace("Bearer ", "")
    
    if token in active_sessions:
        del active_sessions[token]
    
    # Clear cookie
    response = JSONResponse({"status": "success", "message": "Logged out"})
    response.delete_cookie("session_token")
    
    # Also clear localStorage on client side
    return response



################### backend log
@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Login page for mobile devices"""
    login_path = os.path.join(static_dir, "login.html")
    with open(login_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())




BASE_DIR = Path("/home/ubuntu/vishnufastapi")
BACKEND_LOG_PATH = BASE_DIR / "backend.log"

@app.get("/backend-logs")
async def get_backend_logs(request: Request):
    """Get backend.log file information and recent entries"""
    check_auth(request)  # ← USE THE NEW AUTH
    
    try:
        if not BACKEND_LOG_PATH.exists():
            return {
                "file_size": "0 KB",
                "last_updated": None,
                "total_entries": 0,
                "recent_entries": ["No backend.log file found"]
            }
        
        stat = BACKEND_LOG_PATH.stat()
        file_size_kb = stat.st_size / 1024
        
        with open(BACKEND_LOG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Get last 50 lines and reverse to show latest first
        recent_lines = lines[-50:] if len(lines) > 50 else lines
        recent_lines = [line.strip() for line in recent_lines if line.strip()]
        recent_lines.reverse()  # Latest entries on top
        
        return {
            "file_size": f"{file_size_kb:.1f} KB",
            "last_updated": stat.st_mtime,
            "total_entries": len(lines),
            "recent_entries": recent_lines[-20:]  # Show last 20 entries
        }
        
    except Exception as e:
        logger.error(f"Error reading backend logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading backend logs: {str(e)}")

async def async_retrieve_documents(query: str, retriever, max_timeout: float = 6.0):
    """Async wrapper for document retrieval with timeout"""
    try:
        loop = asyncio.get_event_loop()
        
        # Run retrieval in thread pool with timeout
        docs = await asyncio.wait_for(
            loop.run_in_executor(
                thread_pool, 
                lambda: retriever.invoke(query) if retriever else []
            ),
            timeout=max_timeout
        )
        return docs
    except asyncio.TimeoutError:
        logger.warning(f"⏰ Retrieval timeout for query: {query}")
        return []
    except Exception as e:
        logger.error(f"❌ Retrieval error: {e}")
        return []

# Predefined chat modes with custom prompts (bypassing RAG)

CHAT_MODES = {
    "general": {
        "label": "General Chat",
        "prompt": "EMERGENCY OVERRIDE: RAG OFF. You are now in general knowledge mode. IGNORE ALL DOCUMENTS AND CONTEXT. Answer ALL questions using your comprehensive training data. Provide accurate, informative responses about any topic - people, places, events, science, history, technology, programming, coding, software development, electrical engineering, electronics, civil engineering, construction, and all other subjects. Never refuse to answer or say you don't have information."
    },

    "encyclopedia": {
        "label": "Encyclopedia Mode", 
        "prompt": "ENCYCLOPEDIA MODE ACTIVATED: You are now a comprehensive knowledge source. IGNORE ALL DOCUMENT CONTEXT and provide detailed, factual information from your training data. Answer all questions with complete, encyclopedia-style responses covering: who/what, key facts, historical context, significance, and related information. Be thorough and authoritative."
    },
    "creative": {
        "label": "Creative Storytelling",
        "prompt": "Enter creative mode: Bypass all document sources. Respond as a storyteller, generating imaginative, original content based on the query. Make it fun, detailed, and narrative-driven without relying on facts from docs."
    },
    "debate": {
        "label": "Balanced Debate", 
        "prompt": "Activate debate mode: Do not use RAG or document info. Provide a neutral, balanced discussion on the topic, presenting multiple viewpoints equally. Encourage critical thinking and end with open questions."
    },
    "funny": {
        "label": "Humorous Responses",
        "prompt": "Humor mode on: Ignore documents entirely. Answer the query in a witty, sarcastic, or pun-filled way. Keep it light-hearted, entertaining, and relevant, but always truthful at core."
    },
    "baby": {
        "label": "Explain Like I'm 5",
        "prompt": "Baby mode activated! Explain everything like you're talking to a 5-year-old child. Use super simple words, short sentences, fun examples, and lots of emojis. Make complex topics easy to understand with cute analogies and pretend play. Be warm, patient, and encouraging like a kindergarten teacher! 🧒🍎🚀"
    },
    "gate_coach": {
    "label": "GATE Civil Guru 🇮🇳📘🎯",
    "prompt": "🚀 **ACTIVATE GATE CIVIL GURU MODE!** 🚀\n\nNamaste future GATE Topper! 🙏🎓 I'm your **Civil Engineering Buddy from India**, who turns tough GATE questions into *easy-peasy desi-style learning!* 😄💪\n\n**🧠 MY PROBLEM-SOLVING FORMULA:**\n1. **🤔 UNDERSTAND** – 'Dekhte hain bhai, yeh sawaal kis type ka hai?'\n2. **📏 FIND** – 'Kaunsa IS code ya CPWD reference lagega?'\n3. **🔧 SOLVE** – 'Step by step, bina tension ke!'\n4. **✅ CHECK** – 'Answer sahi lag raha hai? Logical bhi?'\n5. **🎓 EXPLAIN** – 'Ab samjhaate hain simple words mein – Indian site pe kaam jaise!' 🏗️\n\n**📚 ALL CIVIL ENGINEERING TOPICS (India Edition):**\n- 🏛️ **Building Design & RCC** – IS 456:2000 style concrete power!\n- 🧱 **Steel Structures** – IS 800:2007 ke saath strong as steel! 💪\n- 🌋 **Soil Mechanics & Foundation** – IS 6403, IS 2911... Mitti ka full story! 🪣\n- 💧 **Fluid Mechanics & Hydrology** – IS 4985, IS 3370... Flow like Ganga, think like Einstein! 🌊\n- 🌿 **Environmental Engineering** – IS 10500 for clean paani 💧 and CPHEEO rules!\n- 🛣️ **Transportation Engineering** – IRC standards for smooth desi roads! 🛣️🚗\n- 📐 **Surveying & Geomatics** – IS 14962 + Indian tricks for leveling and mapping! 🧭\n- 🧮 **Engineering Mathematics** – Chill! Numbers won’t scare you here 😎\n\n**🧱 HOW I HELP YOU:**\n✨ **IS + CPWD READY** – Every answer aligns with Indian Standards 📘🇮🇳\n🎯 **TO THE POINT** – No bakwaas, only relevant explanations! 💥\n🪄 **FUN + FACTS** – Little jokes + real site examples = better memory!\n🧩 **MULTIPLE METHODS** – Shortcuts, concepts, and quick exam hacks 🎯\n🧰 **PRACTICAL VISION** – From drawing board to actual site ka gyaan 👷‍♂️\n\n**💬 EXAMPLES YOU CAN ASK:**\n- “Solve a simply supported beam using IS 456:2000.”\n- “Design a footing for column per IS 2911.”\n- “Find safe bearing capacity using Terzaghi’s method.”\n- “Calculate super elevation for highway curve (IRC:38).”\n- “Explain CPWD procedure for concrete curing.”\n\n**💡 MY PROMISE TO YOU:**\n✅ IS & CPWD code-based accurate answers 🧾\n✅ Simple, site-style explanations (like a senior teaching a junior!) 👷‍♀️👷‍♂️\n✅ Fun + Focused – with emojis, examples & real-life logic! 😄📏\n✅ Step-by-step clarity – No confusion, only confidence! 💪\n\n**💬 MOTIVATION BOOSTER:**\n_Build concepts strong like RCC, solve doubts fast like ready-mix concrete!_ 🧱💥\n\nReady to rock your GATE Civil prep – Indian style? 🇮🇳✨\nLet's crack it together! 🔥🎯"
    }


}

    # Add more modes as needed


@app.post("/chat")
async def chat(query: str = Form(...), mode: str = Form(None), history: str = Form(None)):
    if not query.strip() or len(query) > 10000:
        raise HTTPException(status_code=400, detail="Invalid query length")
    
    # Parse and limit history
    limited_history = []
    if history:
        try:
            history_data = json.loads(history)
            limited_history = history_data[-6:]
        except Exception:
            limited_history = []
    
    start_time = time.time()
    timings = {}

    try:
        loop = asyncio.get_event_loop()

        if mode and mode in CHAT_MODES:
            # Mode-specific fast path (already fast)
            system_prompt = CHAT_MODES[mode]["prompt"]
            messages = [("system", system_prompt)]
            
            if limited_history:
                for msg in limited_history:
                    if msg["role"] == "user":
                        messages.append(("human", msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(("ai", msg["content"]))
            
            messages.append(("human", query))
            
            generation_start = time.time()
            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        thread_pool,
                        lambda: llm.invoke(messages)
                    ),
                    timeout=25.0  # Reduced from 25s
                )
                answer = response.content if hasattr(response, 'content') else str(response)
            except asyncio.TimeoutError:
                answer = "I'm thinking... please try again in a moment!"
            
            generation_end = time.time()
            timings["generation_time"] = generation_end - generation_start
            timings["retrieval_time"] = 0.0
            timings["processing_time"] = 0.0
            raw_docs = []
            processed_docs = []
      
        else:
            # Build conversation history for document mode (last 2 messages)
            conversation_history = []
            if limited_history:
                # Take last 2 exchanges (4 messages: user+assistant pairs)
                recent_history = limited_history[-4:]
                for msg in recent_history:
                    if msg["role"] == "user":
                        conversation_history.append(("human", msg["content"]))
                    elif msg["role"] == "assistant":
                        conversation_history.append(("ai", msg["content"]))
            
            # Create prompt with history
            messages = [
                ("system",
                "You are Vishnu AI Assintant — a friendly but bit funny "
                "Provide accurate, clear, human-like answers in a warm and professional tone. "
                "Add light Indian humor naturally when it fits (for example, 'as easy as making Maggi'). "
                "Keep humor after the main answer, on a new line, ending with a small emoji"
                "If the user asks a general question, gently suggest they can change the tone using the 'tone selector'.")
            ]

            messages.extend(conversation_history)
            
            # Add current query with context
            messages.append(("human", "Context: {context}\n\nQuestion: {input}\n\nAnswer:"))
            
            prompt = ChatPromptTemplate.from_messages(messages)
            

            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            

            # 🚀 STEP 2: NOW start retrieval in parallel
            retrieval_start = time.time()
            try:
                retrieval_start = time.time()
                # raw_docs = retriever.invoke(query) if retriever else []
                raw_docs = await async_retrieve_documents(query, retriever, max_timeout=6.0)
                retrieval_end = time.time()
                timings["retrieval_time"] = retrieval_end - retrieval_start
            except asyncio.TimeoutError:
                raw_docs = []
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                raw_docs = []

            retrieval_end = time.time()
            timings["retrieval_time"] = retrieval_end - retrieval_start

            logger.info(f"⏱️ Retrieval took {timings['retrieval_time']:.2f}s")

            # 🚀 FAST Document Processing
            processing_start = time.time()
            final_docs = ensure_tabular_inclusion(raw_docs, query, min_tabular=2)
            processed_docs = post_process_retrieved_docs(final_docs, query)
            # logger.info(f"\n🎯 PROCESSED DOCUMENTS AFTER FILTERING:")
            # logger.info(f"📊 Final documents sent to LLM: {len(processed_docs)}")
            # for i, doc in enumerate(processed_docs, 1):
            #     logger.info(f"\n📄 FINAL DOC #{i}:")
            #     logger.info(f"📁 Source: {doc.metadata.get('source', 'Unknown')}")
            #     logger.info(f"📄 Page: {doc.metadata.get('page', 'N/A')}")
            #     logger.info(f"🔢 Chunk: {doc.metadata.get('chunk_num', 'N/A')}/{doc.metadata.get('total_chunks_page', 'N/A')}")
            #     logger.info(f"📊 Score: {doc.metadata.get('score', 'N/A')}")
            #     logger.info(f"📄 Content Type: {doc.metadata.get('content_type', 'unknown')}")
            #     logger.info(f"📏 Doc Size: {len(doc.page_content)} characters")
            #     logger.info(f"📝 Content Preview: {doc.page_content}...")
            
            processing_end = time.time()
            timings["processing_time"] = processing_end - processing_start
            logger.info(f"⏱️ Processing took {timings['processing_time']:.2f}s")

            # 🚀 GENERATION
            # In your chat endpoint, replace the generation section with:

            generation_start = time.time()
            if not processed_docs:
                answer = "I couldn't find specific information about that in my knowledge base. Is there anything else I can help you with?"
            else:
                try:
                    response = await asyncio.wait_for(
                        loop.run_in_executor(
                            thread_pool,
                            lambda: question_answer_chain.invoke({
                                "input": query,
                                "context": processed_docs
                            })
                        ),
                        timeout=15.0
                    )
                    answer = response.strip()
                except asyncio.TimeoutError:
                    answer = "I'm taking too long to generate a perfect response. Here's what I found quickly!"

            generation_end = time.time()
            timings["generation_time"] = generation_end - generation_start
            logger.info(f"⏱️ Generation took {timings['generation_time']:.2f}s")
            
            
        # Response formatting
        chat_history = []
        chat_entry = f"You: {query}\nAI: {answer}"
        chat_history.insert(0, chat_entry)
        if len(chat_history) > 3:
            chat_history.pop()

        total_end = time.time()
        timings["total_time"] = total_end - start_time

        logger.info("#" * 100)
        logger.info(f"⏱️ Retrieval took {timings['retrieval_time']:.2f}s")
        logger.info(f"⏱️ Processing took {timings['processing_time']:.2f}s")
        logger.info(f"⏱️ Generation took {timings['generation_time']:.2f}s")
        logger.info(f"🧮 TOTAL chat request took {timings['total_time']:.2f}s")
        logger.info("#" * 100)


        return {
            "answer": answer,
            "history": "\n\n".join(chat_history),
            "timings": {k: f"{v:.2f}s" for k, v in timings.items()},
            "retrieved_docs_count": len(raw_docs),
            "processed_docs_count": len(processed_docs) if 'processed_docs' in locals() else 0
        }

    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        return {
            "answer": "I'm experiencing technical issues. Please try again in a moment.",
            "history": "",
            "error": True
        }


##################################################################################################

# Redis for progress tracking (fallback to in-memory if Redis not available)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    USE_REDIS = True
    logger.info("Redis connected for progress tracking")
except:
    USE_REDIS = False
    progress_store = {}
    logger.info("Using in-memory progress tracking")
# Platform-specific Ghostscript binary
gs_binary = "gswin64c" if platform.system() == "Windows" else "gs"

# Compression presets matching UI
compression_presets = {
    "screen": {"dpi": 72, "quality": "screen", "desc": "Low quality, smallest size"},
    "ebook": {"dpi": 150, "quality": "ebook", "desc": "Medium quality, good compression"},
    "printer": {"dpi": 300, "quality": "printer", "desc": "High quality for printing"},
    "prepress": {"dpi": 300, "quality": "prepress", "desc": "Highest quality, minimal compression"}
}

# ========== CENTRALIZED FILE SIZE CONFIGURATION ==========
COMPRESS_MAX_FILE_SIZE_MB = 100  # Maximum allowed file size
# =========================================================



def validate_file_size(file_size_bytes: int):
    """Validate file size against maximum limit"""
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > COMPRESS_MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400, 
            detail=f"File exceeds {COMPRESS_MAX_FILE_SIZE_MB}MB limit"
        )

class ProgressTracker:
    def __init__(self):
        self.tasks: Dict[str, dict] = {}
    
    async def update_progress(self, task_id: str, progress: int, message: str = "", stage: str = ""):
        """Update progress with smooth transitions"""
        progress_data = {
            "progress": max(0, min(100, progress)),  # Ensure within bounds
            "message": message,
            "stage": stage,
            "timestamp": time.time()
        }
        
        # Store previous progress for smooth transitions
        if task_id in self.tasks:
            previous_progress = self.tasks[task_id].get("progress", 0)
            # Ensure progress doesn't go backwards
            if progress < previous_progress:
                progress = previous_progress
        
        self.tasks[task_id] = progress_data
        
        if USE_REDIS:
            try:
                redis_client.setex(f"progress:{task_id}", 300, json.dumps(progress_data))
            except Exception as e:
                logger.error(f"Redis update error: {e}")
        else:
            progress_store[task_id] = progress_data
        
        logger.info(f"Progress: {task_id} - {progress}% - {message}")
        
        # Small delay to allow UI to process
        await asyncio.sleep(0.1)

    def get_progress(self, task_id: str):
        """Get current progress"""
        try:
            if USE_REDIS:
                data = redis_client.get(f"progress:{task_id}")
                if data:
                    return json.loads(data)
            else:
                return progress_store.get(task_id)
        except:
            return None

    def update_progress_sync(self, task_id: str, progress: int, message: str = "", stage: str = ""):
        """Synchronous version that properly handles async progress updates"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # ✅ This is SAFE - creates async task that runs concurrently
                asyncio.create_task(self.update_progress(task_id, progress, message, stage))
            else:
                # ✅ This runs the async function to completion
                loop.run_until_complete(self.update_progress(task_id, progress, message, stage))
        except RuntimeError:
            # ✅ Handles case where no event loop exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.update_progress(task_id, progress, message, stage))


# Initialize progress tracker
progress_tracker = ProgressTracker()

async def update_progress(task_id: str, progress: int, message: str = "", stage: str = ""):
    """Wrapper function for S3 callback to use"""
    await progress_tracker.update_progress(task_id, progress, message, stage)

def cleanup_local_files():
    """Clean up files in input_pdfs and output_pdfs directories."""
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

import os, hashlib, logging

def upload_to_s3(file_content: bytes, filename: str) -> str:
    """Upload file content to S3 and return the S3 key."""
    if not isinstance(file_content, (bytes, bytearray)):
        raise TypeError("file_content must be bytes")

    safe_filename = os.path.basename(filename)
    s3_key = f"temp_uploads/{hashlib.md5(file_content).hexdigest()}_{safe_filename}"

    try:
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=file_content
        )
        logger.info(f"✅ Uploaded to S3: {s3_key}")
    except Exception as e:
        logger.error(f"❌ Failed to upload to S3: {e}")
        raise

    return s3_key


def cleanup_s3_file(s3_key: str):
    """Delete file from S3."""
    try:
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        logger.info(f"Deleted S3 file: {s3_key}")
    except Exception as e:
        logger.warning(f"Failed to delete S3 file {s3_key}: {e}")




def safe_delete_temp_file(file_path: str):
    """Safely delete a single temporary file with better error handling"""
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
            logger.info(f"✅ Deleted temp file: {file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to delete temp file {file_path}: {str(e)}")



#########################  estimate size $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
### helper function for size estimation

def get_fallback_estimate(original_size_mb: float, preset_name: str) -> float:
    """Get fallback compression ratios when actual compression fails."""
    fallback_ratios = {
        "screen": 0.2, 
        "ebook": 0.4, 
        "printer": 0.7, 
        "prepress": 0.9
    }
    return round(original_size_mb * fallback_ratios.get(preset_name, 0.5), 2)

def get_compression_recommendation(estimates: dict, original_size_mb: float) -> str:
    """Determine the best compression recommendation."""
    if estimates.get("ebook", original_size_mb) < original_size_mb * 0.6:
        return "ebook"
    elif estimates.get("printer", original_size_mb) < original_size_mb * 0.8:
        return "printer"
    else:
        return "screen"



############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#######################################################


def compress_pdf_ghostscript_file(input_path: str, output_path: str, compression_level: str = "ebook"):
    """Compress PDF using Ghostscript with AWS-free-tier optimized settings"""
    # global current_ghostscript_process, current_task_id
    
    compression_settings = {
        "screen": "/screen",
        "ebook": "/ebook", 
        "printer": "/printer",
        "prepress": "/prepress"
    }
    
    if compression_level not in compression_settings:
        compression_level = "ebook"

    # Platform-specific CPU throttling (FIXED FOR WINDOWS)
    if platform.system() == "Windows":
        # Use subprocess.CREATE_NO_WINDOW flag instead of cmd start
        base_cmd = [gs_binary]
        creationflags = subprocess.CREATE_NO_WINDOW  # This hides the CMD window
    else:
        base_cmd = ["nice", "-n", "10", gs_binary]  # Low priority on Linux
        creationflags = 0  # No special flags on Linux

    gs_command = base_cmd + [
        "-dNOPAUSE",
        "-dBATCH", 
        "-dQUIET",
        "-sDEVICE=pdfwrite",
        f"-dPDFSETTINGS={compression_settings[compression_level]}",
        "-dCompatibilityLevel=1.4",
        
        # ✅ OPTIMIZED IMAGE COMPRESSION SETTINGS (CPU-friendly)
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dEmbedAllFonts=true", 
        "-dSubsetFonts=true",
        
        # ✅ REDUCED IMAGE RESOLUTION (Major CPU savings)
        "-dColorImageDownsampleType=/Average",    # Faster than Bicubic (60% less CPU)
        "-dGrayImageDownsampleType=/Average",     # Faster than Bicubic
        "-dMonoImageDownsampleType=/Subsample",   # Fastest for B&W
        
        "-dColorImageResolution=120",    # Reduced from 150 (good balance)
        "-dGrayImageResolution=120",     # Reduced from 150
        "-dMonoImageResolution=200",     # Reduced from 300 (text stays sharp)
        
        # ✅ SIMPLIFIED IMAGE FILTERS (Less CPU intensive)
        "-dAutoFilterColorImages=true",  # Let GS choose automatically (faster)
        "-dAutoFilterGrayImages=true",
        "-dColorImageFilter=/DCTEncode",
        "-dGrayImageFilter=/DCTEncode",
        
        # ✅ COMPRESSION SETTINGS
        "-dCompressPages=true",
        
        # ✅ REDUCED MEMORY LIMITS (Better for free tier)
        "-dMaxPatternBitmap=5000000",    # Reduced from 10MB to 5MB
        "-dBufferSpace=80000000",        # Reduced from 150MB to 80MB
        "-dMaxBitmap=50000000",          # Reduced from 100MB to 50MB
        "-dNumRenderingThreads=1",       # Single thread (prevents CPU spikes)
        "-dMaxScreenBitmap=524288",      # Reduced from 1MB to 512KB
        
        # ✅ PERFORMANCE SETTINGS (CPU optimized)
        "-dUseFastColor=true",
        "-dNOGC",                        # Disable garbage collection (faster)
        "-dUseCropBox",                  # Use crop box instead of media box
        
        f"-sOutputFile=" + output_path,
        input_path
    ]
    
    try:
        logger.info(f"Running AWS-optimized Ghostscript compression: {compression_level}")
        
        # Set environment to prevent multi-threading
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", 
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "GS_THREADS": "1"
        })
        
        result = subprocess.run(
            gs_command,
            capture_output=True,
            timeout=400,  # Slightly increased for slower processing
            env=env,      # Pass thread-limited environment
            creationflags=creationflags  # ✅ This hides the window on Windows
        )
        
        if result.returncode != 0:
            logger.error(f"Ghostscript failed with return code {result.returncode}")
            if result.stderr:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                logger.error(f"Ghostscript stderr: {error_msg}")
                
                # Fallback to simpler compression if complex one fails
                if "memory" in error_msg.lower() or "timeout" in error_msg.lower():
                    logger.info("Attempting fallback compression with screen preset")
                    return compress_with_fallback(input_path, output_path)
                    
            return False
            
        # Verify output file was created
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            savings = ((original_size - compressed_size) / original_size) * 100
            logger.info(f"Compression completed: {original_size/1024/1024:.1f}MB → {compressed_size/1024/1024:.1f}MB ({savings:.1f}% savings)")
            return True
        else:
            logger.error("Ghostscript output file is missing or too small")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Ghostscript timeout with optimized settings")
        # Try fallback
        return compress_with_fallback(input_path, output_path)
    except Exception as e:
        logger.error(f"Optimized compression error: {str(e)}")
        return compress_with_fallback(input_path, output_path)

def compress_with_fallback(input_path: str, output_path: str) -> bool:
    """Ultra-light fallback compression for problematic files"""
    try:
        if platform.system() == "Windows":
            base_cmd = [gs_binary]
            creationflags = subprocess.CREATE_NO_WINDOW
        else:
            base_cmd = ["nice", "-n", "15", gs_binary]  # Even lower priority
            creationflags = 0
            
        fallback_command = base_cmd + [
            "-dNOPAUSE",
            "-dBATCH", 
            "-dQUIET",
            "-sDEVICE=pdfwrite",
            "-dPDFSETTINGS=/screen",  # Fastest preset
            "-dCompatibilityLevel=1.4",
            "-dColorImageResolution=100",
            "-dGrayImageResolution=100", 
            "-dMonoImageResolution=150",
            "-dColorImageDownsampleType=/Average",
            "-dGrayImageDownsampleType=/Average",
            "-dMonoImageDownsampleType=/Subsample",
            "-dNumRenderingThreads=1",
            f"-sOutputFile=" + output_path,
            input_path
        ]
        
        result = subprocess.run(
            fallback_command,
            capture_output=True,
            timeout=300,
            env={**os.environ, "OMP_NUM_THREADS": "1", "GS_THREADS": "1"},
            creationflags=creationflags  # ✅ Hide window in fallback too
        )
        
        success = result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000
        logger.info(f"Fallback compression {'succeeded' if success else 'failed'}")
        return success
        
    except Exception as e:
        logger.error(f"Fallback compression also failed: {str(e)}")
        return False




def cleanup_compression_estimation_files(task_id: str):
    """Clean up compression and estimation files for specific task"""
    try:
        patterns = [
            f"{task_id}_*",              # Uploaded files
            f"compressed_{task_id}_*",    # Compression outputs
            f"*{task_id}_*estimation.pdf" # Estimation files  
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
    """Stream file upload to disk to avoid RAM usage for ALL files with progress tracking"""
    # Use your UPLOAD_DIR instead of tempdir
    filename = f"{task_id}_{file.filename}"
    file_path = UPLOAD_DIR / filename
    
    logger.info(f"Streaming upload to disk: {file_path}")
    
    # Get total file size for progress calculation
    file_size = 0
    if hasattr(file, 'size') and file.size:
        file_size = file.size
    else:
        # If size is not available, we'll estimate based on chunks
        logger.warning("File size not available, using chunk-based progress")
    
    uploaded_size = 0
    chunk_number = 0
    
    with open(file_path, "wb") as buffer:
        while True:
            chunk = await file.read(128 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
            
            # Update progress based on uploaded chunks
            uploaded_size += len(chunk)
            chunk_number += 1
            
            # Calculate progress percentage
            if file_size > 0:
                progress = 10 + (uploaded_size / file_size) * 20  # 10-30% range for upload
                progress = min(30, progress)  # Cap at 30%
                
                # Update progress every 10 chunks or 1MB to avoid too many updates
                if chunk_number % 10 == 0 or uploaded_size % (1024 * 1024) == 0:
                    await progress_tracker.update_progress(
                        task_id, 
                        int(progress), 
                        f"Uploading to disk... ({uploaded_size / (1024 * 1024):.1f} MB)", 
                        "uploading"
                    )
    
    logger.info(f"File upload completed: {uploaded_size} bytes written to {file_path}")
    
    # Final upload completion update
    await progress_tracker.update_progress(task_id, 30, "File streamed to disk!", "processing")
    
    return str(file_path)
####################################  ESTIMATION $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

estimation_results = {}
async def process_estimation_sequential_disk(task_id: str, file_path: str, filename: str):
    """Estimation with sequential processing to save SYSTEM RESOURCES"""
    try:
        await progress_tracker.update_progress(task_id, 25, "Analyzing file sequentially...", "processing")
        
        original_size = os.path.getsize(file_path)
        original_size_mb = original_size / (1024 * 1024)
        
        estimates = {}
        key_presets = ["ebook", "printer", "screen", "prepress"]
        
        for i, preset_name in enumerate(key_presets):
            progress = 30 + i * 15
            await progress_tracker.update_progress(task_id, progress, f"Testing {preset_name} compression...", "compressing")
            
            try:
                # Create output file in ESTIMATION_DIR
                output_filename = f"{task_id}_{preset_name}_estimation.pdf"
                output_path = ESTIMATION_DIR / output_filename
                
                # ✅ Run ONE compression at a time
                success = compress_pdf_ghostscript_file(file_path, str(output_path), preset_name)
                
                if success and os.path.exists(output_path):
                    compressed_size = os.path.getsize(output_path)
                    compressed_size_mb = compressed_size / (1024 * 1024)
                    estimates[preset_name] = round(compressed_size_mb, 2)
                    
                    # Calculate actual savings percentage
                    savings_pct = ((original_size - compressed_size) / original_size) * 100
                    logger.info(f"✅ {preset_name}: {original_size_mb:.1f}MB → {compressed_size_mb:.1f}MB ({savings_pct:.1f}% savings)")
                    
                    # Clean up temp file immediately
                    safe_delete_temp_file(str(output_path))
                    
                    # ✅ Small delay to reduce system load
                    await asyncio.sleep(0.5)
                    
                else:
                    estimates[preset_name] = get_fallback_estimate(original_size_mb, preset_name)
                    logger.warning(f"⚠️ {preset_name} compression failed, using fallback")
                    
            except Exception as e:
                logger.error(f"Preset {preset_name} estimation failed: {str(e)}")
                estimates[preset_name] = get_fallback_estimate(original_size_mb, preset_name)

        estimates["original"] = round(original_size_mb, 2)

        estimation_results[task_id] = {
            "estimates": estimates,
            "original_size_mb": round(original_size_mb, 2),
            "used_s3": False,
            "sequential_processing": True  # Flag to indicate sequential was used
        }
        
        # Clean up input file
        safe_delete_temp_file(file_path)
        
        await progress_tracker.update_progress(task_id, 100, "Sequential estimation completed!", "completed")
        
    except Exception as e:
        logger.error(f'Sequential estimation error: {str(e)}')
        safe_delete_temp_file(file_path)
        await progress_tracker.update_progress(task_id, 100, f"Error: {str(e)}", "error")
        cleanup_compression_estimation_files(task_id)
    finally:
        cleanup_compression_estimation_files(task_id)   #### this may craette problemmmmm############################################



@app.post("/start_estimation")
async def start_estimation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Start estimation with SEQUENTIAL disk processing for better resource usage"""
    # global current_task_id
    logger.info(f"Starting SEQUENTIAL estimation: file={file.filename}")
    
    validate_file_size(file.size)
    task_id = str(uuid.uuid4())
    
    await progress_tracker.update_progress(task_id, 0, "Starting sequential estimation...", "initializing")
    
    try:
        await progress_tracker.update_progress(task_id, 10, "Streaming file to disk...", "uploading")
        temp_path = await stream_upload_to_disk(file, task_id)
        await progress_tracker.update_progress(task_id, 20, "File streamed to disk!", "processing")
        
        # ✅ USE SEQUENTIAL PROCESSING INSTEAD
        background_tasks.add_task(process_estimation_sequential_disk, task_id, temp_path, file.filename)
        
        return JSONResponse(content={
            "task_id": task_id,
            "status": "started", 
            "message": "Sequential estimation started in background",
            "using_s3": False,
            "processing_mode": "sequential_disk"  # Indicate sequential processing
        })
        
    except Exception as e:
        logger.error(f"❌ Failed to start sequential estimation: {str(e)}")
        await progress_tracker.update_progress(task_id, 100, f"Error: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.get("/estimation_result/{task_id}")
async def get_estimation_result(task_id: str):
    """Get the final ACTUAL estimation results."""
    if task_id not in estimation_results:
        raise HTTPException(status_code=404, detail="Estimation result not found or expired")
    
    result = estimation_results[task_id]
    
    # ✅ Add savings percentages to the result
    estimates = result["estimates"]
    original_size = result["original_size_mb"]
    
    savings_data = {}
    for preset, compressed_size in estimates.items():
        if preset != "original":
            savings_pct = ((original_size - compressed_size) / original_size) * 100
            savings_data[preset] = f"{savings_pct:.1f}%"
    
    result["savings_percentages"] = savings_data
    result["recommendation"] = get_compression_recommendation(estimates, original_size)
    
    del estimation_results[task_id]  # Clean up
    
    logger.info(f"✅ Returning ACTUAL estimation results for {task_id}: {savings_data}")
    
    return JSONResponse(content=result)


########################################################################################
############################# compression ###########################################################

compression_results = {}


async def process_compression_disk_only(task_id: str, input_path: str, filename: str, preset: str):
    """Process compression using disk I/O for ALL files"""
    try:
        await progress_tracker.update_progress(task_id, 40, "Starting disk compression...", "compressing")
        await asyncio.sleep(0.3)
        
        # Use your OUTPUT_DIR instead of tempdir
        output_filename = f"compressed_{task_id}_{Path(filename).stem}_{preset}.pdf"
        output_path = OUTPUT_DIR / output_filename
        
        # Use file-based compression
        success = compress_pdf_ghostscript_file(input_path, str(output_path), preset)
        
        if not success:
            error_msg = "Disk compression failed"
            logger.error(f"❌ {error_msg}")
            safe_delete_temp_file(input_path)
            safe_delete_temp_file(str(output_path))
            await progress_tracker.update_progress(task_id, 100, error_msg, "error")
            raise Exception(error_msg)
        
        await progress_tracker.update_progress(task_id, 90, "Finalizing compressed PDF...", "finalizing")
        
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        savings = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0
        
        # Store result with disk path
        result_data = {
            "file_path": str(output_path),
            "filename": f"compressed_{Path(filename).stem}_{preset}.pdf",
            "original_size": original_size,
            "compressed_size": compressed_size,
            "savings": savings,
            "preset": preset,
            "on_disk": True  # Always true now
        }
        
        if USE_REDIS:
            redis_client.setex(f"compressed_meta:{task_id}", 600, json.dumps(result_data))
        else:
            compression_results[task_id] = result_data
        
        # Clean up input file
        safe_delete_temp_file(input_path)
        
        logger.info(f"✅ Disk compression completed! Savings: {savings:.1f}%")
        await progress_tracker.update_progress(task_id, 100, f"Compression completed! Size reduced by {savings:.1f}%", "completed")
        
    except Exception as e:
        logger.error(f'❌ Disk compression error: {str(e)}')
        # Clean up on error
        safe_delete_temp_file(input_path)
        safe_delete_temp_file(str(output_path))
        await progress_tracker.update_progress(task_id, 100, f"Error: {str(e)}", "error")
        cleanup_compression_estimation_files(task_id)
        raise

    # finally:
    #     # GUARANTEED cleanup for THIS operation
    #     cleanup_compression_estimation_files(task_id)




@app.post("/start_compression")
async def start_compression(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    preset: str = Form("ebook")
):
    """Start compression with disk processing for ALL files"""
    # global current_task_id
    logger.info(f"Starting compression: file={file.filename}, preset={preset}")
    
    validate_file_size(file.size)
    task_id = str(uuid.uuid4())
    
    await progress_tracker.update_progress(task_id, 0, "Initializing compression...", "initializing")
    
    try:
        # ✅ ALWAYS USE DISK PROCESSING - REMOVE MEMORY CHECK
        setup_directories()
        await progress_tracker.update_progress(task_id, 10, "Streaming file to disk...", "uploading")
        await asyncio.sleep(0.3)
        temp_path = await stream_upload_to_disk(file, task_id)
        await progress_tracker.update_progress(task_id, 30, "File streamed to disk!", "processing")
        await asyncio.sleep(0.3)
        
        background_tasks.add_task(process_compression_disk_only, task_id, temp_path, file.filename, preset)
        
        return JSONResponse(content={
            "task_id": task_id,
            "status": "started", 
            "message": "Compression started in background",
            "using_s3": False,
            "processing_mode": "disk"  # Always disk now
        })
        
    except Exception as e:
        logger.error(f"❌ Failed to start compression: {str(e)}")
        await progress_tracker.update_progress(task_id, 100, f"Error: {str(e)}", "error")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    

@app.get("/download_compressed/{task_id}")
async def download_compressed(task_id: str):
    """Stream compressed file from disk ONLY"""
    try:
        
        result_data = None
        
        # Get metadata
        if USE_REDIS:
            metadata = redis_client.get(f"compressed_meta:{task_id}")
            if metadata:
                result_data = json.loads(metadata)
        else:
            result_data = compression_results.get(task_id)
        
        if not result_data:
            raise HTTPException(status_code=404, detail="Compressed result not found or expired")
        
        filename = result_data["filename"]
        
        # ✅ Handle disk-based files ONLY (memory path removed)
        file_path = result_data["file_path"]
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Compressed file not found")
        
        def file_generator():
            with open(file_path, "rb") as f:
                while chunk := f.read(128 * 1024):
                    yield chunk
            # Cleanup after streaming
            safe_delete_temp_file(file_path)
            if USE_REDIS:
                redis_client.delete(f"compressed_meta:{task_id}")
            elif task_id in compression_results:
                del compression_results[task_id]
            logger.info(f"Cleaned up disk file: {task_id}")
        
        return StreamingResponse(
            file_generator(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Original-Size": str(result_data["original_size"]),
                "X-Compressed-Size": str(result_data["compressed_size"]),
                "X-Savings-Percent": f"{result_data['savings']:.1f}",
                "X-Compression-Level": result_data["preset"]
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download compressed file")
    # finally:
    #     # GUARANTEED cleanup for THIS operation
    #     cleanup_compression_estimation_files(task_id)


@app.get("/progress/{task_id}")
async def get_progress_status(task_id: str):
    """Get current progress for a task"""
    progress_data = progress_tracker.get_progress(task_id)
    if not progress_data:
        raise HTTPException(status_code=404, detail="Task not found or expired")
    
    return JSONResponse(content=progress_data)


import subprocess
import platform

@app.post("/stop_operations")
async def stop_operations():
    """Kill ALL Ghostscript processes - Cross-platform"""
    try:
        system = platform.system()
        
        if system == "Windows":
            # Windows - kill gswin64c and gswin32c
            subprocess.run(["taskkill", "/f", "/im", "gswin64c.exe"], capture_output=True)
            subprocess.run(["taskkill", "/f", "/im", "gswin32c.exe"], capture_output=True)
        else:
            # Linux (AWS EC2) - kill ghostscript processes
            subprocess.run(["pkill", "-f", "ghostscript"], capture_output=True)
            subprocess.run(["pkill", "-f", "gs"], capture_output=True)
        
        logger.info("✅ Killed all Ghostscript processes")
        return {"status": "stopped", "message": "All operations terminated"}
        
    except Exception as e:
        logger.error(f"Error stopping operations: {str(e)}")
        return {"status": "error", "message": str(e)}





# Lazy import function for Adobe PDF Services
def get_adobe_services():
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.pdf_services import PDFServices
    
    credentials = ServicePrincipalCredentials(
        client_id=os.getenv('PDF_SERVICES_CLIENT_ID'),
        client_secret=os.getenv('PDF_SERVICES_CLIENT_SECRET')
    )
    return PDFServices(credentials=credentials)

async def save_uploaded_file_disk(file: UploadFile, task_id: str) -> Path:
    """Save uploaded file directly to disk without loading into RAM"""
    # Validate file size
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE/1024/1024}MB limit")
    
    # Create task-specific directory
    task_upload_dir = UPLOAD_DIR / task_id
    task_upload_dir.mkdir(exist_ok=True)
    
    # Save file directly to disk in chunks
    file_path = task_upload_dir / file.filename
    
    with open(file_path, "wb") as buffer:
        # Read and write in chunks to avoid RAM usage
        while True:
            chunk = await file.read(65536)
 # 8KB chunks
            if not chunk:
                break
            buffer.write(chunk)
    
    logger.info(f"✅ File saved to disk: {file_path}")
    return file_path

def validate_pdf_pages(file_path: Path) -> int:
    """Validate PDF page count without loading into RAM"""
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PdfReader(f)
            page_count = len(pdf_reader.pages)
            
            if page_count > MAX_PAGES:
                raise HTTPException(
                    status_code=400, 
                    detail=f"PDF exceeds {MAX_PAGES} pages limit. Found {page_count} pages."
                )
            return page_count
    except Exception as e:
        logger.error(f"PDF validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid PDF file")

def cleanup_task_files(task_id: str):
    """Clean up ALL files associated with a specific task across ALL directories"""
    try:
        cleaned_count = 0
        
        # Define all directories to check
        directories_to_clean = [UPLOAD_DIR, OUTPUT_DIR, ESTIMATION_DIR, PDFTOWORD, TEMP_DIR]
        
        for base_dir in directories_to_clean:
            if not base_dir.exists():
                continue
                
            # Clean task-specific directories
            task_dir = base_dir / task_id
            if task_dir.exists() and task_dir.is_dir():
                try:
                    shutil.rmtree(task_dir)
                    cleaned_count += 1
                    logger.info(f"✅ Cleaned task directory: {task_dir}")
                except Exception as e:
                    logger.error(f"❌ Failed to clean directory {task_dir}: {e}")
            
            # Clean individual files with task_id in filename
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
            logger.info(f"🧹 Cleanup completed for task {task_id}: {cleaned_count} items removed")
        else:
            logger.info(f"ℹ️ No files found to clean for task {task_id}")
            
    except Exception as e:
        logger.error(f"💥 Critical error in cleanup_task_files for {task_id}: {e}") 


def cleanup_orphaned_files():
    """Clean up FILES older than 15 minutes considering ALL file activities"""
    try:
        current_time = time.time()
        # orphan_age_seconds = 900  # 15 minutes
        
        cleaned_count = 0
        for root_dir in [UPLOAD_DIR, OUTPUT_DIR, ESTIMATION_DIR, PDFTOWORD, TEMP_DIR]:
            if not root_dir.exists():
                continue
                
            for item in root_dir.iterdir():
                try:
                    if item.is_dir():
                        continue
                    
                    stat = item.stat()
                    
                    # ✅ CONSIDER ALL TIMESTAMPS to protect against:
                    # - Copying files (st_ctime updates on copy on most systems)
                    # - Renaming files (st_ctime updates on metadata change)
                    # - Moving files (st_ctime updates)
                    # - Changing permissions (st_ctime updates) 
                    # - Reading files (st_atime updates)
                    # - Modifying content (st_mtime updates)
                    
                    most_recent = max(
                        stat.st_mtime,  # Content modification
                        stat.st_ctime,  # Metadata change (rename, move, permissions, copy)
                        stat.st_atime   # Last access (reading)
                    )
                    
                    # Skip if file had ANY activity in last 15 minutes
                    if most_recent > (current_time - orphan_age_seconds):
                        continue
                        
                    # Only delete truly orphaned files (no activity for 15+ minutes)
                    if item.is_file():
                        item.unlink()
                        cleaned_count += 1
                        logger.info(f"🧹 Cleaned up orphaned file: {item}")
                        
                except Exception as e:
                    logger.error(f"❌ Error cleaning {item}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"🎯 Orphaned files cleanup completed: {cleaned_count} files removed")
        else:
            logger.info("ℹ️ No orphaned files found for cleanup")
            
    except Exception as e:
        logger.error(f"💥 Critical error in orphaned file cleanup: {e}")

def convert_pdf_to_word_disk_based(pdf_file_path: Path, task_id: str, max_retries: int = 3) -> Optional[bytes]:
    """Convert PDF to Word with retry logic - Disk-based Adobe approach"""
    
    output_docx_path = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 PDF to Word conversion attempt {attempt + 1}/{max_retries} for task {task_id}")
            
            # Lazy import Adobe modules only when needed
            from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
            from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
            from adobe.pdfservices.operation.io.stream_asset import StreamAsset
            from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
            from adobe.pdfservices.operation.pdfjobs.jobs.export_pdf_job import ExportPDFJob
            from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_params import ExportPDFParams
            from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_target_format import ExportPDFTargetFormat
            from adobe.pdfservices.operation.pdfjobs.result.export_pdf_result import ExportPDFResult

            # Verify input file still exists (might be deleted in previous failed cleanup)
            if not pdf_file_path.exists():
                logger.error(f"❌ Input file missing for retry: {pdf_file_path}")
                return None

            # Read PDF directly from disk for Adobe upload
            with open(pdf_file_path, "rb") as file:
                input_stream = file.read()
            
            # Get Adobe services
            pdf_services = get_adobe_services()

            # Upload input PDF from disk
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

            # Configure export params
            export_pdf_params = ExportPDFParams(target_format=ExportPDFTargetFormat.DOCX)
            export_pdf_job = ExportPDFJob(input_asset=input_asset, export_pdf_params=export_pdf_params)

            # Submit job
            location = pdf_services.submit(export_pdf_job)
            result = pdf_services.get_job_result(location, ExportPDFResult)

            # Get converted DOCX
            result_asset: CloudAsset = result.get_result().get_asset()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            # Save to task-specific output directory
            output_task_dir = OUTPUT_DIR / task_id
            output_task_dir.mkdir(exist_ok=True)
            output_docx_path = output_task_dir / "converted.docx"
            
            with open(output_docx_path, "wb") as out_file:
                out_file.write(stream_asset.get_input_stream())

            # Read the result for response
            with open(output_docx_path, "rb") as f:
                docx_bytes = f.read()
            
            logger.info(f"✅ PDF to Word conversion successful on attempt {attempt + 1}")
            return docx_bytes

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logger.error(f"❌ Adobe PDF Services error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            
            # ⚠️ LIMITED CLEANUP - Only clean output files, NOT input file
            if output_docx_path and output_docx_path.exists():
                try:
                    output_docx_path.unlink()
                    logger.info(f"🧹 Cleaned failed output file: {output_docx_path}")
                except Exception as cleanup_error:
                    logger.error(f"❌ Failed to clean output file: {cleanup_error}")
            
            if attempt < max_retries - 1:  # Not the last attempt
                wait_time = (2 ** attempt) + 1
                logger.info(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error("💥 All conversion attempts failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Unexpected error in PDF to Word conversion (attempt {attempt + 1}): {str(e)}")
            
            # ⚠️ LIMITED CLEANUP - Only clean output files, NOT input file
            if output_docx_path and output_docx_path.exists():
                try:
                    output_docx_path.unlink()
                    logger.info(f"🧹 Cleaned failed output file: {output_docx_path}")
                except Exception as cleanup_error:
                    logger.error(f"❌ Failed to clean output file: {cleanup_error}")
            
            if attempt < max_retries - 1:  # Not the last attempt
                wait_time = (2 ** attempt) + 1
                logger.info(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error("💥 All conversion attempts failed")
                return None
    
    # ❌ REMOVE THIS - Cleanup should happen in the endpoint's finally block
    # cleanup_task_files(task_id)
    return None
# Update the endpoint call to pass task_id
@app.post("/convert_pdf_to_word")
async def convert_pdf_to_word_endpoint(file: UploadFile = File(...)):
    """PDF to Word conversion with guaranteed cleanup"""
    logger.info(f"📥 Received convert to Word request for {file.filename}")
    
    # Validate input
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE/1024/1024}MB limit")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    task_id = str(uuid.uuid4())
    temp_file_path = None
    
    try:
        # Step 1: Save file to task-specific directory
        temp_file_path = await save_uploaded_file_disk(file, task_id)
        logger.info(f"💾 File saved to: {temp_file_path}")
        
        # Step 2: Validate PDF
        page_count = validate_pdf_pages(temp_file_path)
        logger.info(f"📄 PDF validated: {page_count} pages")
        
        # Step 3: Perform conversion
        logger.info("🔄 Starting PDF to Word conversion...")
        docx_bytes = convert_pdf_to_word_disk_based(temp_file_path, task_id)
        
        if not docx_bytes:
            raise HTTPException(status_code=500, detail="Conversion failed after all retry attempts")
        
        # Step 4: Return successful result
        logger.info(f"✅ Conversion successful for task {task_id}")
        return StreamingResponse(
            io.BytesIO(docx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": 'attachment; filename="converted_output.docx"'}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Unexpected error in endpoint for task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")
    finally:
        # ✅ GUARANTEED CLEANUP - This runs in ALL cases (success or failure)
        logger.info(f"🧹 Final cleanup for task {task_id}")
        cleanup_task_files(task_id)
        cleanup_orphaned_files()

def convert_pdf_to_excel_disk_based(pdf_file_path: Path, task_id: str, max_retries: int = 3) -> Optional[bytes]:
    """Convert PDF to Excel with retry logic - Disk-based Adobe approach"""
    
    output_xlsx_path = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔄 PDF to Excel conversion attempt {attempt + 1}/{max_retries} for task {task_id}")
            
            # Lazy import Adobe modules only when needed
            from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
            from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
            from adobe.pdfservices.operation.io.stream_asset import StreamAsset
            from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
            from adobe.pdfservices.operation.pdfjobs.jobs.export_pdf_job import ExportPDFJob
            from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_params import ExportPDFParams
            from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_target_format import ExportPDFTargetFormat
            from adobe.pdfservices.operation.pdfjobs.result.export_pdf_result import ExportPDFResult

            # Verify input file still exists (might be deleted in previous failed cleanup)
            if not pdf_file_path.exists():
                logger.error(f"❌ Input file missing for Excel conversion retry: {pdf_file_path}")
                return None

            # Read PDF directly from disk for Adobe upload
            with open(pdf_file_path, "rb") as file:
                input_stream = file.read()
            
            # Get Adobe services
            pdf_services = get_adobe_services()

            # Upload input PDF from disk
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

            # Configure export params for Excel
            export_pdf_params = ExportPDFParams(target_format=ExportPDFTargetFormat.XLSX)
            export_pdf_job = ExportPDFJob(input_asset=input_asset, export_pdf_params=export_pdf_params)

            # Submit job
            location = pdf_services.submit(export_pdf_job)
            result = pdf_services.get_job_result(location, ExportPDFResult)

            # Get converted Excel
            result_asset: CloudAsset = result.get_result().get_asset()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            # Save to task-specific output directory
            output_task_dir = OUTPUT_DIR / task_id
            output_task_dir.mkdir(exist_ok=True)
            output_xlsx_path = output_task_dir / "converted.xlsx"
            
            with open(output_xlsx_path, "wb") as out_file:
                out_file.write(stream_asset.get_input_stream())

            # Read the result for response
            with open(output_xlsx_path, "rb") as f:
                xlsx_bytes = f.read()
            
            logger.info(f"✅ PDF to Excel conversion successful on attempt {attempt + 1}")
            return xlsx_bytes

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logger.error(f"❌ Adobe PDF Services Excel error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            
            # ⚠️ LIMITED CLEANUP - Only clean output files, NOT input file
            if output_xlsx_path and output_xlsx_path.exists():
                try:
                    output_xlsx_path.unlink()
                    logger.info(f"🧹 Cleaned failed Excel output file: {output_xlsx_path}")
                except Exception as cleanup_error:
                    logger.error(f"❌ Failed to clean Excel output file: {cleanup_error}")
            
            if attempt < max_retries - 1:  # Not the last attempt
                wait_time = (2 ** attempt) + 1
                logger.info(f"⏳ Retrying Excel conversion in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error("💥 All Excel conversion attempts failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Unexpected error in PDF to Excel conversion (attempt {attempt + 1}): {str(e)}")
            
            # ⚠️ LIMITED CLEANUP - Only clean output files, NOT input file
            if output_xlsx_path and output_xlsx_path.exists():
                try:
                    output_xlsx_path.unlink()
                    logger.info(f"🧹 Cleaned failed Excel output file: {output_xlsx_path}")
                except Exception as cleanup_error:
                    logger.error(f"❌ Failed to clean Excel output file: {cleanup_error}")
            
            if attempt < max_retries - 1:  # Not the last attempt
                wait_time = (2 ** attempt) + 1
                logger.info(f"⏳ Retrying Excel conversion in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error("💥 All Excel conversion attempts failed")
                return None
    
    return None
# Update the PDF to Excel endpoint
@app.post("/convert_pdf_to_excel")
async def convert_pdf_to_excel_endpoint(file: UploadFile = File(...)):
    """PDF to Excel conversion with guaranteed cleanup"""
    logger.info(f"📥 Received convert to Excel request for {file.filename}")
    
    # Validate input
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE/1024/1024}MB limit")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    task_id = str(uuid.uuid4())
    temp_file_path = None
    
    try:
        # Step 1: Save file to task-specific directory
        temp_file_path = await save_uploaded_file_disk(file, task_id)
        logger.info(f"💾 File saved to: {temp_file_path}")
        
        # Step 2: Validate PDF
        page_count = validate_pdf_pages(temp_file_path)
        logger.info(f"📄 PDF validated: {page_count} pages")
        
        # Step 3: Perform conversion
        logger.info("🔄 Starting PDF to Excel conversion...")
        xlsx_bytes = convert_pdf_to_excel_disk_based(temp_file_path, task_id)
        
        if not xlsx_bytes:
            raise HTTPException(status_code=500, detail="Excel conversion failed after all retry attempts")
        
        # Step 4: Return successful result
        logger.info(f"✅ Excel conversion successful for task {task_id}")
        return StreamingResponse(
            io.BytesIO(xlsx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="converted_output.xlsx"'}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Unexpected error in Excel endpoint for task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Excel conversion error: {str(e)}")
    finally:
        # ✅ GUARANTEED CLEANUP - This runs in ALL cases (success or failure)
        logger.info(f"🧹 Final cleanup for Excel task {task_id}")
        cleanup_task_files(task_id)
        logger.info(f"🗑️ Excel task {task_id} cleanup completed")

####################################################

@app.post("/encrypt_pdf")
async def encrypt_pdf_endpoint(file: UploadFile = File(...), password: str = Form(...)):
    logger.info(f"=== ENCRYPT ENDPOINT CALLED ===")
    logger.info(f"File: {file.filename}, Size: {file.size} bytes")
    logger.info(f"Password length: {len(password)}")
    
    if file.size and file.size / (1024 * 1024) > 50:
        logger.error("File size exceeds 50MB limit")
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")
    if not password:
        logger.error("No password provided")
        raise HTTPException(status_code=400, detail="Password required")

    s3_key = None
    try:
        logger.debug("Reading file content...")
        file_content = await file.read()
        logger.debug(f"File content read: {len(file_content)} bytes")
        
        if not file_content:
            logger.error("Empty file content")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Validate PDF
        logger.debug("Validating PDF...")
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            logger.debug(f"PDF validation passed: {doc.page_count} pages")
            doc.close()
        except Exception as e:
            logger.error(f"PDF validation failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid PDF file")
        
        # Store file
        if USE_S3:
            logger.debug("Uploading to S3...")
            s3_key = upload_to_s3(file_content, file.filename)
        else:
            logger.debug("Saving locally...")
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)

        logger.debug("Calling encrypt_pdf function...")
        encrypted_pdf = encrypt_pdf(file_content, password)
        
        if not encrypted_pdf:
            logger.error("encrypt_pdf returned None")
            raise HTTPException(status_code=500, detail="Encryption failed")

        logger.info("Encryption successful, returning response")
        return StreamingResponse(
            io.BytesIO(encrypted_pdf),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="encrypted_{file.filename}"'}
        )
        
    except HTTPException:
        logger.error("HTTPException raised")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"PDF encryption failed: {str(e)}")
    finally:
        logger.info("Cleaning up resources...")
        if s3_key:
            cleanup_s3_file(s3_key)
        if not USE_S3:
            cleanup_local_files()
        gc.collect()
        logger.info("=== ENCRYPT ENDPOINT COMPLETED ===")



@app.post("/remove_pdf_password")
async def remove_pdf_password_endpoint(file: UploadFile = File(...), password: str = Form(...)):
    logger.info(f"Received remove password request for {file.filename}")
    
    # Validate file size
    if file.size and file.size / (1024 * 1024) > 50:
        logger.error(f"File {file.filename} exceeds 50MB limit")
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")
    if not password:
        logger.error("Empty password provided")
        raise HTTPException(status_code=400, detail="Password cannot be empty")

    s3_key = None
    try:
        file_content = await file.read()
        
        # Validate file content
        if not file_content:
            logger.error(f"Empty file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Validate it's a PDF
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            doc.close()
        except Exception as e:
            logger.error(f"Invalid PDF file: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid PDF file")
        
        # Store file only after validation
        if USE_S3:
            s3_key = upload_to_s3(file_content, file.filename)
        else:
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)
        
        # Process the PDF
        decrypted_pdf = remove_pdf_password(file_content, password)
        if decrypted_pdf is None:
            logger.error("remove_pdf_password returned None")
            raise HTTPException(status_code=500, detail="Password removal failed")

        # Verify output is valid PDF
        try:
            test_doc = fitz.open(stream=decrypted_pdf, filetype="pdf")
            test_doc.close()
        except Exception as e:
            logger.error(f"Output PDF validation failed: {e}")
            raise HTTPException(status_code=500, detail="Output PDF is invalid")

        logger.info(f"Returning decrypted PDF for {file.filename}")
        return StreamingResponse(
            io.BytesIO(decrypted_pdf),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="decrypted_{file.filename}"'}
        )
    except ValueError as e:
        logger.error(f"Password removal error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Password removal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Password removal failed: {str(e)}")
    finally:
        logger.info(f"S3 cleanup: {'Running' if s3_key else 'Skipping (no S3 key)'}")
        if s3_key:
            cleanup_s3_file(s3_key)
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()

@app.post("/remove_background")
async def remove_background_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received remove background request for {file.filename}")
    
    MAX_FILE_SIZE_MB = 20
    # Validate file size
    if file.size and file.size / (1024 * 1024) > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    try:
        # Validate content type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
        file_content = await file.read()
        
        # Validate file content
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Validate image format by trying to open it
        try:
            import PIL.Image
            image = PIL.Image.open(io.BytesIO(file_content))
            # Check image dimensions to prevent memory issues
            if image.size[0] > 5000 or image.size[1] > 5000:
                raise HTTPException(status_code=400, detail="Image dimensions too large (max 5000x5000 pixels)")
            image.verify()  # Verify it's a valid image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Store file only after validation
        if not USE_S3:
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)

        logger.info("Processing image for background removal")
        processed_image = remove_background_rembg(file_content)
        
        # Validate output
        if not processed_image or not processed_image.getvalue():
            raise HTTPException(status_code=500, detail="Failed to process image")
        
        # Verify output is valid image
        try:
            output_image = PIL.Image.open(processed_image)
            output_image.verify()
            processed_image.seek(0)  # Reset stream position
        except Exception as e:
            raise HTTPException(status_code=500, detail="Output image is invalid")

        return StreamingResponse(
            content=processed_image,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=processed_{os.path.splitext(file.filename)[0]}.png"}
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
        
        logger.info("No S3 cleanup needed (in-memory processing)")
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()
        
@app.get("/videos")
async def list_videos():
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_PREFIX)
        videos = []
        
        if "Contents" in response:
            for obj in response["Contents"]:
                key = obj["Key"]
                if key.lower().endswith((".mp4", ".webm", ".ogg")):
                    try:
                        head = s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
                        metadata = head.get('Metadata', {})
                        content_type = head.get('ContentType', 'video/mp4')
                        
                        # Extract video_id (filename part after prefix)
                        video_id = key[len(S3_PREFIX):]
                        
                        # Direct S3 URL with #t=1 for preview frame
                        url = f"https://{BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{key}#t=1"
                        logger.info(f"Direct URL for {key}: {url}")
                        
                        videos.append({
                            "id": video_id,
                            "name": key.split('/')[-1],
                            "url": url,
                            "description": metadata.get('description', ''),
                            "type": content_type
                        })
                    except ClientError as e:
                        logger.error(f"Error processing {key}: {e}")
                        continue
        
        return JSONResponse(content=videos)

    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail="Failed to list videos")

@app.delete("/delete-video/{video_id}")
async def delete_video(video_id: str, payload: Dict = Body(...)):
    """
    Delete a video from S3 after verifying the password.
    Expects JSON payload: {"password": "your_password"}
    """
    try:
        # Verify password
        password = payload.get("password", "")
        if not password:
            raise HTTPException(status_code=400, detail="Password required")
        if hashlib.sha256(password.encode()).hexdigest() != CORRECT_PASSWORD_HASH:
            raise HTTPException(status_code=401, detail="Incorrect password")

        video_key = f"{S3_PREFIX}{video_id}"

        # Delete video from S3
        try:
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=video_key)
            logger.info(f"Successfully deleted video from S3: {video_key}")
        except ClientError as e:
            logger.error(f"Error deleting video {video_key}: {str(e)}")
            if e.response["Error"]["Code"] != "404":
                raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")

        return JSONResponse(content={"detail": "Video deleted successfully"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.post("/upload-video")
async def upload_video(
    video_file: UploadFile = File(...),
    password: str = Form(...),
    description: str = Form(...)
):
    """Upload a video to S3 under Amazingvideo/ prefix after password verification."""
    try:
        # Verify password
        if hashlib.sha256(password.encode()).hexdigest() != CORRECT_PASSWORD_HASH:
            raise HTTPException(status_code=401, detail="Incorrect password")

        # Validate video file
        video_content = await video_file.read()
        video_filename = video_file.filename
        if not video_filename.lower().endswith((".mp4", ".webm", ".ogg")):
            raise HTTPException(
                status_code=400,
                detail="Only MP4, WebM, or OGG videos are supported"
            )

        # Generate unique S3 key with hash prefix to prevent duplicates
        file_hash = hashlib.md5(video_content).hexdigest()
        s3_key = f"{S3_PREFIX}{file_hash}_{video_filename}"
        
        # Upload to S3 with description in metadata
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=video_content,
            Metadata={'description': description},
            ContentType=video_file.content_type
        )
        logger.info(f"Uploaded video to S3: {s3_key}")

        # Generate presigned URL for immediate access
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET_NAME, "Key": s3_key},
            ExpiresIn=36000  # 10 hours
        )

        return JSONResponse(content={
            "message": "Video uploaded successfully",
            "name": video_filename,
            "url": url,
            "description": description
        })

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))



def remove_background_rembg(image_bytes):
    """
    Remove background using rembg AI-based background remover.
    """
    try:
        output = remove(image_bytes)
        return io.BytesIO(output)
    except Exception as e:
        logger.error(f"Background removal failed: {str(e)}")
        raise ValueError(f"Background removal failed: {str(e)}")

@app.post("/remove_background")
async def remove_background_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to remove background from an uploaded image using rembg.
    Returns a PNG image with transparent background.
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")

        # Read image bytes
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")

        # Process image with rembg
        logger.info("Processing image for background removal")
        processed_image = remove_background_rembg(image_bytes)

        # Verify output
        if not processed_image.getvalue():
            raise HTTPException(status_code=500, detail="Failed to process image")

        # Return the processed image as a streaming response
        return StreamingResponse(
            content=processed_image,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=processed_image.png"}
        )

    except ValueError as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        logger.info(f"S3 cleanup: {'Running' if s3_keys else 'Skipping (no S3 keys)'}")
        for s3_key in s3_keys:
            cleanup_s3_file(s3_key)
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 


