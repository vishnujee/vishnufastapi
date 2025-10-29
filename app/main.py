from fastapi import FastAPI,Request, File, UploadFile, HTTPException, Form,Body,Response,BackgroundTasks,Depends
from fastapi.responses import HTMLResponse, FileResponse,StreamingResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import platform
import subprocess
import uuid
import aiofiles
import redis
import os
import boto3
import io
import logging
import pdfplumber
import tempfile
import requests
import psutil
import time
import PyPDF2
import re
import pandas as pd
import json
import asyncio
# Initialize colorama
from pinecone import Pinecone, ServerlessSpec
# from langchain_core.vectorstores.base import VectorStoreRetriever
# from pydantic import BaseModel
from botocore.exceptions import ClientError
import pathlib
import shutil

from pypdf import PdfReader
import gc

from typing import Any, Dict, List, Optional
from pydantic import Field,BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#langchain
# LangChain & Embeddings
# from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader,PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
#### for openai
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import hashlib
# from PyPDF2 import  PdfReader



########  load pdf library


# from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
# from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
# from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
# from adobe.pdfservices.operation.io.stream_asset import StreamAsset
# from adobe.pdfservices.operation.pdf_services import PDFServices
# from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
# from adobe.pdfservices.operation.pdfjobs.jobs.export_pdf_job import ExportPDFJob
# from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_params import ExportPDFParams
# from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_target_format import ExportPDFTargetFormat
# from adobe.pdfservices.operation.pdfjobs.result.export_pdf_result import ExportPDFResult
#############



from rembg import remove
from dotenv import load_dotenv
from colorama import Fore, Style, init
init() 
load_dotenv()


from app.pdf_operations import  (
    upload_to_s3, cleanup_s3_file,
    merge_pdfs_pypdf2, merge_pdfs_ghostscript, safe_compress_pdf, encrypt_pdf,
    convert_pdf_to_images, split_pdf, delete_pdf_pages,  convert_image_to_pdf, remove_pdf_password,reorder_pdf_pages,
    add_page_numbers, add_signature,remove_background_rembg,convert_pdf_to_ppt,convert_pdf_to_editable_ppt,
    estimate_compression_sizes,cleanup_local_files
)



app = FastAPI()





# Pre-compile patterns for better performance
BLOCKED_PATTERNS = [
    re.compile(r'wp-admin', re.IGNORECASE),
    re.compile(r'wordpress', re.IGNORECASE),
    re.compile(r'phpmyadmin', re.IGNORECASE),
    re.compile(r'administrator', re.IGNORECASE),
    re.compile(r'mysql', re.IGNORECASE),
    re.compile(r'sql', re.IGNORECASE),
]

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    path = request.url.path
    
    # Fast pattern matching
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(path):
            return JSONResponse(
                status_code=403,
                content={"detail": "Access forbidden"}
            )
    
    return await call_next(request)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename="logs/main.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
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
PDF_URL_TABULAR = "https://vishnufastapi.s3.ap-south-1.amazonaws.com/daily_pdfs/2.pdf"  # Your main PDF with tables
PDF_URL_NONTABULAR = "https://vishnufastapi.s3.ap-south-1.amazonaws.com/daily_pdfs/1.pdf"
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
        headers = {"User-Agent": os.getenv("USER_AGENT", "VishnuAI/1.0")}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")
    


######################################################### NEW IMPLEMENTATION FOR RAG #########################################################


# Configurable lists for table detection
HEADER_KEYWORDS = [
    "Company", "Duration", 
]

TITLE_KEYWORDS = [
    "Work Experience"
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

def extract_text_from_nontabular_pdf(pdf_bytes):
    """Improved PDF extraction that preserves both text and tables without over-filtering"""
    full_text = []
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages, 1):
                # Extract raw text first (preserve layout)
                text = page.extract_text(layout=True)
                if text:
                    full_text.append(f"--- PAGE {i} ---\n{text}")
                
                # Extract tables separately
                tables = page.extract_tables()
                for table in tables:
                    try:
                        # Convert table to markdown format
                        df = pd.DataFrame(table)
                        markdown_table = df.to_markdown(index=False)
                        full_text.append(f"\nTABLE (Page {i}):\n{markdown_table}\n")
                    except Exception as e:
                        # Fallback to raw table if conversion fails
                        full_text.append(f"\nTABLE_RAW (Page {i}):\n{str(table)}\n")
    
    except Exception as e:
        raise Exception(f"Error processing PDF: {e}")
    
    return "\n".join(full_text)

def extract_text_with_tables(pdf_bytes):
    """Enhanced PDF extraction with robust table detection"""
    full_text = []
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                table_content = set()  # Track all table-related content
                filtered_lines = []  # Non-table text
                cleaned_tables = []  # Collect cleaned tables

                # Extract tables
                tables = page.extract_tables() or []
                for table in tables:
                    cleaned_table = []
                    header_row = None
                    title_row = None

                    for row in table:
                        # Handle None cells
                        row = ["" if cell is None else str(cell).strip() for cell in row]
                        row_str = ' '.join(cell for cell in row if cell)
                        if not row_str.strip():
                            continue

                        # Normalize row string
                        norm_row_str = normalize_text(row_str)
                        table_content.add(norm_row_str)

                        # Add individual cell content, handling multi-line cells
                        for cell in row:
                            if cell:
                                cell_lines = cell.split('\n')
                                for cell_line in cell_lines:
                                    if cell_line.strip():
                                        table_content.add(normalize_text(cell_line))

                        if is_table_title(row_str, TITLE_KEYWORDS):
                            title_row = row_str
                            continue

                        if header_row is None and is_header_row(row_str, HEADER_KEYWORDS):
                            header_row = row
                            continue

                        cleaned_table.append(row)

                    if cleaned_table:
                        cleaned_tables.append((title_row, header_row, cleaned_table))

                # Process text lines (only if text exists)
                if text:
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        # Normalize line for comparison
                        norm_line = normalize_text(line)
                        
                        # Skip lines that are table-related
                        if (norm_line in table_content or
                            is_substring_match(line, table_content) or
                            is_table_title(line, TITLE_KEYWORDS) or
                            is_header_row(line, HEADER_KEYWORDS) or
                            is_raw_table_text(line)):
                            continue
                            
                        # # Additional table content checks
                        # if (re.match(r'^\d+\s', line) or  
                        #     any(x in line.lower() for x in ['no', 'nos', 'qty', 'km', 'mtr']) or
                        #     re.search(r'\d{4,}', line)):  # Long numbers
                        #     continue

                        filtered_lines.append(line)

                # Add filtered text to full_text
                if filtered_lines:
                    full_text.append(f"--- PAGE {i} ---\n" + "\n".join(filtered_lines))

                # Add tables to full_text
                for title_row, header_row, cleaned_table in cleaned_tables:
                    try:
                        # Ensure consistent column lengths
                        max_cols = max(len(row) for row in cleaned_table)
                        cleaned_table = [row + [""] * (max_cols - len(row)) for row in cleaned_table]

                        if header_row and cleaned_table:
                            df = pd.DataFrame(cleaned_table, columns=header_row)
                        else:
                            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0]) if len(cleaned_table) > 1 else pd.DataFrame(cleaned_table)

                        markdown_table = df.to_markdown(index=False)
                        if title_row:  # If we found a natural title
                            full_text.append(f"\n{title_row}\n{markdown_table}\n")
                        else:  # Otherwise use the default header
                            full_text.append(f"\nTABLE (Page {i}):\n{markdown_table}\n")
                    except Exception as e:
                        full_text.append(f"\nTABLE_RAW (Page {i}):\n{str(cleaned_table)}\n")

        return "\n".join(full_text)

    except pdfplumber.exceptions.PDFSyntaxError as e:
        raise Exception(f"Invalid PDF format: {e}")
    except Exception as e:
        raise Exception(f"Error processing PDF: {e}")



# ====================== Enhanced Logging Functions ======================

def log_retrieved_documents(docs: List[Document], query: str):
    """Log the top retrieved documents with full content"""
    logger.info(f"\n{'='*80}")
    logger.info(f"QUERY: {query}")
    logger.info(f"RETRIEVED {len(docs)} DOCUMENTS:")
    logger.info(f"{'='*80}")
    
    for i, doc in enumerate(docs, 1):
        logger.info(f"\n📄 DOCUMENT #{i}:")
        logger.info(f"📁 Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"📄 Content Type: {doc.metadata.get('content_type', 'Unknown')}")
        logger.info(f"📊 Score: {doc.metadata.get('score', 'N/A')}")
        logger.info(f"📝 Content (FULL):\n{doc.page_content}")
        logger.info(f"🏷️ Metadata: {doc.metadata}")
        logger.info(f"{'-'*60}")

def log_final_documents(docs: List[Document], query: str):
    """Log the documents being sent to LLM"""
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL DOCUMENTS BEING SENT TO LLM FOR QUERY: {query}")
    logger.info(f"Total documents: {len(docs)}")
    logger.info(f"{'='*80}")
    
    for i, doc in enumerate(docs, 1):
        logger.info(f"\n📤 DOCUMENT #{i} SENT TO LLM:")
        logger.info(f"📁 Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"📄 Content Type: {doc.metadata.get('content_type', 'Unknown')}")
        
        # Check for table markers
        has_table_start = "TABLE_START" in doc.page_content
        has_table_end = "TABLE_END" in doc.page_content
        logger.info(f"📊 Table Markers - START: {has_table_start}, END: {has_table_end}")
        
        if has_table_start and has_table_end:
            # Extract and log table content separately
            table_sections = doc.page_content.split("TABLE_START")
            for j, section in enumerate(table_sections[1:], 1):
                if "TABLE_END" in section:
                    table_content = section.split("TABLE_END")[0]
                    logger.info(f"📋 TABLE CONTENT #{j}:\n{table_content}")
        
        logger.info(f"📝 FULL CONTENT:\n{doc.page_content}")
        logger.info(f"🏷️ Metadata: {doc.metadata}")
        logger.info(f"{'-'*60}")

def log_embedding_process(documents: List[Document], source: str):
    """Log the embedding process for documents"""
    logger.info(f"\n{'='*80}")
    logger.info(f"EMBEDDING PROCESS FOR: {source}")
    logger.info(f"Total documents to embed: {len(documents)}")
    logger.info(f"{'='*80}")
    
    for i, doc in enumerate(documents, 1):
        logger.info(f"\n🔤 DOCUMENT #{i} TO BE EMBEDDED:")
        logger.info(f"📁 Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"📄 Content Type: {doc.metadata.get('content_type', 'Unknown')}")
        # logger.info(f"📝 Content Preview (first 500 chars):\n{doc.page_content[:500]}...")
        logger.info(f"📝 Full Content Preview:\n{doc.page_content}")
        logger.info(f"📏 Content Length: {len(doc.page_content)} characters")
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
        # logger.info(f"🔧 Processing document from {source}")

      #  logger.info(f"📄 FULL CONTENT:\n{content}")
        # Check if it's a work experience table
        is_work_table = (
            content.count("|") > 3 and 
            any(header in content for header in table_headers) and
            "2.pdf" in source  # Only process tables from the work experience PDF
        )

        if is_work_table:
            doc.metadata["content_type"] = "work_experience_table"
            
            # Extract and highlight relevant company information
            lines = content.split("\n")
            enhanced_content = []
            
            # Find and highlight matching companies
            for line in lines:
                if "|" in line and any(company in line.lower() for company in [
                    "kei", "larsen", "toubro", "vindhya", "punj", "gng"
                ]):
                    # Highlight relevant rows
                    enhanced_content.append(f"⭐ {line}")
                else:
                    enhanced_content.append(line)
            
            if enhanced_content:
                doc.page_content = "WORK_EXPERIENCE_TABLE_START\n" + "\n".join(enhanced_content) + "\nWORK_EXPERIENCE_TABLE_END"
            
            doc.metadata["table_type"] = "work_experience"
            
        else:
            doc.metadata["content_type"] = "text"

        processed.append(doc)

    return processed

def ensure_tabular_inclusion(docs, query, min_tabular=2):
    """Ensure relevant content is included based on query type"""
    query_lower = query.lower()
    
    # Check if query is about work/companies
    is_work_query = any(keyword in query_lower for keyword in [
        'company', 'work', 'experience', 'job', 'project',
        'kei', 'larsen', 'toubro', 'vindhya', 'punj', 'gng','l&t'
    ])
    
    # NEW: Check if query is about websites
    is_website_query = any(keyword in query_lower for keyword in [
        'website', 'site', 'url', 'link', 'web',
        'recallmind', 'parcelfile', 'vishnuji', 'file transfer',
        'cloud storage', 'pdf editing', 'portfolio'
    ])
    
    if is_work_query:
        tabular_docs = [d for d in docs if "2.pdf" in d.metadata.get("source", "")]
        other_docs = [d for d in docs if "2.pdf" not in d.metadata.get("source", "")]
        
        # Force include tabular docs first
        final_docs = tabular_docs[:min_tabular]
        
        # Add top-scoring other docs to reach desired count
        remaining_slots = 5 - len(final_docs)
        if remaining_slots > 0:
            final_docs.extend(other_docs[:remaining_slots])
        
        # logger.info(f"📊 Final docs for work query: {len(final_docs)} total, {len(tabular_docs)} from work experience PDF")
    
    elif is_website_query:
        # NEW: Prioritize website-related documents
        website_docs = [d for d in docs if any(keyword in d.page_content.lower() for keyword in [
            'recallmind', 'parcelfile', 'vishnuji.com', 'website', 'file transfer'
        ])]
        
        other_docs = [d for d in docs if d not in website_docs]
        
        # Include website docs first, then others
        final_docs = website_docs[:3]  # Get up to 3 website-specific docs
        remaining_slots = 5 - len(final_docs)
        if remaining_slots > 0:
            final_docs.extend(other_docs[:remaining_slots])
            
        # logger.info(f"🌐 Final docs for website query: {len(final_docs)} total, {len(website_docs)} website-specific")
    
    else:
        final_docs = docs[:5]  # Return top 5 for general queries
    
    return final_docs







###  PINECONE CLOUD 

class PineconeRetriever(BaseRetriever):
    index: Any = Field(...)
    embeddings: Any = Field(...)
    search_type: str = Field(default="similarity")
    search_kwargs: Optional[Dict] = Field(default_factory=lambda: {"k": 10})

    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        try:
            # logger.info(f"🔍 Starting retrieval for query: {query}")
            query_embedding = self.embeddings.embed_query(query)
            # logger.info(f"✅ Query embedding generated, length: {len(query_embedding)}")

             # Add timeout configuration
           
            
            # Configure session with timeout
            session = requests.Session()
            session.timeout = 30  # 30 second timeout
            
            # Add timeout and retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    results = self.index.query(
                        vector=query_embedding,
                        top_k=self.search_kwargs.get("k", 10),
                        include_metadata=True,
                        namespace="vishnu_ai_docs"
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"⚠️ Pinecone query attempt {attempt + 1} failed: {e}")
                    time.sleep(1)  # Wait before retry
            
            # logger.info(f"📊 Pinecone returned {len(results['matches'])} matches")
            
            documents = []
            for match in results["matches"]:
                text_content = match["metadata"].get("page_content", match["metadata"].get("text", ""))
                document = Document(
                    page_content=text_content,
                    metadata={
                        "source": match["metadata"].get("source", ""),
                        "page": match["metadata"].get("page", 0),
                        "score": match["score"],
                        "content_type": match["metadata"].get("content_type", "unknown"),
                        "document_type": match["metadata"].get("document_type", "unknown")
                    }
                )
                documents.append(document)
                # logger.info(f"📄 Retrieved doc - Score: {match['score']:.4f}, Source: {document.metadata['source']}")
            
            return documents
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}", exc_info=True)
            # Return empty list instead of raising exception to allow fallback
            return []

    def invoke(
        self, 
        input: str, 
        config: Optional[RunnableConfig] = None, 
        **kwargs
    ) -> List[Document]:
        """Handle invoke calls with config parameter"""
        return self._get_relevant_documents(input)

def initialize_vectorstore():
    try:
        logger.info("🚀 Starting vectorstore initialization...")
        os.environ['OPENAI_API_BASE'] = 'https://api.openai.com/v1'
        os.environ['NO_PROXY'] = 'api.openai.com'
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "vishnu-ai-docs"
        
        # Check if index exists
        if index_name not in pc.list_indexes().names():
            logger.info(f"📦 Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=512,  # Updated to match text-embedding-3-large. or small
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"✅ Created new index: {index_name}")
            time.sleep(20)
        
        index = pc.Index(index_name)
        # embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)  # Updated to large model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512,request_timeout=20,max_retries=1,)
  
        # Efficient empty check
        stats = index.describe_index_stats()
        namespace_stats = stats.get("namespaces", {}).get("vishnu_ai_docs", {})
        
        if namespace_stats.get("vector_count", 0) == 0:
            logger.info("📭 Index is empty, processing documents with enhanced extraction...")
            
            all_documents = []
            
            # Process Tabular PDF
            logger.info("📊 Processing tabular PDF...")
            pdf_bytes_tabular = download_from_url(PDF_URL_TABULAR)
            # logger.info(f"📥 Downloaded tabular PDF, size: {len(pdf_bytes_tabular)} bytes")
            
            # Use enhanced PDF extraction for tabular data
            tabular_text = extract_text_with_tables(pdf_bytes_tabular)
            # logger.info(f"📄 Extracted tabular text length: {len(tabular_text)} characters")
            
            # Create document with metadata for tabular content
            tabular_doc = Document(
                page_content=tabular_text,
                metadata={
                    "source": PDF_URL_TABULAR,
                    "content_type": "mixed_text_and_tables",
                    "document_type": "tabular",
                    "section": "work_experience",
                    "processed_at": time.time()
                }
            )
            all_documents.append(tabular_doc)
            
            # Process Non-Tabular PDF
            logger.info("📝 Processing non-tabular PDF...")
            try:
                pdf_bytes_nontabular = download_from_url(PDF_URL_NONTABULAR)
                # logger.info(f"📥 Downloaded non-tabular PDF, size: {len(pdf_bytes_nontabular)} bytes")
                
                # Use non-tabular extraction for better text preservation
                nontabular_text = extract_text_from_nontabular_pdf(pdf_bytes_nontabular)
                # logger.info(f"📄 Extracted non-tabular text length: {len(nontabular_text)} characters")
                
                # Split non-tabular content into logical sections
                sections = split_into_logical_sections(nontabular_text)
                
                for section_name, section_content in sections.items():
                    if section_content.strip():
                        section_doc = Document(
                            page_content=section_content,
                            metadata={
                                "source": PDF_URL_NONTABULAR,
                                "content_type": "text_heavy",
                                "document_type": "nontabular",
                                "section": section_name,
                                "processed_at": time.time()
                            }
                        )
                        all_documents.append(section_doc)
                        # logger.info(f"📑 Created section: {section_name} ({len(section_content)} chars)")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to process non-tabular PDF: {e}. Continuing with tabular PDF only.")
            
            # Define separate text splitters for tabular and non-tabular documents
            tabular_splitter = RecursiveCharacterTextSplitter(
                chunk_size=5090,  # Chunk size for tabular content
                chunk_overlap=300,
                separators=["\n\n", "\n", "•", " - ", "|", " "]
            )
            
            nontabular_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Chunk size for non-tabular content
                chunk_overlap=200,
                separators=["\n\n", "\n", "•", " - ", "|", " "]
            )
            
            # Split documents based on their type
            splits = []
            for doc in all_documents:
                if doc.metadata.get("document_type") == "tabular":
                    splits.extend(tabular_splitter.split_documents([doc]))
                else:
                    splits.extend(nontabular_splitter.split_documents([doc]))
            
            # logger.info(f"✂️ Split into {len(splits)} chunks from {len(all_documents)} documents")
            
            # Log embedding process
            log_embedding_process(splits, "combined_pdfs")
            
            # Batch processing with progress
            batch_size = 50
            total_vectors = 0


            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                texts = [doc.page_content for doc in batch]
                
                # logger.info(f"🔤 Embedding batch {i//batch_size + 1}/{(len(splits)-1)//batch_size + 1}")
                
                try:
                    # Batch embed for efficiency
                    embeddings_list = embeddings.embed_documents(texts)
                    
                    # ADD THESE LINES
                    # logger.info(f"🔤 Generated {len(embeddings_list)} embeddings for batch")
                    if embeddings_list:
                        logger.info(f"📏 First embedding dimensions: {len(embeddings_list[0])}")
                    
                    # logger.info(f"✅ Embedded {len(embeddings_list)} documents in batch")
                    
                    vectors = []
                    for j, (doc, embedding) in enumerate(zip(batch, embeddings_list)):
          
                        vectors.append({
                            "id": f"doc_{i+j}",
                            "values": embedding,
                            "metadata": {
                                "page_content": doc.page_content,  # ✅ NO TRUNCATION
                                "source": doc.metadata.get("source", "unknown"),
                                "content_type": doc.metadata.get("content_type", "mixed"),
                                "document_type": doc.metadata.get("document_type", "unknown"),
                                "section": doc.metadata.get("section", "general"),
                                "chunk_index": i+j
                            }
                        })
                                            
                    if vectors:
                        # logger.info(f"📤 About to upsert {len(vectors)} vectors to Pinecone")
                        
                        # Upsert to Pinecone
                        upsert_response = index.upsert(vectors=vectors, namespace="vishnu_ai_docs")
                        
                        # logger.info(f"📤 Successfully upserted {len(vectors)} vectors to Pinecone")
                        # logger.info(f"📊 Upsert response: {upsert_response}")
                        
                        total_vectors += len(vectors)
                        logger.info(f"📤 Upserted batch {i//batch_size + 1} with {len(vectors)} vectors")
                    else:
                        logger.warning("⚠️ No vectors to upsert in this batch")
                        
                except Exception as e:
                    logger.error(f"❌ Batch {i//batch_size + 1} failed: {e}", exc_info=True)
                    continue  # Continue with next batch
            
            
            
            # logger.info(f"🎉 Successfully upserted {total_vectors} documents to Pinecone from {len(all_documents)} source documents")
        
        logger.info("✅ Vectorstore initialization completed successfully")
        return index, embeddings
        
    except Exception as e:
        logger.error(f"❌ Vectorstore initialization failed: {e}", exc_info=True)
        return None, None

def split_into_logical_sections(text):
    """Split text into logical sections for better retrieval"""
    sections = {
        "personal_info": "",
        "education": "", 
        "work_experience": "",
        "skills": "",
        "awards": "",
        "websites": "",  # NEW: Add websites section
        "pdf_guide": "",
        "file_transfer": "",  # NEW: Specific section for file transfer
        "other": ""
    }
    
    lines = text.split('\n')
    current_section = "other"
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Detect section headers - ENHANCED WITH WEBSITE KEYWORDS
        if any(keyword in line_lower for keyword in ['about', 'personal', 'date of birth', 'hometown']):
            current_section = "personal_info"
        elif any(keyword in line_lower for keyword in ['education', 'qualification', '10th', '12th', 'b.tech']):
            current_section = "education"
        elif any(keyword in line_lower for keyword in ['experience', 'project', 'company', 'duration']):
            current_section = "work_experience"
        elif any(keyword in line_lower for keyword in ['skill', 'web development', 'ai', 'machine learning']):
            current_section = "skills"
        elif any(keyword in line_lower for keyword in ['award', 'recognition', 'trophy']):
            current_section = "awards"
        elif any(keyword in line_lower for keyword in ['website', 'recallmind', 'parcelfile', 'vishnuji.com']):
            current_section = "websites"
        elif any(keyword in line_lower for keyword in ['file transfer', 'p2p', 'cloud storage', 'parcelfile']):
            current_section = "file_transfer"
        elif any(keyword in line_lower for keyword in ['pdf', 'tool', 'guide', 'operation']):
            current_section = "pdf_guide"
        
        # Add line to current section
        if line.strip():
            sections[current_section] += line + "\n"
    
    return sections


# Initialize LLM
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
       # model="gemini-2.5-pro",
        temperature=0.3,
        max_tokens=3000,
        timeout=None,
        api_key=GOOGLE_API_KEY
    )

retriever = None
llm = None
thread_pool = ThreadPoolExecutor(max_workers=4)
# chat_history = []


# ====================== Startup Event ======================

@app.on_event("startup")
async def startup_event():
    global retriever, llm
    logger.info("🚀 Starting AI services initialization...")
    setup_directories()
    
    try:
        # Initialize only once at startup
        index, embeddings = initialize_vectorstore()
        retriever = PineconeRetriever(
            index=index,
            embeddings=embeddings,
            search_type="similarity",
            search_kwargs={
                "k": 10,  # Increased for better coverage
                "score_threshold": 0.3
            }
        )
        llm = get_llm()
        logger.info("✅ AI services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Startup initialization failed: {e}", exc_info=True)
        # Set fallback values
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        
        class FallbackRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, *, run_manager = None) -> List[Document]:
                logger.warning("⚠️ Using fallback retriever")
                return [Document(page_content="System initializing...", metadata={})]
        
        retriever = FallbackRetriever()
        llm = get_llm()


# @app.get("/memory-usage")
# async def memory_usage_stream(request: Request):
#     """Stream memory usage data every second"""
#     async def event_stream():
#         while True:
#             if await request.is_disconnected():
#                 break
            
#             # Get memory info
#             mem = psutil.virtual_memory()
#             swap = psutil.swap_memory()
            
#             data = {
#                 "ram": {
#                     "total": mem.total,
#                     "used": mem.used,
#                     "free": mem.free,
#                     "percent": mem.percent
#                 },
#                 "rom": {
#                     "total": swap.total,
#                     "used": swap.used,
#                     "free": swap.free,
#                     "percent": swap.percent
#                 }
#             }
            
#             yield f"data: {json.dumps(data)}\n\n"
#             await asyncio.sleep(1)
    
#     return StreamingResponse(
#         event_stream(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#         }
#     )



def analyze_table_chunking(docs, query):
    """Analyze how table content is chunked across documents"""
    table_docs = [d for d in docs if "2.pdf" in d.metadata.get("source", "")]
    
    # logger.info(f"📊 TABLE CHUNK ANALYSIS FOR QUERY: '{query}'")
    # logger.info(f"📋 Found {len(table_docs)} documents from work experience PDF")
    
    for i, doc in enumerate(table_docs, 1):
        content = doc.page_content
        # logger.info(f"\n🔍 TABLE CHUNK #{i}:")
        # logger.info(f"📏 Content length: {len(content)} characters")
        # logger.info(f"📄 Metadata: {doc.metadata}")
        
        # Check table structure
        lines = content.split('\n')
        table_lines = [line for line in lines if '|' in line]
        # logger.info(f"📊 Table lines count: {len(table_lines)}")
        
        # Check for specific companies
        companies_to_check = ['KEI', 'Larsen', 'Toubro', 'Vindhya', 'Punj', 'GNG']
        found_companies = []
        for company in companies_to_check:
            if company.lower() in content.lower():
                found_companies.append(company)
        
        # logger.info(f"🏢 Companies found in this chunk: {found_companies}")
        
        # Log first 500 chars of content
        preview = content[:500] + "..." if len(content) > 500 else content
        # logger.info(f"👀 Content preview:\n{preview}")
    
    return table_docs

def check_complete_table_in_vectorstore():
    """Check if the complete table exists as a single chunk in vector store"""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("vishnu-ai-docs")
        
        # Get all documents from PDF 2
        results = index.query(
            vector=[0] * 512,  # Dummy vector to get all docs
            top_k=100,
            include_metadata=True,
            namespace="vishnu_ai_docs",
            filter={
                "source": {"$eq": "https://vishnufastapi.s3.ap-south-1.amazonaws.com/daily_pdfs/2.pdf"}
            }
        )
        
        # logger.info("🔍 COMPLETE TABLE CHUNK ANALYSIS IN VECTOR STORE")
        # logger.info("=" * 80)
        
        all_companies = ['KEI', 'Larsen', 'Toubro', 'Vindhya', 'Punj', 'GNG']
        found_companies = set()
        
        for i, match in enumerate(results['matches'], 1):
            content = match['metadata'].get('page_content', '')
            # logger.info(f"\n📄 CHUNK #{i}:")
            # logger.info(f"📏 Length: {len(content)} chars")
            # logger.info(f"📊 Score: {match['score']:.4f}")
            
            # Check companies in this chunk
            chunk_companies = [c for c in all_companies if c.lower() in content.lower()]
            # logger.info(f"🏢 Companies: {chunk_companies}")
            found_companies.update(chunk_companies)
            
            # Show FULL content for table chunks
            # if "TABLE" in content:
            #     logger.info(f"📋 FULL TABLE CONTENT:\n{content}")
            # else:
            #     logger.info(f"📝 Content preview: {content[:200]}...")
            

            
            chunkresults = index.query(
                vector=[0] * 512,    # dummy vector
                top_k=1000,           # increase this if you have more chunks
                include_metadata=True,
                namespace="vishnu_ai_docs"
            )
            # logger.info("*$" * 50)
            # for i, match in enumerate(chunkresults['matches'], 1):
            #     print(f"\n--- CHUNK #{i} ---")
            #     print(f"ID: {match['id']}")
            #     print(f"Content:\n{match['metadata'].get('page_content', '')}\n")



            # logger.info("*$" * 50)
        
        # logger.info(f"\n🎯 SUMMARY:")
        # logger.info(f"Total table chunks: {len(results['matches'])}")
        # logger.info(f"Companies found across all chunks: {sorted(found_companies)}")
        # logger.info(f"Missing companies: {set(all_companies) - found_companies}")
        
        return len(results['matches']), found_companies
        
    except Exception as e:
        logger.error(f"❌ Vector store analysis failed: {e}")
        return 0, set()


def quick_table_analysis(retriever, query):
    """Fast table analysis - remove if you want maximum speed"""
    try:
        query_embedding = retriever.embeddings.embed_query(query)
        all_table_results = retriever.index.query(
            vector=query_embedding,
            top_k=5,  # Reduced from 20
            include_metadata=True,
            namespace="vishnu_ai_docs",
            filter={
                "source": {"$eq": "https://vishnufastapi.s3.ap-south-1.amazonaws.com/daily_pdfs/2.pdf"}
            }
        )
        # logger.info(f"📋 Table docs in store: {len(all_table_results['matches'])}")
    except Exception:
        pass  # Silent fail - this is just diagnostic


@app.get("/", response_class=HTMLResponse)
async def serve_index():
   
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())



DASHBOARD_PASSWORD = os.getenv("CLEANUP_DASHBOARD_PASSWORD",)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
security = HTTPBasic()

def verify_dashboard_access(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify dashboard access with password from .env"""
    correct_username = "admin"  # You can make this configurable too
    correct_password = DASHBOARD_PASSWORD
    
    is_correct_username = secrets.compare_digest(credentials.username, correct_username)
    is_correct_password = secrets.compare_digest(credentials.password, correct_password)
    
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

@app.get("/cleanup", response_class=HTMLResponse)
async def cleanup_dashboard(auth: bool = Depends(verify_dashboard_access)):
    """Serve the comprehensive cleanup dashboard - PASSWORD PROTECTED"""
    cleanup_path = os.path.join(static_dir, "cleanup.html")
    with open(cleanup_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())





@app.get("/cleanup-logs")
async def get_cleanup_logs():
    """Get cleanup logs as JSON for the dashboard"""
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

@app.post("/clear-cleanup-logs")
async def clear_cleanup_logs():
    """Clear the cron cleanup logs"""
    try:
        log_path = "/home/ubuntu/cron_cleanup.log"
        if os.path.exists(log_path):
            os.remove(log_path)
        return {"message": "Logs cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing logs: {str(e)}")


@app.post("/test-cleanup")
async def test_cleanup():
    """Manual cleanup trigger for testing"""
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

@app.get("/cleanup-status")
async def get_cleanup_status():
    """Get current cleanup status and file statistics"""
    try:
        current_time = time.time()
        stats = {
            "last_cleanup_time": _last_cleanup_time if '_last_cleanup_time' in globals() else 0,
            "current_time": current_time,
            "next_cleanup_in": max(0, (_last_cleanup_time + 900) - current_time) if '_last_cleanup_time' in globals() else 0,
            "directories": {}
        }
        
        # Count files in each directory
        for root_dir in [UPLOAD_DIR, OUTPUT_DIR, ESTIMATION_DIR, PDFTOWORD, TEMP_DIR]:
            if root_dir.exists():
                files = list(root_dir.iterdir())
                stats["directories"][root_dir.name] = {
                    "total_files": len([f for f in files if f.is_file()]),
                    "total_dirs": len([f for f in files if f.is_dir()]),
                    "old_files": len([f for f in files if f.is_file() and f.stat().st_mtime < (current_time - 900)])
                }
        
        return stats
    except Exception as e:
        logger.error(f"❌ Cleanup status check failed: {e}")
        return {"error": str(e)}

@app.get("/debug-table-chunking")
async def debug_table_chunking():
    """Debug endpoint to check table chunking"""
    chunk_count, companies = check_complete_table_in_vectorstore()
    
    return {
        "total_table_chunks": chunk_count,
        "companies_found": list(companies),
        "message": "Check logs for detailed analysis"
    }


@app.post("/debug-retrieval")
async def debug_retrieval(query: str = Form(...)):
    """Debug endpoint to see what documents are retrieved for a query"""
    if not retriever:
        return {"error": "Retriever not initialized"}
    
    raw_docs = retriever.invoke(query)
    
    debug_info = {
        "query": query,
        "total_docs_retrieved": len(raw_docs),
        "docs": []
    }
    
    for i, doc in enumerate(raw_docs):
        debug_info["docs"].append({
            "rank": i + 1,
            "source": doc.metadata.get("source", ""),
            "content_type": doc.metadata.get("content_type", "unknown"),
            "score": doc.metadata.get("score", "N/A"),
            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "has_website_keywords": any(keyword in doc.page_content.lower() for keyword in [
                'recallmind', 'parcelfile', 'file transfer', 'website'
            ])
        })
    
    return debug_info




class ChatRequest(BaseModel):
    query: str
    mode: str = None  # Optional chat mode selection
    history: str = None

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
    
    # logger.info(f"🎯 CHAT QUERY: '{query}' | Mode: {mode if mode else 'Default'} | History: {'Provided' if history else 'None'}") 
    
    # ✅ ADD THIS: Parse and limit history to last 4 conversations (8 messages)
    limited_history = []
    if history:
        try:
            history_data = json.loads(history)
            # Keep only last 6 messages (3 conversations)
            limited_history = history_data[-6:]
            # logger.info(f"📝 HISTORY RECEIVED: {len(history_data)} messages, LIMITED TO: {len(limited_history)} messages")
        except Exception as e:
            # logger.warning(f"Failed to parse chat history: {e}")
            limited_history = []
    else:
        logger.info(f"📝 HISTORY RECEIVED: None")
    
    start_time = time.time()
    timings = {}

    try:
        loop = asyncio.get_event_loop()

        if mode and mode in CHAT_MODES:
            # Bypass RAG: Use mode-specific prompt and call LLM directly
            system_prompt = CHAT_MODES[mode]["prompt"]
            
            # Process LIMITED chat history
            messages = [("system", system_prompt)]
            
            # ✅ USE LIMITED_HISTORY instead of full history
            if limited_history:
                # logger.info(f"🧠 MODE-SPECIFIC HISTORY: {len(limited_history)} messages")
                # Add conversation history to messages
                for msg in limited_history:
                    if msg["role"] == "user":
                        messages.append(("human", msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(("ai", msg["content"]))
            
            # Add current user message
            messages.append(("human", query))
            
            # logger.info(f"📨 FINAL MESSAGES TO LLM: {len(messages)} total messages")
            
      
            
            generation_start = time.time()
            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        thread_pool,
                        lambda: llm.invoke(messages)
                    ),
                    timeout=25.0
                )


                answer = response.content if hasattr(response, 'content') else str(response)
                # logger.info(f"✅ LLM generation completed in mode '{mode}', response length: {len(answer)}")
            except asyncio.TimeoutError:
                # logger.warning("⏰ LLM generation timeout in mode")
                answer = "I'm taking too long to generate a response. Please try again."
            generation_end = time.time()
            timings["generation_time"] = generation_end - generation_start
            timings["retrieval_time"] = 0.0
            timings["processing_time"] = 0.0
            raw_docs = []
            processed_docs = []
      
        else:
         
          # ---------------- OPTIMIZED RETRIEVAL ----------------
            retrieval_start = time.time()
            logger.info("🔍 Starting document retrieval...")

            try:
                raw_docs = await asyncio.wait_for(
                    loop.run_in_executor(
                        thread_pool, 
                        lambda: retriever.invoke(query) if retriever else []
                    ),
                    timeout=45.0
                )
            except asyncio.TimeoutError:
                # logger.warning(f"⏰ Retrieval timeout for query: {query}")
                raw_docs = []
            except Exception as e:
                # logger.error(f"❌ Retrieval error: {e}")
                raw_docs = []

            retrieval_end = time.time()
            timings["retrieval_time"] = retrieval_end - retrieval_start
            
            # if raw_docs:
            #     logger.info(f"✅ Retrieval completed in {timings['retrieval_time']:.2f}s, found {len(raw_docs)} documents")
            # else:
            #     logger.warning(f"⚠️ Retrieval completed in {timings['retrieval_time']:.2f}s, but found 0 documents")


            # ---------------- DOCUMENT PROCESSING ----------------
            processing_start = time.time()
            
            final_docs = ensure_tabular_inclusion(raw_docs, query, min_tabular=2)
            processed_docs = post_process_retrieved_docs(final_docs, query)
            
            processing_end = time.time()
            timings["processing_time"] = processing_end - processing_start
            # logger.info(f"✅ Document processing completed in {timings['processing_time']:.2f}s")

            # ---------------- GENERATION ----------------
            generation_start = time.time()
            
            if not processed_docs:
                answer = "I couldn't find specific information about that in my knowledge base. Is there anything else I can help you with?"
                # logger.warning("⚠️ No relevant documents found for query")
            else:
                table_context = any("2.pdf" in doc.metadata.get("source", "") for doc in processed_docs)
                # if table_context:
                #     logger.info("✅ TABLE DATA IS IN LLM CONTEXT!")
                # else:
                #     logger.info("ℹ️ No table data in context for this query")

    # ✅ LOG FINAL DOCUMENTS GOING TO LLM
                # logger.info("📤 FINAL DOCUMENTS BEING SENT TO LLM:")
                # for i, doc in enumerate(processed_docs, 1):
                #     logger.info(f"📄 Document {i}/{len(processed_docs)}:")
                #     logger.info(f"   Source: {doc.metadata.get('source', 'unknown')}")
                #     logger.info(f"   Content Type: {doc.metadata.get('content_type', 'unknown')}")
                #     logger.info(f"   Content Preview: {doc.page_content}...")
                #     logger.info(f"   Full Content Length: {len(doc.page_content)} chars")
                #     logger.info("   ---")





                # ✅ PREPARE LLM CHAIN (fast - no need for parallel)
                # prompt = ChatPromptTemplate.from_messages([
                #     ("system", "You are a helpful VISHNU AI assistant. Provide direct, conversational answers."),
                #     ("human", "Context: {context}\n\nQuestion: {input}\nAnswer:")
                # ])
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are Vishnu AI assistant — concise, friendly, and accurate. Give clear, human-like answers."),
                    ("human", "Context: {context}\n\nQuestion: {input}\nAnswer:")
                ])


                question_answer_chain = create_stuff_documents_chain(llm, prompt)

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
                    # logger.info(f"✅ LLM generation completed, response length: {len(answer)}")
                except asyncio.TimeoutError:
                    # logger.warning("⏰ LLM generation timeout")
                    answer = "I'm taking too long to generate a response. Please try again."

            generation_end = time.time()
            timings["generation_time"] = generation_end - generation_start
            # logger.info(f"✅ Generation completed in {timings['generation_time']:.2f}s")

        # ---------------- RESPONSE ----------------
        chat_history = []
        chat_entry = f"You: {query}\nAI: {answer}"
        chat_history.insert(0, chat_entry)
        if len(chat_history) > 3:
            chat_history.pop()

        total_end = time.time()
        timings["total_time"] = total_end - start_time
        # logger.info(f"🎉 Total processing time: {timings['total_time']:.2f}s")

        return {
            "answer": answer,
            "history": "\n\n".join(chat_history),
            "timings": {k: f"{v:.2f}s" for k, v in timings.items()},
            "retrieved_docs_count": len(raw_docs),
            "processed_docs_count": len(processed_docs)
        }

    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        return {
            "answer": "I'm experiencing technical issues. Please try again in a moment.",
            "history": "",
            "error": True
        }






@app.post("/merge_pdf")
async def merge_pdfs(files: List[UploadFile] = File(...), method: str = Form("PyPDF2"), file_order: Optional[str] = Form(None)):
    logger.info(f"Received merge request with {len(files)} files, method: {method}, file_order: {file_order}")
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 PDF files required")
    
    total_size_mb = sum(file.size for file in files) / (1024 * 1024)
    if method == "PyPDF2" and (len(files) > 51 or total_size_mb > 90):
        raise HTTPException(status_code=400, detail="PyPDF2 limits: 51 files, 90MB total")
    if method == "Ghostscript" and (len(files) > 30 or total_size_mb > 50):
        raise HTTPException(status_code=400, detail="Ghostscript limits: 30 files, 50MB total")

    # Reorder files based on file_order
    if file_order:
        try:
            order = [int(i) for i in file_order.split(',')]
            if len(order) != len(files) or not all(0 <= i < len(files) for i in order):
                raise HTTPException(status_code=400, detail="Invalid file order")
            files = [files[i] for i in order]
            logger.info(f"Reordered files: {[file.filename for file in files]}")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid file order format")

    s3_keys = []
    try:
        file_contents = []
        for file in files:
            file_content = await file.read()
            s3_key = upload_to_s3(file_content, file.filename)
            s3_keys.append(s3_key)
            file_contents.append(file_content)
        
        merged_pdf = None
        if method == "PyPDF2":
            logger.info("Merging with PyPDF2")
            merged_pdf = merge_pdfs_pypdf2(file_contents)
        else:
            logger.info("Merging with Ghostscript")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_output:
                output_path = tmp_output.name
            merged_pdf = merge_pdfs_ghostscript(file_contents, output_path)
        
        if not merged_pdf:
            logger.error("Merge failed: No output produced")
            raise HTTPException(status_code=500, detail="PDF merge failed")
        
        logger.info("Merge successful, returning response")
        return StreamingResponse(
            io.BytesIO(merged_pdf),
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="merged_output.pdf"'}
        )
    except Exception as e:
        logger.error(f"Merge error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF merge failed: {str(e)}")
    # finally:
    #     for s3_key in s3_keys:
    #         cleanup_s3_file(s3_key)
    #     gc.collect()
    finally:
        logger.info(f"S3 cleanup: {'Running' if s3_keys else 'Skipping (no S3 keys)'}")
        for s3_key in s3_keys:
            cleanup_s3_file(s3_key)
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()


######################################################################################################################################
######################################################################################################################################
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

def upload_to_s3(file_path: str, filename: str) -> str:
    """Upload file to S3 from disk path without loading in RAM."""
    s3_key = f"temp_uploads/{uuid.uuid4()}_{filename}"
    logger.info(f"Uploading to S3: {s3_key}")
    
    try:
        s3_client.upload_file(file_path, BUCKET_NAME, s3_key)
        logger.info(f"Uploaded to S3: {s3_key}")
        return s3_key
    except Exception as e:
        logger.error(f"Failed to upload to S3: {str(e)}")
        raise

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


#########################################################################################################################################################

@app.post("/encrypt_pdf")
async def encrypt_pdf_endpoint(file: UploadFile = File(...), password: str = Form(...)):
    logger.info(f"Received encrypt request for {file.filename}")
    if file.size / (1024 * 1024) > 50:
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")
    if not password:
        raise HTTPException(status_code=400, detail="Password required")

    s3_key = None
    try:
        file_content = await file.read()
        if USE_S3:
            s3_key = upload_to_s3(file_content, file.filename)
        else:
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)
        encrypted_pdf = encrypt_pdf(file_content, password)
        if not encrypted_pdf:
            raise HTTPException(status_code=500, detail="Encryption failed")

        return StreamingResponse(
            io.BytesIO(encrypted_pdf),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="encrypted_{file.filename}"'}
        )
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF encryption failed: {str(e)}")
    finally:
        logger.info(f"S3 cleanup: {'Running' if s3_key else 'Skipping (no S3 key)'}")
        if s3_key:
            cleanup_s3_file(s3_key)
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()

@app.post("/convert_pdf_to_images")
async def convert_pdf_to_images_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received convert to images request for {file.filename}")
    if file.size / (1024 * 1024) > 50:
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")

    s3_key = None
    try:
        file_content = await file.read()
        if USE_S3:
            s3_key = upload_to_s3(file_content, file.filename)
        else:
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)
        zip_bytes = convert_pdf_to_images(file_content)
        if not zip_bytes:
            raise HTTPException(status_code=500, detail="Conversion failed")

        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="pdf_images.zip"'}
        )
    except Exception as e:
        logger.error(f"PDF to images failed: {e}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
    finally:
        logger.info(f"S3 cleanup: {'Running' if s3_key else 'Skipping (no S3 key)'}")
        if s3_key:
            cleanup_s3_file(s3_key)
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()

@app.post("/split_pdf")
async def split_pdf_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received split request for {file.filename}")
    if file.size / (1024 * 1024) > 100:
        raise HTTPException(status_code=400, detail="File exceeds 100MB limit")

    s3_key = None
    try:
        file_content = await file.read()
        if USE_S3:
            s3_key = upload_to_s3(file_content, file.filename)
        else:
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)
        zip_bytes, total_pages = split_pdf(file_content)
        if not zip_bytes:
            raise HTTPException(status_code=500, detail="Split failed")

        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="split_pages.zip"'}
        )
    except Exception as e:
        logger.error(f"Split error: {e}")
        raise HTTPException(status_code=500, detail=f"Split failed: {str(e)}")
    finally:
        logger.info(f"S3 cleanup: {'Running' if s3_key else 'Skipping (no S3 key)'}")
        if s3_key:
            cleanup_s3_file(s3_key)
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()

@app.post("/delete_pdf_pages")
async def delete_pdf_pages_endpoint(file: UploadFile = File(...), pages: str = Form(...)):
    logger.info(f"Received delete pages request for {file.filename}")
    if file.size / (1024 * 1024) > 55:
        raise HTTPException(status_code=400, detail="File exceeds 55MB limit")

    try:
        pages_to_delete = set(int(p) for p in pages.split(",") if p.strip().isdigit())
        if not pages_to_delete:
            raise HTTPException(status_code=400, detail="Invalid page numbers")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid page format. Use comma-separated numbers (e.g., 2,5,7)")

    s3_key = None
    try:
        file_content = await file.read()
        if USE_S3:
            s3_key = upload_to_s3(file_content, file.filename)
        else:
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)
        modified_pdf = delete_pdf_pages(file_content, pages_to_delete)
        if not modified_pdf:
            raise HTTPException(status_code=500, detail="Page deletion failed")

        return StreamingResponse(
            io.BytesIO(modified_pdf),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="modified_{file.filename}"'}
        )
    except Exception as e:
        logger.error(f"Delete pages error: {e}")
        raise HTTPException(status_code=500, detail=f"Page deletion failed: {str(e)}")
    finally:
        logger.info(f"S3 cleanup: {'Running' if s3_key else 'Skipping (no S3 key)'}")
        if s3_key:
            cleanup_s3_file(s3_key)
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()




##########################################



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

@app.post("/convert_pdf_to_ppt")
async def convert_pdf_to_ppt_endpoint(
    file: UploadFile = File(...),
    conversion_type: str = Form("image")
):
    logger.info(f"Received convert to PPT request for {file.filename} (type: {conversion_type})")
    if file.size / (1024 * 1024) > 50:
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")

    s3_key = None
    try:
        file_content = await file.read()
        if USE_S3:
            s3_key = upload_to_s3(file_content, file.filename)
        else:
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)
        
        if conversion_type == "editable":
            ppt_bytes = convert_pdf_to_editable_ppt(file_content)
            filename = "editable_output.pptx"
        else:
            ppt_bytes = convert_pdf_to_ppt(file_content)
            filename = "image_based_output.pptx"
        
        if not ppt_bytes:
            raise HTTPException(status_code=500, detail="Conversion failed")

        return StreamingResponse(
            io.BytesIO(ppt_bytes),
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        logger.error(f"PDF to PPT error ({conversion_type}): {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"{conversion_type.capitalize()} conversion failed: {str(e)}"
        )
    finally:
        logger.info(f"S3 cleanup: {'Running' if s3_key else 'Skipping (no S3 key)'}")
        if s3_key:
            cleanup_s3_file(s3_key)
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()




@app.post("/convert_image_to_pdf")
async def convert_image_to_pdf_endpoint(
    file: UploadFile = File(...),
    page_size: str = Form("A4"),
    orientation: str = Form("Portrait"),
    description: str = Form(""),
    description_position: str = Form("bottom"),
    description_font_size: int = Form(12),
    custom_x: float = Form(None),
    custom_y: float = Form(None),
    font_color: str = Form("#000000"),  
    font_family: str = Form("helv"),    
    font_weight: str = Form("normal"),  
):
    # Input validation
    if file.size / (1024 * 1024) > 50:
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")
    if page_size not in ["A4", "Letter"]:
        raise HTTPException(status_code=400, detail="Invalid page size. Choose: A4, Letter")
    if orientation not in ["Portrait", "Landscape"]:
        raise HTTPException(status_code=400, detail="Invalid orientation. Choose: Portrait, Landscape")
    if description_position not in ["top", "bottom", "top-center", "top-left", "top-right", 
                                  "bottom-left", "bottom-center", "bottom-right", "custom"]:
        raise HTTPException(status_code=400, detail="Invalid description position")
    if description_position == "custom" and (custom_x is None or custom_y is None):
        raise HTTPException(status_code=400, detail="Custom position requires both X and Y coordinates")

    # Convert HEX color to RGB tuple (0-1 range)
    try:
        hex_color = font_color.lstrip('#')
        if len(hex_color) == 3:  # Handle shorthand #RGB
            hex_color = ''.join([c*2 for c in hex_color])
        rgb_color = tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    except Exception as e:
        logger.warning(f"Invalid color {font_color}, defaulting to black. Error: {str(e)}")
        rgb_color = (0, 0, 0)

    # Terminal logging with colored output
    print("\n" + "="*50)
    print(f"{Fore.YELLOW}📝 PDF Conversion Parameters:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}• Filename:{Style.RESET_ALL} {file.filename}")
    print(f"{Fore.CYAN}• Page Size:{Style.RESET_ALL} {page_size}")
    print(f"{Fore.CYAN}• Orientation:{Style.RESET_ALL} {orientation}")
    print(f"{Fore.CYAN}• Description:{Style.RESET_ALL} '{description}'")
    print(f"{Fore.CYAN}• Position:{Style.RESET_ALL} {description_position}")
    print(f"{Fore.CYAN}• Font Size:{Style.RESET_ALL} {description_font_size}pt")
    print(f"{Fore.CYAN}• Custom Coords:{Style.RESET_ALL} X={custom_x}, Y={custom_y}")
    print(f"{Fore.CYAN}• Font Color:{Style.RESET_ALL} {font_color} (RGB: {rgb_color})")
    print(f"{Fore.CYAN}• Font Family:{Style.RESET_ALL} {font_family}")
    print(f"{Fore.CYAN}• Font Weight:{Style.RESET_ALL} {font_weight}")
    print("="*50 + "\n")



    s3_key = None
    try:
        file_content = await file.read()
        
        if USE_S3:
            s3_key = upload_to_s3(file_content, file.filename)
        else:
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)
        
        pdf_bytes = convert_image_to_pdf(
            image_bytes=file_content,
            page_size=page_size,
            orientation=orientation,
            description=description,
            description_position=description_position.lower(),  # Ensure lowercase
            description_font_size=description_font_size,
            custom_x=custom_x,
            custom_y=custom_y,
            font_color=rgb_color,  # Pass the converted RGB
            font_family=font_family,
            font_weight=font_weight
        )
        
        if not pdf_bytes:
            raise HTTPException(status_code=500, detail="Conversion failed")

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{file.filename.split(".")[0]}.pdf"'}
        )

    except Exception as e:
        logger.error(f"🚨 Conversion failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")
    finally:
        if USE_S3 and s3_key:
            cleanup_s3_file(s3_key)
        elif not USE_S3:
            cleanup_local_files()
        gc.collect()


@app.post("/remove_pdf_password")
async def remove_pdf_password_endpoint(file: UploadFile = File(...), password: str = Form(...)):
    logger.info(f"Received remove password request for {file.filename}")
    if file.size / (1024 * 1024) > 50:
        logger.error(f"File {file.filename} exceeds 50MB limit")
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")
    if not password:
        logger.error("Empty password provided")
        raise HTTPException(status_code=400, detail="Password cannot be empty")

    s3_key = None
    try:
        file_content = await file.read()
        if not file_content:
            logger.error(f"Empty file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        if USE_S3:
            s3_key = upload_to_s3(file_content, file.filename)
        else:
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)
        decrypted_pdf = remove_pdf_password(file_content, password)
        if decrypted_pdf is None:
            logger.error("remove_pdf_password returned None")
            raise HTTPException(status_code=500, detail="Password removal failed")

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

@app.post("/reorder_pages")
async def reorder_pages(file: UploadFile = File(...), page_order: str = Form(...)):
    logger.info(f"Received reorder pages request for {file.filename}, page_order: {page_order}")
    try:
        file_content = await file.read()
        if not USE_S3:
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        num_pages = len(pdf_reader.pages)
        
        page_indices = [int(p) - 1 for p in page_order.split(",") if p.strip()]
        if not page_indices or len(page_indices) != num_pages:
            raise ValueError("Page order must include all pages exactly once")
        if any(idx < 0 or idx >= num_pages for idx in page_indices):
            raise ValueError("Invalid page numbers")
        if len(set(page_indices)) != len(page_indices):
            raise ValueError("Duplicate page numbers detected")
        
        pdf_writer = PyPDF2.PdfWriter()
        for idx in page_indices:
            pdf_writer.add_page(pdf_reader.pages[idx])
        
        output = io.BytesIO()
        pdf_writer.write(output)
        output.seek(0)
        
        def iterfile():
            yield output.getvalue()
            output.close()
        
        return StreamingResponse(
            iterfile(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=reordered_{file.filename}"}
        )
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
    finally:
        logger.info("No S3 cleanup needed (in-memory processing)")
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()

@app.post("/add_page_numbers")
async def add_page_numbers_endpoint(
    file: UploadFile = File(...),
    position: str = Form("bottom"),
    alignment: str = Form("center"),
    format: str = Form("page_x")
):
    logger.info(f"Received add page numbers request for {file.filename}")
    try:
        file_content = await file.read()
        if not USE_S3:
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)
        result = add_page_numbers(file_content, position, alignment, format)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to add page numbers")
        return StreamingResponse(
            io.BytesIO(result),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=numbered_{file.filename}"}
        )
    except Exception as e:
        logger.error(f"Error adding page numbers: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding page numbers: {str(e)}")
    finally:
        logger.info("No S3 cleanup needed (in-memory processing)")
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()

@app.post("/add_signature")
async def add_signature_endpoint(
    pdf_file: UploadFile = File(...),
    signature_file: UploadFile = File(...),
    specific_pages: str = Form(...),
    size: str = Form(...),
    position: str = Form(...),
    alignment: str = Form(...),
    remove_bg: bool = Form(False)
):
    logger.info(f"Received add signature request: pdf={pdf_file.filename}, signature={signature_file.filename}")
    try:
        pdf_bytes = await pdf_file.read()
        signature_bytes = await signature_file.read()
        if not USE_S3:
            pdf_path = os.path.join("input_pdfs", f"{hashlib.md5(pdf_bytes).hexdigest()}_{pdf_file.filename}")
            sig_path = os.path.join("input_pdfs", f"{hashlib.md5(signature_bytes).hexdigest()}_{signature_file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            with open(sig_path, "wb") as f:
                f.write(signature_bytes)

        if not pdf_file.filename.lower().endswith('.pdf'):
            logger.error("Invalid PDF file type")
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files allowed.")
        if not signature_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.error("Invalid signature file type")
            raise HTTPException(status_code=400, detail="Invalid signature file type. Only PNG or JPEG allowed.")
        if len(pdf_bytes) > 50 * 1024 * 1024:
            logger.error("PDF file too large")
            raise HTTPException(status_code=400, detail="PDF file size exceeds 50MB.")
        if len(signature_bytes) > 10 * 1024 * 1024:
            logger.error("Signature file too large")
            raise HTTPException(status_code=400, detail="Signature file size exceeds 10MB.")

        try:
            pages = [int(p) for p in specific_pages.split(',')] if specific_pages else []
            logger.info(f"Pages to sign: {pages}")
        except ValueError:
            logger.error("Invalid page numbers format")
            raise HTTPException(status_code=400, detail="Invalid page numbers format.")

        logger.info("Calling add_signature function")
        result = add_signature(
            pdf_bytes=pdf_bytes,
            signature_bytes=signature_bytes,
            pages=pages,
            size=size,
            position=position,
            alignment=alignment,
            remove_bg=remove_bg
        )

        if result is None:
            logger.error("add_signature returned None")
            raise HTTPException(status_code=500, detail="Failed to add signature")

        return StreamingResponse(
            io.BytesIO(result),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=signed_{pdf_file.filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in add_signature_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        logger.info("No S3 cleanup needed (in-memory processing)")
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()

@app.post("/remove_background")
async def remove_background_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received remove background request for {file.filename}")
    
    MAX_FILE_SIZE_MB = 20  # Add this line
    if file.size / (1024 * 1024) > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty image file")
        if not USE_S3:
            local_path = os.path.join("input_pdfs", f"{hashlib.md5(file_content).hexdigest()}_{file.filename}")
            os.makedirs("input_pdfs", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)

        logger.info("Processing image for background removal")
        processed_image = remove_background_rembg(file_content)
        if not processed_image.getvalue():
            raise HTTPException(status_code=500, detail="Failed to process image")

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
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1) 


