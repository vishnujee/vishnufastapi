from fastapi import FastAPI,Request, File, UploadFile, HTTPException, Form,Body
from fastapi.responses import HTMLResponse, FileResponse,StreamingResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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
import io
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

import gc

from typing import Any, Dict, List, Optional
from pydantic import Field,BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

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
# from pathlib import Path
import hashlib
# from PyPDF2 import  PdfReader

from rembg import remove
from dotenv import load_dotenv
from colorama import Fore, Style, init
init() 
load_dotenv()


from app.pdf_operations import  (
    upload_to_s3, cleanup_s3_file,
    merge_pdfs_pypdf2, merge_pdfs_ghostscript, safe_compress_pdf, encrypt_pdf,
    convert_pdf_to_images, split_pdf, delete_pdf_pages, convert_pdf_to_word,
    convert_pdf_to_excel, convert_image_to_pdf, remove_pdf_password,reorder_pdf_pages,
    add_page_numbers, add_signature,remove_background_rembg,convert_pdf_to_ppt,convert_pdf_to_editable_ppt,
    estimate_compression_sizes,cleanup_local_files
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()
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
USE_S3 = all([BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY])
CORRECT_PASSWORD_HASH = os.getenv("CORRECT_PASSWORD_HASH")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
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
        logger.info(f"🔧 Processing document from {source}")
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
        
        logger.info(f"📊 Final docs for work query: {len(final_docs)} total, {len(tabular_docs)} from work experience PDF")
    
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
            
        logger.info(f"🌐 Final docs for website query: {len(final_docs)} total, {len(website_docs)} website-specific")
    
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
            logger.info(f"🔍 Starting retrieval for query: {query}")
            query_embedding = self.embeddings.embed_query(query)
            logger.info(f"✅ Query embedding generated, length: {len(query_embedding)}")

             # Add timeout configuration
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
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
            
            logger.info(f"📊 Pinecone returned {len(results['matches'])} matches")
            
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
                logger.info(f"📄 Retrieved doc - Score: {match['score']:.4f}, Source: {document.metadata['source']}")
            
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
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512,request_timeout=20)
        
        # Efficient empty check
        stats = index.describe_index_stats()
        namespace_stats = stats.get("namespaces", {}).get("vishnu_ai_docs", {})
        
        if namespace_stats.get("vector_count", 0) == 0:
            logger.info("📭 Index is empty, processing documents with enhanced extraction...")
            
            all_documents = []
            
            # Process Tabular PDF
            logger.info("📊 Processing tabular PDF...")
            pdf_bytes_tabular = download_from_url(PDF_URL_TABULAR)
            logger.info(f"📥 Downloaded tabular PDF, size: {len(pdf_bytes_tabular)} bytes")
            
            # Use enhanced PDF extraction for tabular data
            tabular_text = extract_text_with_tables(pdf_bytes_tabular)
            logger.info(f"📄 Extracted tabular text length: {len(tabular_text)} characters")
            
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
                logger.info(f"📥 Downloaded non-tabular PDF, size: {len(pdf_bytes_nontabular)} bytes")
                
                # Use non-tabular extraction for better text preservation
                nontabular_text = extract_text_from_nontabular_pdf(pdf_bytes_nontabular)
                logger.info(f"📄 Extracted non-tabular text length: {len(nontabular_text)} characters")
                
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
                        logger.info(f"📑 Created section: {section_name} ({len(section_content)} chars)")
                
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
            
            logger.info(f"✂️ Split into {len(splits)} chunks from {len(all_documents)} documents")
            
            # Log embedding process
            log_embedding_process(splits, "combined_pdfs")
            
            # Batch processing with progress
            batch_size = 50
            total_vectors = 0


            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                texts = [doc.page_content for doc in batch]
                
                logger.info(f"🔤 Embedding batch {i//batch_size + 1}/{(len(splits)-1)//batch_size + 1}")
                
                try:
                    # Batch embed for efficiency
                    embeddings_list = embeddings.embed_documents(texts)
                    
                    # ✅ ADD THESE LINES
                    logger.info(f"🔤 Generated {len(embeddings_list)} embeddings for batch")
                    if embeddings_list:
                        logger.info(f"📏 First embedding dimensions: {len(embeddings_list[0])}")
                    
                    logger.info(f"✅ Embedded {len(embeddings_list)} documents in batch")
                    
                    vectors = []
                    for j, (doc, embedding) in enumerate(zip(batch, embeddings_list)):
                        # vectors.append({
                        #     "id": f"doc_{i+j}",
                        #     "values": embedding,
                        #     "metadata": {
                        #         "page_content": doc.page_content[:800],
                        #         "source": doc.metadata.get("source", "unknown"),
                        #         "content_type": doc.metadata.get("content_type", "mixed"),
                        #         "document_type": doc.metadata.get("document_type", "unknown"),
                        #         "section": doc.metadata.get("section", "general"),
                        #         "chunk_index": i+j
                        #     }
                        # })
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
                        logger.info(f"📤 About to upsert {len(vectors)} vectors to Pinecone")
                        
                        # Upsert to Pinecone
                        upsert_response = index.upsert(vectors=vectors, namespace="vishnu_ai_docs")
                        
                        logger.info(f"📤 Successfully upserted {len(vectors)} vectors to Pinecone")
                        logger.info(f"📊 Upsert response: {upsert_response}")
                        
                        total_vectors += len(vectors)
                        logger.info(f"📤 Upserted batch {i//batch_size + 1} with {len(vectors)} vectors")
                    else:
                        logger.warning("⚠️ No vectors to upsert in this batch")
                        
                except Exception as e:
                    logger.error(f"❌ Batch {i//batch_size + 1} failed: {e}", exc_info=True)
                    continue  # Continue with next batch
            
            
            
            logger.info(f"🎉 Successfully upserted {total_vectors} documents to Pinecone from {len(all_documents)} source documents")
        
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





@app.get("/memory-usage")
async def memory_usage_stream(request: Request):
    """Stream memory usage data every second"""
    async def event_stream():
        while True:
            if await request.is_disconnected():
                break
            
            # Get memory info
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            data = {
                "ram": {
                    "total": mem.total,
                    "used": mem.used,
                    "free": mem.free,
                    "percent": mem.percent
                },
                "rom": {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "percent": swap.percent
                }
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )



def analyze_table_chunking(docs, query):
    """Analyze how table content is chunked across documents"""
    table_docs = [d for d in docs if "2.pdf" in d.metadata.get("source", "")]
    
    logger.info(f"📊 TABLE CHUNK ANALYSIS FOR QUERY: '{query}'")
    logger.info(f"📋 Found {len(table_docs)} documents from work experience PDF")
    
    for i, doc in enumerate(table_docs, 1):
        content = doc.page_content
        logger.info(f"\n🔍 TABLE CHUNK #{i}:")
        logger.info(f"📏 Content length: {len(content)} characters")
        logger.info(f"📄 Metadata: {doc.metadata}")
        
        # Check table structure
        lines = content.split('\n')
        table_lines = [line for line in lines if '|' in line]
        logger.info(f"📊 Table lines count: {len(table_lines)}")
        
        # Check for specific companies
        companies_to_check = ['KEI', 'Larsen', 'Toubro', 'Vindhya', 'Punj', 'GNG']
        found_companies = []
        for company in companies_to_check:
            if company.lower() in content.lower():
                found_companies.append(company)
        
        logger.info(f"🏢 Companies found in this chunk: {found_companies}")
        
        # Log first 500 chars of content
        preview = content[:500] + "..." if len(content) > 500 else content
        logger.info(f"👀 Content preview:\n{preview}")
    
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
        
        logger.info("🔍 COMPLETE TABLE CHUNK ANALYSIS IN VECTOR STORE")
        logger.info("=" * 80)
        
        all_companies = ['KEI', 'Larsen', 'Toubro', 'Vindhya', 'Punj', 'GNG']
        found_companies = set()
        
        for i, match in enumerate(results['matches'], 1):
            content = match['metadata'].get('page_content', '')
            logger.info(f"\n📄 CHUNK #{i}:")
            logger.info(f"📏 Length: {len(content)} chars")
            logger.info(f"📊 Score: {match['score']:.4f}")
            
            # Check companies in this chunk
            chunk_companies = [c for c in all_companies if c.lower() in content.lower()]
            logger.info(f"🏢 Companies: {chunk_companies}")
            found_companies.update(chunk_companies)
            
            # Show FULL content for table chunks
            if "TABLE" in content:
                logger.info(f"📋 FULL TABLE CONTENT:\n{content}")
            else:
                logger.info(f"📝 Content preview: {content[:200]}...")
            

            
            chunkresults = index.query(
                vector=[0] * 512,    # dummy vector
                top_k=1000,           # increase this if you have more chunks
                include_metadata=True,
                namespace="vishnu_ai_docs"
            )
            logger.info("*$" * 50)
            for i, match in enumerate(chunkresults['matches'], 1):
                print(f"\n--- CHUNK #{i} ---")
                print(f"ID: {match['id']}")
                print(f"Content:\n{match['metadata'].get('page_content', '')}\n")



            logger.info("*$" * 50)
        
        logger.info(f"\n🎯 SUMMARY:")
        logger.info(f"Total table chunks: {len(results['matches'])}")
        logger.info(f"Companies found across all chunks: {sorted(found_companies)}")
        logger.info(f"Missing companies: {set(all_companies) - found_companies}")
        
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
        logger.info(f"📋 Table docs in store: {len(all_table_results['matches'])}")
    except Exception:
        pass  # Silent fail - this is just diagnostic


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

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



# Fix the system prompt to work with RAG chain
system_prompt = (
    "You are Vishnu AI, a precise and efficient assistant. Provide accurate, relevant answers quickly.\n\n"
    "Guidelines:\n"
    "- Never mention 'based on context/text/portfolio' —just deliver the facts.\n"
    "- Summarize key points concisely. \n"
    "- Use provided context first; supplement with general knowledge only if needed.\n"
    "- Verify facts against context to avoid assumptions.\n\n"
    "**Tables**: Always generate complete, full tables when requested—include headers, data, and clear formatting.\n\n"
    "**Style**: Concise, professional, and friendly.\n"
)





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
    }
    # "gate_coach": {
    #     "label": "GATE Problem Solver 🔢🎯",
    #        "prompt": "🚀 ACTIVATE GATE PROBLEM CRUSHER MODE! 🚀\n\nHey future GATE topper! 🎓 I'm your friendly Civil Engineering buddy who makes hard problems easy to understand! No tough words, no confusing language - just simple, clear explanations! 🎉\n\n**MY EASY PROBLEM-SOLVING STEPS 📝:**\n1. **🤔 UNDERSTAND** - \"Let me see what this problem is about!\"\n2. **🔍 FIND** - \"What formulas and ideas do we need?\"\n3. **🔄 SOLVE** - \"Let's go step by step - super simple!\"\n4. **✅ CHECK** - \"Does our answer make sense?\"\n5. **🎓 EXPLAIN** - \"Here's why it works - easy peasy!\"\n\n**ALL CIVIL ENGINEERING TOPICS 🏗️:**\n- 🏛️ Building Design & Concrete (Making strong buildings!)\n- 🌋 Soil & Foundation (Working with earth and rocks!)\n- 💧 Water Flow (How liquids move and behave!)\n- 🌿 Environment Protection (Keeping our world clean!)\n- 🛣️ Roads & Transport (Building smooth travels!)\n- 📐 Math for Engineers (Numbers made easy!)\n- 📡 Land Measuring (Mapping and surveying!)\n\n**HOW I HELP YOU:**\n🎯 **NO HARD WORDS** - I speak like a friend explaining to a friend!\n💥 **STRAIGHT TO ANSWER** - No going around in circles!\n🎨 **DIFFERENT WAYS** - I show you multiple simple methods!\n📚 **BASIC IDEAS + EXAM TRICKS** - Learn the simple truth and smart shortcuts!\n\n**EXAMPLES OF WHAT YOU CAN ASK:**\n\"Solve this beam bending problem\"\n\"Find the strength of this concrete column\"\n\"Calculate how water flows through soil\"\n\"Design a simple road curve\"\n\n**MY PROMISE TO YOU:**\n- No dictionary needed! 📖❌\n- No confusing engineering jargon! 🗣️❌\n- Only simple, clear words! ✅\n- Step-by-step like a teacher! 👨‍🏫\n- High-fives when you learn! 🙌\n\nReady to make GATE problems easy? Let's start! 🔥💥\n\n*Remember: Easy learning = Better scores! 🎯*"
    # }
}

    # Add more modes as needed



@app.post("/chat")
async def chat(query: str = Form(...), mode: str = Form(None), history: str = Form(None)):
    if not query.strip() or len(query) > 10000:
        raise HTTPException(status_code=400, detail="Invalid query length")
    
    logger.info(f"🎯 CHAT QUERY: '{query}' | Mode: {mode if mode else 'Default'} | History: {'Provided' if history else 'None'}") 
    
    # ✅ ADD THIS: Parse and limit history to last 4 conversations (8 messages)
    limited_history = []
    if history:
        try:
            history_data = json.loads(history)
            # Keep only last 8 messages (4 conversations)
            limited_history = history_data[-8:]
            logger.info(f"📝 HISTORY RECEIVED: {len(history_data)} messages, LIMITED TO: {len(limited_history)} messages")
        except Exception as e:
            logger.warning(f"Failed to parse chat history: {e}")
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
                logger.info(f"🧠 MODE-SPECIFIC HISTORY: {len(limited_history)} messages")
                # Add conversation history to messages
                for msg in limited_history:
                    if msg["role"] == "user":
                        messages.append(("human", msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(("ai", msg["content"]))
            
            # Add current user message
            messages.append(("human", query))
            
            logger.info(f"📨 FINAL MESSAGES TO LLM: {len(messages)} total messages")
            
      
            
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
                logger.info(f"✅ LLM generation completed in mode '{mode}', response length: {len(answer)}")
            except asyncio.TimeoutError:
                logger.warning("⏰ LLM generation timeout in mode")
                answer = "I'm taking too long to generate a response. Please try again."
            generation_end = time.time()
            timings["generation_time"] = generation_end - generation_start
            timings["retrieval_time"] = 0.0
            timings["processing_time"] = 0.0
            raw_docs = []
            processed_docs = []
      
        else:
            # ✅ ADD THIS: Parse and limit history for RAG mode too
            limited_history = []
            if history:
                try:
                    history_data = json.loads(history)
                    # Keep only last 8 messages (4 conversations)
                    limited_history = history_data[-6:]
                    logger.info(f"📝 HISTORY RECEIVED: {len(history_data)} messages, LIMITED TO: {len(limited_history)} messages")
                except Exception as e:
                    logger.warning(f"Failed to parse chat history: {e}")
                    limited_history = []
            else:
                logger.info(f"📝 HISTORY RECEIVED: None")
            
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
                logger.warning(f"⏰ Retrieval timeout for query: {query}")
                raw_docs = []
            except Exception as e:
                logger.error(f"❌ Retrieval error: {e}")
                raw_docs = []

            retrieval_end = time.time()
            timings["retrieval_time"] = retrieval_end - retrieval_start
            
            if raw_docs:
                logger.info(f"✅ Retrieval completed in {timings['retrieval_time']:.2f}s, found {len(raw_docs)} documents")
            else:
                logger.warning(f"⚠️ Retrieval completed in {timings['retrieval_time']:.2f}s, but found 0 documents")

            # Log retrieved documents
            log_retrieved_documents(raw_docs, query)

            # ---------------- DOCUMENT PROCESSING ----------------
            processing_start = time.time()
            
            final_docs = ensure_tabular_inclusion(raw_docs, query, min_tabular=2)
            processed_docs = post_process_retrieved_docs(final_docs, query)
            
            processing_end = time.time()
            timings["processing_time"] = processing_end - processing_start
            logger.info(f"✅ Document processing completed in {timings['processing_time']:.2f}s")

            # ---------------- GENERATION ----------------
            generation_start = time.time()
            
            if not processed_docs:
                answer = "I couldn't find specific information about that in my knowledge base. Is there anything else I can help you with?"
                logger.warning("⚠️ No relevant documents found for query")
            else:
                table_context = any("2.pdf" in doc.metadata.get("source", "") for doc in processed_docs)
                if table_context:
                    logger.info("✅ TABLE DATA IS IN LLM CONTEXT!")
                else:
                    logger.info("ℹ️ No table data in context for this query")

                # ✅ PREPARE LLM CHAIN WITH LIMITED CHAT HISTORY
                messages = [
                    ("system", "You are Vishnu AI assistant — concise, friendly, and accurate. Give clear, human-like answers. Use the provided context and conversation history to answer naturally."),
                ]
                
                # ✅ USE LIMITED_HISTORY instead of full history
                if limited_history:
                    for msg in limited_history:
                        if msg["role"] == "user":
                            messages.append(("human", msg["content"]))
                        elif msg["role"] == "assistant":
                            messages.append(("ai", msg["content"]))
                
                # Add current context and question
                messages.extend([
                    ("human", "Context: {context}\n\nCurrent Question: {input}\nAnswer:")
                ])

                    
                logger.info(f"📨 FINAL MESSAGES TO LLM: {len(messages)} total messages")
                
                prompt = ChatPromptTemplate.from_messages(messages)

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
                    logger.info(f"✅ LLM generation completed, response length: {len(answer)}")
                except asyncio.TimeoutError:
                    logger.warning("⏰ LLM generation timeout")
                    answer = "I'm taking too long to generate a response. Please try again."

            generation_end = time.time()
            timings["generation_time"] = generation_end - generation_start
            logger.info(f"✅ Generation completed in {timings['generation_time']:.2f}s")

        # ---------------- RESPONSE ----------------
        # chat_entry = f"You: {query}\nAI: {answer}"
        # chat_history.insert(0, chat_entry)
        # if len(chat_history) > 3:
        #     chat_history.pop()

        total_end = time.time()
        timings["total_time"] = total_end - start_time
        logger.info(f"🎉 Total processing time: {timings['total_time']:.2f}s")

        return {
            "answer": answer,
            # "history": "\n\n".join(chat_history),
            "timings": {k: f"{v:.2f}s" for k, v in timings.items()},
            "retrieved_docs_count": len(raw_docs),
            "processed_docs_count": len(processed_docs)
        }

    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        return {
            "answer": "I'm experiencing technical issues. Please try again in a moment.",
            # "history": "\n\n".join(chat_history),
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



compression_presets = {
    "High": {"dpi": 72, "quality": 20},
    "Medium": {"dpi": 100, "quality": 30},
    "Low": {"dpi": 120, "quality": 40},
    "Custom": {"dpi": 180, "quality": 50}
}

@app.post("/compress_pdf")
async def compress_pdf(
    file: UploadFile = File(...),
    preset: str = Form(...),
    custom_dpi: int = Form(None),
    custom_quality: int = Form(None)
):
    logger.info(f"Compress request: file={file.filename}, preset={preset}, custom_dpi={custom_dpi}, custom_quality={custom_quality}")
    MAX_FILE_SIZE_MB = 160
    if file.size / (1024 * 1024) > MAX_FILE_SIZE_MB:
        logger.error(f"File {file.filename} exceeds {MAX_FILE_SIZE_MB}MB limit")
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    if preset not in compression_presets:
        logger.error(f"Invalid preset: {preset}")
        raise HTTPException(status_code=422, detail="Invalid preset. Choose: High, Medium, Low, Custom")

    dpi = compression_presets[preset]["dpi"]
    quality = compression_presets[preset]["quality"]
    if preset == "Custom":
        if custom_dpi is None or custom_quality is None:
            logger.error("Custom preset missing dpi or quality")
            raise HTTPException(status_code=400, detail="Custom preset requires custom_dpi and custom_quality")
        if not (50 <= custom_dpi <= 400 and 10 <= custom_quality <= 100):
            logger.error(f"Invalid custom_dpi={custom_dpi} or custom_quality={custom_quality}")
            raise HTTPException(status_code=400, detail="Invalid custom_dpi (50-400) or custom_quality (10-100)")
        dpi = custom_dpi
        quality = custom_quality

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
        compressed_pdf = safe_compress_pdf(file_content, dpi, quality)
        if not compressed_pdf:
            logger.error("Compression failed, no output returned")
            raise HTTPException(status_code=500, detail="Compression failed")

        logger.info("Compression successful")
        return StreamingResponse(
            io.BytesIO(compressed_pdf),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="compressed_{file.filename}"'}
        )
    except Exception as e:
        logger.error(f"Compression error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF compression failed: {str(e)}")
    finally:
        logger.info(f"S3 cleanup: {'Running' if s3_key else 'Skipping (no S3 key)'}")
        if s3_key:
            cleanup_s3_file(s3_key)
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()

@app.post("/estimate_compression_sizes")
async def estimate_sizes(
    file: UploadFile = File(...),
    custom_dpi: int = Form(180),
    custom_quality: int = Form(50)
):
    logger.info(f"Estimate sizes request: file={file.filename}, custom_dpi={custom_dpi}, custom_quality={custom_quality}")
    MAX_FILE_SIZE_MB = 160
    if file.size / (1024 * 1024) > MAX_FILE_SIZE_MB:
        logger.error(f"File {file.filename} exceeds {MAX_FILE_SIZE_MB}MB limit")
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    if not (50 <= custom_dpi <= 400 and 10 <= custom_quality <= 100):
        logger.error(f"Invalid custom_dpi={custom_dpi} or custom_quality={custom_quality}")
        raise HTTPException(status_code=400, detail="Invalid custom_dpi (50-400) or custom_quality (10-100)")

    try:
        file_content = await file.read()
        sizes = estimate_compression_sizes(file_content, custom_dpi, custom_quality)
        if not sizes:
            logger.error("Size estimation failed")
            raise HTTPException(status_code=500, detail="Size estimation failed")

        logger.info("Size estimation successful")
        return JSONResponse(content={
            "high": sizes["high"] / (1024 * 1024),
            "medium": sizes["medium"] / (1024 * 1024),
            "low": sizes["low"] / (1024 * 1024),
            "custom": sizes["custom"] / (1024 * 1024)
        })
    except Exception as e:
        logger.error(f"Size estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Size estimation failed: {str(e)}")
    finally:
        logger.info("No S3 cleanup needed (no S3 uploads)")
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()


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

@app.post("/convert_pdf_to_word")
async def convert_pdf_to_word_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received convert to Word request for {file.filename}")
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
        docx_bytes = convert_pdf_to_word(file_content)
        if not docx_bytes:
            raise HTTPException(status_code=500, detail="Conversion failed")

        return StreamingResponse(
            io.BytesIO(docx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": 'attachment; filename="converted_output.docx"'}
        )
    except Exception as e:
        logger.error(f"PDF to Word error: {e}")
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

@app.post("/convert_pdf_to_excel")
async def convert_pdf_to_excel_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received convert to Excel request for {file.filename}")
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
        xlsx_bytes = convert_pdf_to_excel(file_content)
        if not xlsx_bytes:
            raise HTTPException(status_code=500, detail="Conversion failed")
        
        return StreamingResponse(
            io.BytesIO(xlsx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="converted_output.xlsx"'}
        )
    except ValueError as ve:
        logger.error(f"Conversion failed: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")
    finally:
        logger.info(f"S3 cleanup: {'Running' if s3_key else 'Skipping (no S3 key)'}")
        if s3_key:
            cleanup_s3_file(s3_key)
        logger.info(f"Local cleanup: {'Running' if not USE_S3 else 'Skipping (USE_S3=True)'}")
        if not USE_S3:
            cleanup_local_files()
        logger.info("Running garbage collection")
        gc.collect()

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


