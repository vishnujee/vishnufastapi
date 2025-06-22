# vishnu_ai_logic.py
import os
import boto3
import io
import logging
import pdfplumber
import tempfile
import hashlib
import time
import psutil
import json
import asyncio
import re
from pydantic import Field

import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import pytz
from pinecone import Pinecone, ServerlessSpec
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

# Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PERSIST_DIRECTORY = "chroma_db"

# Initialize S3 client
# s3 = boto3.client(
#     "s3",
#     aws_access_key_id=AWS_ACCESS_KEY,
#     aws_secret_access_key=AWS_SECRET_KEY,
# )

# Use the IAM role attached to the Lambda function (no need to pass keys)
s3 = boto3.client("s3")

# Constants
HEADER_KEYWORDS = [
    "name", "role", "description", "desc", "responsibility",
    "material", "activity", "qty", "quantity", "unit",
    "date", "no.", "number", "Sr. No.", "code", "UOM",
    "submitted", "Staff Name", "percentage", "indicator",
    'Issued', 'Returned', "Checklist", "Designation"
]

TITLE_KEYWORDS = [
    "RDSS VALSAD", "wise", "Safety"
]

# Document Processing
class DocumentCache:
    def __init__(self):
        self.cache = set()
    
    def get_file_hash(self, file_content):
        return hashlib.md5(file_content).hexdigest()
    
    def is_processed(self, file_hash):
        return file_hash in self.cache
    
    def mark_processed(self, file_hash):
        self.cache.add(file_hash)

class S3Monitor:
    def __init__(self, bucket_name, prefix=""):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.last_checked = datetime.now(pytz.timezone('Asia/Kolkata'))
    
    def check_for_updates(self):
        try:
            response = s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.prefix,
                StartAfter=self.last_checked.isoformat(),
            )
            all_files = response.get("Contents", [])
            new_files = [obj["Key"] for obj in all_files if not obj["Key"].endswith("/")]
            if new_files:
                self.last_checked = datetime.now(pytz.timezone('Asia/Kolkata'))
            return new_files
        except Exception as e:
            logger.error(f"S3 monitoring error: {e}")
            return []

def is_raw_table_text(text):
    line = text.strip()
    if not line:
        return False
    words = line.split()
    number_count = len(re.findall(r'\b\d+\b|\d+\.\d+', line))
    starts_with_number = bool(re.match(r'^\d+[\s\.]', line))
    has_tabular = len(words) >= 3
    has_header_keywords = any(h.lower() in line.lower() for h in HEADER_KEYWORDS)
    has_table_patterns = bool(re.search(r'\b(NO\.|KM|NOS|Mtr)\b|\d{6,}', line))
    return (
        (number_count >= 1 and has_tabular and not has_header_keywords) or
        starts_with_number or
        (has_table_patterns and has_tabular and not has_header_keywords)
    )

def extract_text_with_tables(pdf_path):
    full_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                table_content = set()
                filtered_lines = []
                cleaned_tables = []

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
                        norm_row_str = ' '.join(row_str.strip().split())
                        table_content.add(norm_row_str)
                        for cell in row:
                            if cell:
                                cell_lines = cell.split('\n')
                                for cell_line in cell_lines:
                                    if cell_line.strip():
                                        table_content.add(' '.join(cell_line.strip().split()))
                        if any(kw.lower() in row_str.lower() for kw in TITLE_KEYWORDS):
                            title_row = row_str
                            continue
                        if header_row is None and any(hk.lower() in row_str.lower() for hk in HEADER_KEYWORDS):
                            header_row = row
                            continue
                        cleaned_table.append(row)
                    if cleaned_table:
                        cleaned_tables.append((title_row, header_row, cleaned_table))

                if text:
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        norm_line = ' '.join(line.strip().split())
                        if (norm_line in table_content or
                            is_raw_table_text(line) or
                            any(kw.lower() in line.lower() for kw in TITLE_KEYWORDS) or
                            any(hk.lower() in line.lower() for hk in HEADER_KEYWORDS)):
                            continue
                        filtered_lines.append(line)

                if filtered_lines:
                    full_text.append(f"--- PAGE {i} ---\n" + "\n".join(filtered_lines))

                for title_row, header_row, cleaned_table in cleaned_tables:
                    try:
                        max_cols = max(len(row) for row in cleaned_table)
                        cleaned_table = [row + [""] * (max_cols - len(row)) for row in cleaned_table]
                        if header_row and cleaned_table:
                            df = pd.DataFrame(cleaned_table, columns=header_row)
                        else:
                            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0]) if len(cleaned_table) > 1 else pd.DataFrame(cleaned_table)
                        markdown_table = df.to_markdown(index=False)
                        if title_row:
                            full_text.append(f"\n{title_row}\n{markdown_table}\n")
                        else:
                            full_text.append(f"\nTABLE (Page {i}):\n{markdown_table}\n")
                    except Exception as e:
                        full_text.append(f"\nTABLE_RAW (Page {i}):\n{str(cleaned_table)}\n")

        return "\n".join(full_text)
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        raise

def post_process_retrieved_docs(docs: List[Document], query: str) -> List[Document]:
    processed = []
    headers_to_check = [
        "Material Description", "Stock at Store", "Received", "Issued", 
        "Sr.No.", "Sl.No.", "name", "role", "responsibility",
        "material", "activity", "qty", "Sr. No.", "code", "UOM",
        "submitted", "Staff Name", "percentage", "indicator",
        'Issued', 'Returned', "Checklist"
    ]
    query_lower = query.lower().strip()
    for doc in docs:
        content = doc.page_content
        metadata = doc.metadata.copy()
        is_table = content.count("|") > 3 and any(header.lower() in content.lower() for header in headers_to_check)
        is_inventory = "stock at store" in content.lower() or "material description" in content.lower()
        if is_table and is_inventory:
            metadata["content_type"] = "structured_table"
            lines = content.split("\n")
            simplified_content = ["| Material Description | Stock at Store |"]
            simplified_content.append("|---|---|")
            matched_row = None
            for line in lines:
                if "|" in line and line.strip() and not line.startswith("| S.No.") and not line.startswith("|---"):
                    parts = line.split("|")
                    if len(parts) >= 11:
                        material = parts[1].strip()
                        stock_value = parts[-1].strip()
                        if query_lower in material.lower():
                            matched_row = f"| {material} | {stock_value} |"
                            simplified_content = ["| Material Description | Stock at Store |", "|---|---|", matched_row]
                            break
                        simplified_content.append(f"| {material} | {stock_value} |")
            new_content = "\n".join(simplified_content) if matched_row else f"{content}\nSIMPLIFIED TABLE:\n" + "\n".join(simplified_content)
            doc.page_content = f"TABLE_START\n{new_content}\nTABLE_END"
            metadata["source_info"] = f"Page: {metadata.get('page', '?')}"
            metadata["inventory_focus"] = "Stock at Store"
        else:
            doc.page_content = content
            metadata["content_type"] = "text"
        processed.append(Document(page_content=doc.page_content, metadata=metadata))
    return processed

# Pinecone Vector Store
class PineconeRetriever(BaseRetriever):
    index: Any
    embeddings: Any
    search_type: str = "similarity"
    search_kwargs: Dict = {"k": 10}
    cache: DocumentCache = Field(default_factory=DocumentCache)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=self.search_kwargs["k"],
                include_metadata=True,
                namespace="vishnu_ai_docs",
                filter=self.search_kwargs.get("filter", {})
            )
            documents = []
            for match in results["matches"]:
                content_hash = hashlib.md5(match["metadata"]["text"].encode()).hexdigest()
                if self.cache.is_processed(content_hash):
                    continue
                documents.append(Document(
                    page_content=match["metadata"]["text"],
                    metadata={
                        "source": match["metadata"].get("source", ""),
                        "page": match["metadata"].get("page", 0),
                        "content_type": match["metadata"].get("content_type", "text"),
                        "score": match["score"]
                    }
                ))
            processed_docs = post_process_retrieved_docs(documents, query)
            return processed_docs[:10]
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

def initialize_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "vishnu-ai-docs"
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info("Created new Pinecone index")
            time.sleep(30)
        index = pc.Index(index_name)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
        stats = index.describe_index_stats()
        if stats.get("namespaces", {}).get("vishnu_ai_docs", {}).get("vector_count", 0) == 0:
            logger.info("Processing initial documents...")
            process_initial_documents(index, embeddings)
        return index, embeddings
    except Exception as e:
        logger.error(f"Pinecone initialization failed: {e}")
        raise

def process_initial_documents(index, embeddings):
    try:
        s3_contents = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="daily_pdfs/").get('Contents', [])
        pdf_files = [obj['Key'] for obj in s3_contents if obj['Key'].lower().endswith('.pdf')]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=350,
            separators=["\n\n", "\nTABLE_END\n"]
        )
        for s3_key in pdf_files:
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
                    s3.download_fileobj(BUCKET_NAME, s3_key, tmp_file)
                    tmp_file.flush()
                    combined_text = extract_text_with_tables(tmp_file.name)
                    doc = Document(
                        page_content=combined_text,
                        metadata={
                            "source": s3_key,
                            "content_type": "mixed_text_and_tables",
                            "processed_at": datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
                        }
                    )
                    splits = text_splitter.split_documents([doc])
                    vectors = []
                    for i, split in enumerate(splits):
                        embedding = embeddings.embed_query(split.page_content)
                        vectors.append({
                            "id": f"{hashlib.md5(s3_key.encode()).hexdigest()}_{i}",
                            "values": embedding,
                            "metadata": {
                                "text": split.page_content,
                                "source": s3_key,
                                "content_type": "mixed_text_and_tables",
                                "page": i+1
                            }
                        })
                    for i in range(0, len(vectors), 100):
                        batch = vectors[i:i+100]
                        index.upsert(vectors=batch, namespace="vishnu_ai_docs")
                    logger.info(f"Processed {s3_key} with {len(splits)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {s3_key}: {e}")
                continue
        logger.info("Initial document processing completed")
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise

def process_new_file(s3_key):
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file_path = tmp_file.name
            s3.download_fileobj(BUCKET_NAME, s3_key, tmp_file)
            tmp_file.flush()
            combined_text = extract_text_with_tables(tmp_file.name)
            doc = Document(
                page_content=combined_text,
                metadata={
                    "source": s3_key,
                    "content_type": "mixed_text_and_tables",
                    "processed_at": datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
                }
            )
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,
                chunk_overlap=350,
                separators=["\n\n", "\nTABLE_END\n"]
            )
            splits = text_splitter.split_documents([doc])
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1536)
            vectors = []
            for i, split in enumerate(splits):
                embedding = embeddings.embed_query(split.page_content)
                vectors.append({
                    "id": f"{hashlib.md5(s3_key.encode()).hexdigest()}_{i}",
                    "values": embedding,
                    "metadata": {
                        "text": split.page_content,
                        "source": s3_key,
                        "content_type": "mixed_text_and_tables",
                        "page": i+1
                    }
                })
            for i in range(0, len(vectors), 100):
                batch = vectors[i:i+100]
                index.upsert(vectors=batch, namespace="vishnu_ai_docs")
            logger.info(f"Processed {s3_key} with {len(splits)} chunks")
        os.unlink(tmp_file_path)
    except Exception as e:
        logger.error(f"Failed to process {s3_key}: {e}")

async def monitor_s3_updates(index):
    while True:
        try:
            new_files = S3Monitor(BUCKET_NAME, "daily_pdfs/").check_for_updates()
            if new_files:
                logger.info(f"Found {len(new_files)} new files in S3")
                with ThreadPoolExecutor(max_workers=4) as executor:
                    executor.map(process_new_file, new_files)
            await asyncio.sleep(3600)
        except Exception as e:
            logger.error(f"S3 monitoring error: {e}")
            await asyncio.sleep(600)

def initialize_services(app):
    try:
        app.state.pinecone_index, app.state.embeddings = initialize_pinecone()
        app.state.retriever = PineconeRetriever(
            index=app.state.pinecone_index,
            embeddings=app.state.embeddings,
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "filter": {
                    "$or": [
                        {"content_type": "mixed_text_and_tables"},
                        {"content_type": "structured_table"}
                    ]
                }
            }
        )
        app.state.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            max_tokens=800,
            timeout=None,
            api_key=GOOGLE_API_KEY
        )
        app.state.s3_monitor = S3Monitor(BUCKET_NAME, "daily_pdfs/")
        app.state.background_tasks = set()
        task = asyncio.create_task(monitor_s3_updates(app.state.pinecone_index))
        app.state.background_tasks.add(task)
        task.add_done_callback(app.state.background_tasks.discard)
        logger.info("Application services initialized")
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        raise