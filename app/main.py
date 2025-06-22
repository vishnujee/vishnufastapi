from fastapi import FastAPI,Request, File, UploadFile, HTTPException, Form
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
import json
import asyncio
from pinecone import Pinecone, ServerlessSpec
# from langchain_core.vectorstores.base import VectorStoreRetriever
# from pydantic import BaseModel
from botocore.exceptions import ClientError
import pathlib

import gc

from typing import Any, Dict, List, Optional
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents import Document

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
load_dotenv()


from app.pdf_operations import  (
    upload_to_s3, cleanup_s3_file,
    merge_pdfs_pypdf2, merge_pdfs_ghostscript, safe_compress_pdf, encrypt_pdf,
    convert_pdf_to_images, split_pdf, delete_pdf_pages, convert_pdf_to_word,
    convert_pdf_to_excel, convert_image_to_pdf, remove_pdf_password,reorder_pdf_pages,
    add_page_numbers, add_signature,remove_background_rembg,convert_pdf_to_ppt,convert_pdf_to_editable_ppt,
    estimate_compression_sizes
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
# app.mount("/static", StaticFiles(directory="app/static"), name="static")
# Configure static files
current_dir = pathlib.Path(__file__).parent.resolve()
static_dir = os.path.join(current_dir, "static")

app.mount("/static",StaticFiles(directory=static_dir),name="static")



# AWS S3 Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
USE_S3 = all([BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY])


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
PDF_URL = "https://vishnufastapi.s3.ap-south-1.amazonaws.com/daily_pdfs/websitepdf.pdf"
S3_PREFIX = "Amazingvideo/"
CORRECT_PASSWORD_HASH = "724e9e22f2e156b598fe7a3612a7b01a6b53a39ffc049d256fa9c2df9c49b5f1"

def download_from_url(url):
    try:
        headers = {"User-Agent": os.getenv("USER_AGENT", "VishnuAI/1.0")}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")
    





###  PINECONE CLOUD 


class PineconeRetriever(BaseRetriever):
    index: Any = Field(...)
    embeddings: Any = Field(...)
    search_type: str = Field(default="similarity")
    search_kwargs: Optional[Dict] = Field(default_factory=lambda: {"k": 5})

    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=self.search_kwargs.get("k", 5),
                include_metadata=True,
                namespace="vishnu_ai_docs"
            )
            documents = []
            for match in results["matches"]:
                text_content = match["metadata"].get("page_content", match["metadata"].get("text", ""))
                documents.append(Document(
                    page_content=text_content,
                    metadata={
                        "source": match["metadata"].get("source", ""),
                        "page": match["metadata"].get("page", 0),
                        "score": match["score"]
                    }
                ))
            return documents
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

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
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "vishnu-ai-docs"
        
        # Check if index exists
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Created new index: {index_name}")
            time.sleep(30)  # Wait for index to be ready
        
        index = pc.Index(index_name)
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Check if namespace has data
        stats = index.describe_index_stats()
        if stats.get("namespaces", {}).get("vishnu_ai_docs", {}).get("vector_count", 0) == 0:
            # Process and upsert documents
            pdf_bytes = download_from_url(PDF_URL)
            documents = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text(layout=True) or ""
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "source": PDF_URL,
                                "page": page_num + 1,
                                "content_type": "text",
                                "document_type": "education_records"
                            }
                        ))
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            vectors = []
            for i, doc in enumerate(splits):
                embedding = embeddings.embed_query(doc.page_content)
                vectors.append({
                    "id": f"doc_{i}",
                    "values": embedding,
                    "metadata": {
                        "page_content": doc.page_content,
                        "source": doc.metadata.get("source", PDF_URL),
                        "page": doc.metadata.get("page", i+1),
                        "text": doc.page_content
                    }
                })
                if len(vectors) == 100 or i == len(splits)-1:
                    index.upsert(vectors=vectors, namespace="vishnu_ai_docs")
                    vectors = []
            
            logger.info(f"Successfully upserted {len(splits)} documents")
        
        return index, embeddings
    except Exception as e:
        logger.error(f"Vectorstore initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vectorstore init failed: {str(e)}")

# Initialize LLM
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_tokens=800,
        timeout=None,
        api_key=GOOGLE_API_KEY
    )


# Initialize vectorstore and retriever
index, embeddings = initialize_vectorstore()
retriever = PineconeRetriever(
    index=index,
    embeddings=embeddings,
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "filter": {"document_type": "education_records"}
    }
)

llm = get_llm()

# System Prompt
system_prompt = (
    "You are Vishnu AI assistant providing precise and relevant answers.\n\n"
    "When responding:\n"
    "1. Prioritize the most recent data (latest date) unless the user specifies otherwise.\n"
    "2. Extract key information from the context.\n"
    "3. Never say in response- Based on the provided context/portfolio...\n"
    "4. Summarize main points concisely.\n"
    "5. Provide general information apart from provided pdf.\n\n"
    "**For tables**:\n"
    "- Always create complete & full table whenever asked to create table\n"
    "**Response style**:\n"
    "- Concise, professional, and friendly.\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])




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
            await asyncio.sleep(1)  # Non-blocking sleep
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
# @app.get("/", response_class=HTMLResponse)
# async def serve_index():
#     with open("app/static/index.html", "r") as f:
#         return HTMLResponse(content=f.read())
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())

chat_history = []

@app.post("/chat")
async def chat(query: str = Form(...), typewriter: bool = Form(False)):
    if not query.strip() or len(query) > 250:
        raise HTTPException(status_code=400, detail="Invalid query length")
    
    try:
        # Use invoke() instead of get_relevant_documents()
        raw_docs = retriever.invoke(query)
        
        if not raw_docs:
            answer = "No relevant information found in the PDF."
        else:
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({
                "input": query,
                "context": raw_docs
            })
            answer = response.get("answer", "").strip() or "No relevant information found."

        chat_history.insert(0, f"You: {query}\nAI: {answer}")
        chat_history[:] = chat_history[:10]
        full_history = "\n\n".join(chat_history)

        return {
            "answer": answer,
            "history": full_history,
            "typewriter": typewriter
        }

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat processing failed")



@app.post("/merge_pdf")
async def merge_pdfs(files: List[UploadFile] = File(...), method: str = Form("PyPDF2"), file_order: str = Form(None)):
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
    finally:
        for s3_key in s3_keys:
            cleanup_s3_file(s3_key)
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
        s3_key = upload_to_s3(file_content, file.filename)
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
        if s3_key:
            cleanup_s3_file(s3_key)
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
            "high": sizes["high"] / (1024 * 1024),  # Convert bytes to MB
            "medium": sizes["medium"] / (1024 * 1024),
            "low": sizes["low"] / (1024 * 1024),
            "custom": sizes["custom"] / (1024 * 1024)
        })
    except Exception as e:
        logger.error(f"Size estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Size estimation failed: {str(e)}")
    finally:
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
        s3_key = upload_to_s3(file_content, file.filename)
        encrypted_pdf = encrypt_pdf(file_content, password)  # This now calls the imported function
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
        if s3_key:
            cleanup_s3_file(s3_key)
        gc.collect()


@app.post("/convert_pdf_to_images")
async def convert_pdf_to_images_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received convert to images request for {file.filename}")
    if file.size / (1024 * 1024) > 50:
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")

    s3_key = None
    try:
        file_content = await file.read()
        s3_key = upload_to_s3(file_content, file.filename)
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
        if s3_key:
            cleanup_s3_file(s3_key)
        gc.collect()

@app.post("/split_pdf")
async def split_pdf_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received split request for {file.filename}")
    if file.size / (1024 * 1024) > 100:
        raise HTTPException(status_code=400, detail="File exceeds 100MB limit")

    s3_key = None
    try:
        file_content = await file.read()
        s3_key = upload_to_s3(file_content, file.filename)
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
        if s3_key:
            cleanup_s3_file(s3_key)
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
        s3_key = upload_to_s3(file_content, file.filename)
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
        if s3_key:
            cleanup_s3_file(s3_key)
        gc.collect()

@app.post("/convert_pdf_to_word")
async def convert_pdf_to_word_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received convert to Word request for {file.filename}")
    if file.size / (1024 * 1024) > 50:
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")

    s3_key = None
    try:
        file_content = await file.read()
        s3_key = upload_to_s3(file_content, file.filename)
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
        if s3_key:
            cleanup_s3_file(s3_key)
        gc.collect()

@app.post("/convert_pdf_to_excel")
async def convert_pdf_to_excel_endpoint(file: UploadFile = File(...)):
    logger.info(f"Received convert to Excel request for {file.filename}")
    if file.size / (1024 * 1024) > 50:
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")

    s3_key = None
    try:
        file_content = await file.read()
        s3_key = upload_to_s3(file_content, file.filename)
        xlsx_bytes = convert_pdf_to_excel(file_content)
        if not xlsx_bytes:
            raise HTTPException(status_code=500, detail="Conversion failed")

        return StreamingResponse(
            io.BytesIO(xlsx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="converted_output.xlsx"'}
        )
    except Exception as e:
        logger.error(f"PDF to Excel error: {e}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
    finally:
        if s3_key:
            cleanup_s3_file(s3_key)
        gc.collect()

@app.post("/convert_pdf_to_ppt")
async def convert_pdf_to_ppt_endpoint(
    file: UploadFile = File(...),
    conversion_type: str = Form("image")  # Make sure parameter name matches frontend
):
    logger.info(f"Received convert to PPT request for {file.filename} (type: {conversion_type})")
    
    # File size validation
    if file.size / (1024 * 1024) > 50:
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")

    s3_key = None
    try:
        file_content = await file.read()
        s3_key = upload_to_s3(file_content, file.filename)
        
        # Choose conversion method based on user selection
        if conversion_type == "editable":
            ppt_bytes = convert_pdf_to_editable_ppt(file_content)
            filename = "editable_output.pptx"
        else:
            ppt_bytes = convert_pdf_to_ppt(file_content)  # Your existing image-based function
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
        if s3_key:
            cleanup_s3_file(s3_key)
        gc.collect()


# @app.post("/convert_pdf_to_ppt")
# async def convert_pdf_to_ppt_endpoint(file: UploadFile = File(...)):
#     logger.info(f"Received convert to PPT request for {file.filename}")
#     if file.size / (1024 * 1024) > 50:
#         raise HTTPException(status_code=400, detail="File exceeds 50MB limit")

#     s3_key = None
#     try:
#         file_content = await file.read()
#         s3_key = upload_to_s3(file_content, file.filename)
#         ppt_bytes = convert_pdf_to_ppt(file_content)
#         if not ppt_bytes:
#             raise HTTPException(status_code=500, detail="Conversion failed")

#         return StreamingResponse(
#             io.BytesIO(ppt_bytes),
#             media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
#             headers={"Content-Disposition": 'attachment; filename="converted_output.pptx"'}
#         )
#     except Exception as e:
#         logger.error(f"PDF to PPT error: {e}")
#         raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
#     finally:
#         if s3_key:
#             cleanup_s3_file(s3_key)
#         gc.collect()


@app.post("/convert_image_to_pdf")
async def convert_image_to_pdf_endpoint(
    file: UploadFile = File(...),
    page_size: str = Form("A4"),
    orientation: str = Form("Portrait"),
    description: str = Form(""),  # New
    description_position: str = Form("bottom"),  # New
    description_font_size: int = Form(12),  # New
    custom_x: float = Form(None),  # New
    custom_y: float = Form(None)  # New
):
    logger.info(f"Received convert image to PDF request for {file.filename}")
    if file.size / (1024 * 1024) > 50:
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")
    # Validate inputs
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

    
    # Validate inputs
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

    s3_key = None
    try:
        file_content = await file.read()
        s3_key = upload_to_s3(file_content, file.filename)
        
        pdf_bytes = convert_image_to_pdf(
            file_content,
            page_size,
            orientation,
            description,
            description_position,
            description_font_size,
            custom_x,
            custom_y
        )
        
        if not pdf_bytes:
            raise HTTPException(status_code=500, detail="Conversion failed")

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="image_to_pdf.pdf"'}
        )
    except Exception as e:
        logger.error(f"Image to PDF error: {e}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
    finally:
        if s3_key:
            cleanup_s3_file(s3_key)
        gc.collect()



# @app.post("/convert_image_to_pdf")
# async def convert_image_to_pdf_endpoint(file: UploadFile = File(...), page_size: str = Form("A4"), orientation: str = Form("Portrait")):
#     logger.info(f"Received convert image to PDF request for {file.filename}")
#     if file.size / (1024 * 1024) > 50:
#         raise HTTPException(status_code=400, detail="File exceeds 50MB limit")
#     if page_size not in ["A4", "Letter"]:
#         raise HTTPException(status_code=400, detail="Invalid page size. Choose: A4, Letter")
#     if orientation not in ["Portrait", "Landscape"]:
#         raise HTTPException(status_code=400, detail="Invalid orientation. Choose: Portrait, Landscape")

#     s3_key = None
#     try:
#         file_content = await file.read()
#         s3_key = upload_to_s3(file_content, file.filename)
#         pdf_bytes = convert_image_to_pdf(file_content, page_size, orientation)
#         if not pdf_bytes:
#             raise HTTPException(status_code=500, detail="Conversion failed")

#         return StreamingResponse(
#             io.BytesIO(pdf_bytes),
#             media_type="application/pdf",
#             headers={"Content-Disposition": 'attachment; filename="image_to_pdf.pdf"'}
#         )
#     except Exception as e:
#         logger.error(f"Image to PDF error: {e}")
#         raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
#     finally:
#         if s3_key:
#             cleanup_s3_file(s3_key)
#         gc.collect()

@app.post("/remove_pdf_password")
async def remove_pdf_password_endpoint(file: UploadFile = File(...), password: str = Form(...)):
    logger.info(f"Received remove password request for {file.filename}")
    logger.info(f"Received password {file}")
    s3_key = None
    try:
        file_size_mb = file.size / (1024 * 1024)
        logger.debug(f"File size: {file_size_mb:.2f}MB")
        if file_size_mb > 50:
            logger.error(f"File {file.filename} exceeds 50MB limit")
            raise HTTPException(status_code=400, detail="File exceeds 50MB limit")

        if not password.strip():
            logger.error("Empty password provided")
            raise HTTPException(status_code=400, detail="Password cannot be empty")

        file_content = await file.read()
        if not file_content:
            logger.error(f"Empty file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if USE_S3 and s3_client:
            try:
                s3_key = upload_to_s3(file_content, file.filename)
            except Exception as e:
                logger.error(f"S3 upload failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")

        try:
            decrypted_pdf = remove_pdf_password(file_content, password)
            if decrypted_pdf is None:
                logger.error("remove_pdf_password returned None")
                raise HTTPException(status_code=500, detail="Password removal failed")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

        logger.info(f"Returning decrypted PDF for {file.filename}")
        return StreamingResponse(
            io.BytesIO(decrypted_pdf),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="decrypted_{file.filename}"'}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Password removal failed: {str(e)}")
    finally:
        if s3_key and USE_S3 and s3_client:
            try:
                cleanup_s3_file(s3_key)
            except Exception as e:
                logger.warning(f"Failed to cleanup S3 file {s3_key}: {str(e)}")
        gc.collect()
        logger.debug("Endpoint cleanup completed")

# @app.post("/remove_pdf_password")
# async def remove_pdf_password_endpoint(file: UploadFile = File(...), password: str = Form(...)):
#     logger.info(f"Received remove password request for {file.filename}")
#     if file.size / (1024 * 1024) > 50:
#         raise HTTPException(status_code=400, detail="File exceeds 50MB limit")
#     if not password:
#         raise HTTPException(status_code=400, detail="Password required")

#     s3_key = None
#     try:
#         file_content = await file.read()
#         s3_key = upload_to_s3(file_content, file.filename)
#         decrypted_pdf = remove_pdf_password(file_content, password)
#         if not decrypted_pdf:
#             raise HTTPException(status_code=500, detail="Password removal failed")

#         return StreamingResponse(
#             io.BytesIO(decrypted_pdf),
#             media_type="application/pdf",
#             headers={"Content-Disposition": 'attachment; filename="decrypted_pdf.pdf"'}
#         )
#     except Exception as e:
#         logger.error(f"Password removal error: {e}")
#         raise HTTPException(status_code=500, detail=f"Password removal failed: {str(e)}")
#     finally:
#         if s3_key:
#             cleanup_s3_file(s3_key)
#         gc.collect()



@app.post("/reorder_pages")
async def reorder_pages(file: UploadFile = File(...), page_order: str = Form(...)):
    try:
        print(f"Received page_order: {page_order}")  # Debug log
        pdf_reader = PyPDF2.PdfReader(file.file)
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
        
        # Create a generator function to stream the content
        def iterfile():
            yield output.getvalue()
            output.close()
        
        return StreamingResponse(
            iterfile(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=reordered_{file.filename}"}
        )
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

@app.post("/add_page_numbers")
async def add_page_numbers_endpoint(
    file: UploadFile = File(...),
    position: str = Form("bottom"),
    alignment: str = Form("center"),
    format: str = Form("page_x")
):
    pdf_bytes = await file.read()
    result = add_page_numbers(pdf_bytes, position, alignment, format)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to add page numbers")
    return StreamingResponse(
        io.BytesIO(result),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=numbered_{file.filename}"}
    )

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
    try:
        logger.info(f"Received request: pdf={pdf_file.filename}, signature={signature_file.filename}, specific_pages={specific_pages}, size={size}, position={position}, alignment={alignment}")
        # Read input files
        pdf_bytes = await pdf_file.read()
        signature_bytes = await signature_file.read()

        # Validate inputs
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

        # Parse page numbers
        try:
            pages = [int(p) for p in specific_pages.split(',')] if specific_pages else []
            logger.info(f"Pages to sign: {pages}")
        except ValueError:
            logger.error("Invalid page numbers format")
            raise HTTPException(status_code=400, detail="Invalid page numbers format.")

  
        # Add signature
        logger.info("Calling add_signature function")
        result = add_signature(
            pdf_bytes=pdf_bytes,
            signature_bytes=signature_bytes,
            pages=pages,
            size=size,
            position=position,
            alignment=alignment,
            remove_bg=remove_bg  # <-- this was missing
        )

        if result is None:
            logger.error("add_signature returned None")
            raise HTTPException(status_code=500, detail="Failed to add signature")

        # Return the signed PDF
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
                        
                        # Dynamically generate the direct S3 URL for each object
                        url = f"https://{BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{key}"
                        logger.info(f"Direct URL for {key}: {url}")
                        
                        videos.append({
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1) 