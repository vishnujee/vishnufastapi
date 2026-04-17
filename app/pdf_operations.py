import os
import logging
import boto3
import io
import psutil
import fitz
import gc
from pikepdf import Pdf, PasswordError
from typing import   Optional


from rembg import remove
os.makedirs("logs", exist_ok=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join("logs", "pdfoperations.log"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



def get_memory_info():
    """Return system-wide memory usage info in MB."""
    try:
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        used_memory = psutil.virtual_memory().used / (1024 * 1024)
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        return used_memory, available_memory, total_memory
    except Exception as e:
        logger.error(f"Failed to get memory info: {e}")
        return 0, 0, 0

def get_process_memory(pid):
    """Return memory usage of the current Python process in MB."""
    try:
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return None




# ==================== STREAMING PDF ENCRYPTION (DISK-BASED) ====================
def encrypt_pdf_streaming(input_path: str, output_path: str, password: str) -> bool:
    """
    Encrypt PDF using streaming - processes page by page, minimal RAM usage
    Only reads one page at a time from disk
    """
    try:
        from pypdf import PdfReader, PdfWriter
        
        logger.info(f"🔐 Starting streaming encryption: {input_path} -> {output_path}")
        
        # Open reader from disk (doesn't load entire file)
        reader = PdfReader(input_path)
        writer = PdfWriter()
        
        # Process each page individually (streaming)
        total_pages = len(reader.pages)
        for i, page in enumerate(reader.pages, 1):
            writer.add_page(page)
            if i % 10 == 0:  # Log progress every 10 pages
                logger.info(f"   📄 Encrypted page {i}/{total_pages}")
        
        # Add encryption
        writer.encrypt(password)
        
        # Write directly to disk (not to RAM)
        with open(output_path, "wb") as f:
            writer.write(f)
        
        logger.info(f"✅ Streaming encryption completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Streaming encryption failed: {e}")
        return False

def decrypt_pdf_streaming(input_path: str, output_path: str, password: str) -> bool:
    """
    Remove PDF password using streaming - processes page by page, minimal RAM usage
    """
    try:
        from pypdf import PdfReader, PdfWriter
        
        logger.info(f"🔓 Starting streaming decryption: {input_path} -> {output_path}")
        
        # Open reader from disk
        reader = PdfReader(input_path)
        
        # Check if encrypted and decrypt
        if reader.is_encrypted:
            try:
                reader.decrypt(password)
                logger.info(f"   🔑 Password accepted")
            except Exception as e:
                logger.error(f"   ❌ Wrong password: {e}")
                return False
        
        writer = PdfWriter()
        
        # Process each page individually
        total_pages = len(reader.pages)
        for i, page in enumerate(reader.pages, 1):
            writer.add_page(page)
            if i % 10 == 0:
                logger.info(f"   📄 Decrypted page {i}/{total_pages}")
        
        # Write directly to disk
        with open(output_path, "wb") as f:
            writer.write(f)
        
        logger.info(f"✅ Streaming decryption completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Streaming decryption failed: {e}")
        return False




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


def cleanup_s3_file(s3_key):
    """Delete file from S3 or locally."""

    try:
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        logger.info(f"Deleted S3 file: {s3_key}")
    except Exception as e:
        logger.warning(f"Failed to delete S3 file {s3_key}: {e}")