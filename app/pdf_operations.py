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





def encrypt_pdf(pdf_bytes, password):
    """Encrypt PDF with a password using PyMuPDF."""
    doc = None
    try:
        # Check actual process memory instead of system memory
        current_pid = os.getpid()
        process_memory = get_process_memory(current_pid)
        
        # Only check if process memory is dangerously high (>500MB for small instance)
        # Skip system-wide memory check entirely
        if process_memory and process_memory > 500:  # 500MB threshold
            logger.warning(f"Process memory high: {process_memory:.1f}MB, but proceeding anyway")
        
        # Log current state for debugging
        used_mem, avail_mem, total_mem = get_memory_info()
        logger.info(f"System memory: {used_mem:.1f}/{total_mem:.1f}MB used ({used_mem/total_mem*100:.1f}%), Process: {process_memory:.1f}MB")
        
        # OPEN PDF - this is lightweight
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        logger.info(f"PDF opened: {doc.page_count} pages")
        
        output_buffer = io.BytesIO()
        
        # Save with encryption
        doc.save(
            output_buffer,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw=password,
            user_pw=password,
            permissions=fitz.PDF_PERM_PRINT
        )
        
        output_buffer.seek(0)
        encrypted_bytes = output_buffer.getvalue()
        
        # Verify encryption worked
        test_doc = fitz.open(stream=encrypted_bytes, filetype="pdf")
        is_encrypted = test_doc.is_encrypted
        test_doc.close()
        
        if not is_encrypted:
            raise Exception("Encryption failed: Output PDF is not encrypted")
        
        logger.info(f"Encryption successful: {len(pdf_bytes)} -> {len(encrypted_bytes)} bytes")
        return encrypted_bytes
        
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        return None
    finally:
        if doc:
            doc.close()
        gc.collect()


def remove_pdf_password(pdf_bytes: bytes, password: str) -> Optional[bytes]:
    """Remove password using pikepdf which handles AES-256 better."""
    try:
        # Validate inputs
        if not pdf_bytes:
            raise ValueError("Empty PDF bytes provided")
        if not password:
            raise ValueError("Password cannot be empty")
        
        logger.info(f"Attempting to decrypt PDF of size: {len(pdf_bytes)} bytes")
        
        # Open with password - pikepdf is memory efficient
        with Pdf.open(io.BytesIO(pdf_bytes), password=password) as pdf:
            # Save without password
            output = io.BytesIO()
            pdf.save(output)
            decrypted_bytes = output.getvalue()
            
            logger.info(f"Decryption successful: {len(pdf_bytes)} -> {len(decrypted_bytes)} bytes")
            
            # Verify decryption worked
            with Pdf.open(io.BytesIO(decrypted_bytes)) as test_pdf:
                if test_pdf.is_encrypted:
                    raise RuntimeError("Output PDF is still encrypted")
            
            return decrypted_bytes
            
    except PasswordError:
        logger.error("Incorrect password provided")
        raise ValueError("Incorrect password")
    except Exception as e:
        logger.error(f"pikepdf decryption failed: {str(e)}")
        raise RuntimeError(f"Decryption failed: {str(e)}")





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