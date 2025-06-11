import os
import tempfile
import logging
import boto3
import hashlib
import io
import zipfile
import subprocess
import psutil
import fitz
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
from PIL import Image
import pdfplumber
from pdf2docx import Converter
import pandas as pd
import time
import gc
import platform

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join("logs", "pdfoperations.log"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# S3 and local mode
BUCKET_NAME = os.getenv("BUCKET_NAME", "vishnufastapi")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
LOCAL_MODE = not (AWS_ACCESS_KEY and AWS_SECRET_KEY)

if not LOCAL_MODE:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

# Local directories
INPUT_DIR = "input_pdfs"
OUTPUT_DIR = "output_pdfs"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def upload_to_s3(file_content, filename):
    """Upload file content to S3 or save locally."""
    if LOCAL_MODE:
        s3_key = f"temp_uploads/{hashlib.md5(file_content).hexdigest()}_{filename}"
        local_path = os.path.join(INPUT_DIR, os.path.basename(s3_key))
        with open(local_path, "wb") as f:
            f.write(file_content)
        logger.info(f"Saved locally: {local_path}")
        return s3_key
    else:
        s3_key = f"temp_uploads/{hashlib.md5(file_content).hexdigest()}_{filename}"
        s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=file_content)
        logger.info(f"Uploaded to S3: {s3_key}")
        return s3_key

def download_from_s3(s3_key):
    """Download file content from S3 or read locally."""
    if LOCAL_MODE:
        local_path = os.path.join(INPUT_DIR, os.path.basename(s3_key))
        if not os.path.exists(local_path):
            raise Exception(f"Local file not found: {local_path}")
        with open(local_path, "rb") as f:
            return f.read()
    else:
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
            return response["Body"].read()
        except Exception as e:
            logger.error(f"Failed to download S3 file {s3_key}: {e}")
            raise

def cleanup_s3_file(s3_key):
    """Delete file from S3 or locally."""
    if LOCAL_MODE:
        local_path = os.path.join(INPUT_DIR, os.path.basename(s3_key))
        if os.path.exists(local_path):
            os.unlink(local_path)
            logger.info(f"Deleted local file: {local_path}")
    else:
        try:
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
            logger.info(f"Deleted S3 file: {s3_key}")
        except Exception as e:
            logger.warning(f"Failed to delete S3 file {s3_key}: {e}")

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

def get_subprocess_memory(pid):
    """Get memory usage of a subprocess in MB."""
    try:
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

def merge_pdfs_pypdf2(file_contents):
    """Merge PDFs using PyPDF2 with disk-based operations."""
    merger = PdfMerger()
    temp_files = []
    try:
        # Write each PDF to temp file
        for content in file_contents:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                temp_files.append(tmp.name)
        
        # Append to merger (reads sequentially from disk)
        for tmp_file in temp_files:
            merger.append(tmp_file)
        
        # Write merged PDF to temp file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_output:
            merger.write(tmp_output.name)
            with open(tmp_output.name, "rb") as f:
                return f.read()
    except Exception as e:
        logger.error(f"PyPDF2 merge error: {e}")
        return None
    finally:
        merger.close()
        for tmp_file in temp_files:
            try:
                os.unlink(tmp_file)
            except:
                pass
        gc.collect()
def merge_pdfs_ghostscript(file_contents, output_path):
    """Merge PDFs using Ghostscript."""
    input_files = []
    try:
        total_size_mb = sum(len(content) for content in file_contents) / (1024 * 1024)
        if total_size_mb > 50:
            raise Exception(f"Total file size {total_size_mb:.2f}MB exceeds 50MB limit")
        
        for content in file_contents:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                input_files.append(tmp.name)
        
        # Determine Ghostscript binary based on OS
        gs_binary = "gswin64c" if platform.system() == "Windows" else "gs"
        
        gs_command = [
            gs_binary,
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            "-dPDFSETTINGS=/default",
            "-dNOPAUSE",
            "-dBATCH",
            f"-sOutputFile={output_path}"
        ] + input_files

        logger.info(f"Running Ghostscript command: {' '.join(gs_command)}")
        process = subprocess.run(gs_command, capture_output=True, text=True)
        if process.returncode != 0:
            logger.error(f"Ghostscript error: {process.stderr}")
            return None

        with open(output_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Ghostscript merge error: {e}")
        return None
    finally:
        for f in input_files:
            if os.path.exists(f):
                os.unlink(f)
        if os.path.exists(output_path):
            os.unlink(output_path)
        gc.collect()

def safe_compress_pdf(pdf_bytes, dpi, quality):
    """Compress PDF using Ghostscript with memory monitoring."""
    input_file = None
    output_file = None
    gs_binary = "gswin64c" if platform.system() == "Windows" else "gs"
    try:
        # Verify Ghostscript is installed
        result = subprocess.run([gs_binary, "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError(f"Ghostscript ('{gs_binary}') is not installed or not found in PATH.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_input:
            tmp_input.write(pdf_bytes)
            tmp_input.flush()
            input_file = tmp_input.name

        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

        pdf_settings = "/screen" if dpi <= 72 else "/ebook"
        gs_command = [
            gs_binary,
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            f"-dPDFSETTINGS={pdf_settings}",
            "-dColorImageDownsampleType=/Bicubic",
            f"-dColorImageResolution={dpi}",
            "-dGrayImageDownsampleType=/Bicubic",
            f"-dGrayImageResolution={dpi}",
            "-dMonoImageDownsampleType=/Subsample",
            f"-dMonoImageResolution={dpi}",
            "-dColorImageFilter=/DCTEncode",
            "-dGrayImageFilter=/DCTEncode",
            "-dNOPAUSE",
            "-dBATCH",
            "-sOutputFile=" + output_file,
            input_file
        ]

        peak_gs_memory = 0.0
        process = subprocess.Popen(
            gs_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        while process.poll() is None:
            used_mem, avail_mem, total_mem = get_memory_info()
            mem_percent = (used_mem / total_mem) * 100
            gs_memory = get_subprocess_memory(process.pid)
            if gs_memory is not None:
                peak_gs_memory = max(peak_gs_memory, gs_memory)
            if mem_percent > 80:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise Exception("High memory usage detected")
            time.sleep(0.5)

        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Ghostscript error: {stderr}")
            return None

        with open(output_file, "rb") as f:
            compressed_pdf = f.read()

        return compressed_pdf
    except FileNotFoundError as e:
        logger.error(f"Compression error: {e}")
        return None
    except Exception as e:
        logger.error(f"Compression error: {e}")
        return None
    finally:
        for path in [input_file, output_file]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
        gc.collect()

def encrypt_pdf(pdf_bytes, password):
    """Encrypt PDF with a password using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        used_mem, avail_mem, total_mem = get_memory_info()
        mem_percent = (used_mem / total_mem) * 100
        current_pid = os.getpid()
        process_memory = get_process_memory(current_pid)
        if mem_percent > 70:
            raise Exception("High memory usage detected")
        
        output_buffer = io.BytesIO()
        doc.save(
            output_buffer,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw=password,
            user_pw=password,
            permissions=fitz.PDF_PERM_PRINT
        )
        doc.close()
        output_buffer.seek(0)
        
        encrypted_bytes = output_buffer.getvalue()
        test_doc = fitz.open(stream=encrypted_bytes, filetype="pdf")
        is_encrypted = test_doc.is_encrypted
        test_doc.close()
        
        if not is_encrypted:
            raise Exception("Encryption failed: Output PDF is not encrypted")
        
        return encrypted_bytes
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        return None
    finally:
        gc.collect()

def convert_pdf_to_images(pdf_bytes):
    """Convert PDF pages to images and return a ZIP file."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(doc)
        if num_pages == 0:
            raise Exception("The PDF is empty")
        
        used_mem, avail_mem, total_mem = get_memory_info()
        mem_percent = (used_mem / total_mem) * 100
        current_pid = os.getpid()
        process_memory = get_process_memory(current_pid)
        if mem_percent > 80:
            raise Exception("High memory usage detected")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for page_num in range(num_pages):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                zip_file.writestr(f"page_{page_num + 1}.png", img_bytes)
                
                used_mem, avail_mem, total_mem = get_memory_info()
                mem_percent = (used_mem / total_mem) * 100
                if mem_percent > 80:
                    raise Exception(f"High memory usage during page {page_num + 1}")
        
        doc.close()
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    except Exception as e:
        logger.error(f"PDF to images error: {e}")
        return None
    finally:
        gc.collect()

def split_pdf(pdf_bytes):
    """Split PDF into individual pages and return a ZIP file."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for i, page in enumerate(reader.pages):
                writer = PdfWriter()
                writer.add_page(page)
                page_bytes = io.BytesIO()
                writer.write(page_bytes)
                page_bytes.seek(0)
                zipf.writestr(f"page_{i+1}.pdf", page_bytes.read())
        zip_buffer.seek(0)
        return zip_buffer.getvalue(), total_pages
    except Exception as e:
        logger.error(f"Split error: {e}")
        return None, 0
    finally:
        gc.collect()

def delete_pdf_pages(pdf_bytes, pages_to_delete):
    """Delete specified pages from PDF."""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)
        if not pages_to_delete or max(pages_to_delete) > total_pages:
            raise Exception(f"Invalid pages to delete. PDF has {total_pages} pages")
        
        pdf_writer = PdfWriter()
        for page_num in range(1, total_pages + 1):
            if page_num not in pages_to_delete:
                pdf_writer.add_page(pdf_reader.pages[page_num - 1])
        
        output_buffer = io.BytesIO()
        pdf_writer.write(output_buffer)
        output_buffer.seek(0)
        return output_buffer.getvalue()
    except Exception as e:
        logger.error(f"Delete pages error: {e}")
        return None
    finally:
        gc.collect()

def convert_pdf_to_word(pdf_bytes):
    """Convert PDF to Word using pdf2docx."""
    input_file = None
    output_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_input:
            tmp_input.write(pdf_bytes)
            input_file = tmp_input.name
        
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx").name
        cv = Converter(input_file)
        cv.convert(output_file)
        cv.close()
        
        with open(output_file, "rb") as f:
            return f.read()
    except Exception as e:
        logger.error(f"PDF to Word error: {e}")
        return None
    finally:
        for path in [input_file, output_file]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
        gc.collect()

def convert_pdf_to_excel(pdf_bytes):
    """Convert PDF to Excel using pdfplumber."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            all_tables = []
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    all_tables.append(pd.DataFrame(table[1:], columns=table[0]))
        
        if not all_tables:
            raise Exception("No tables found in PDF")
        
        df = pd.concat(all_tables, ignore_index=True) if len(all_tables) > 1 else all_tables[0]
        output_buffer = io.BytesIO()
        df.to_excel(output_buffer, index=False, engine="openpyxl")
        output_buffer.seek(0)
        return output_buffer.getvalue()
    except Exception as e:
        logger.error(f"PDF to Excel error: {e}")
        return None
    finally:
        gc.collect()

def convert_image_to_pdf(image_bytes, page_size="A4", orientation="Portrait"):
    """Convert a single image to a one-page PDF."""
    try:
        doc = fitz.open()
        page_sizes = {"A4": (595, 842), "Letter": (612, 792)}
        page_width, page_height = page_sizes.get(page_size, page_sizes["A4"])
        if orientation == "Landscape":
            page_width, page_height = page_height, page_width
        
        used_mem, avail_mem, total_mem = get_memory_info()
        mem_percent = (used_mem / total_mem) * 100
        if mem_percent > 80:
            raise Exception("High memory usage detected")
        
        img = Image.open(io.BytesIO(image_bytes))
        if img.format not in ["PNG", "JPEG"]:
            raise Exception("Only PNG and JPEG are supported")
        img_width, img_height = img.size
        
        margin = 10
        usable_width = page_width - 2 * margin
        usable_height = page_height - 2 * margin
        scale = min(usable_width / img_width, usable_height / img_height)
        new_width = img_width * scale
        new_height = img_height * scale
        
        page = doc.new_page(width=page_width, height=page_height)
        x0 = (page_width - new_width) / 2
        y0 = (page_height - new_height) / 2
        rect = fitz.Rect(x0, y0, x0 + new_width, y0 + new_height)
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        page.insert_image(rect, stream=img_buffer.getvalue())
        img.close()
        
        pdf_bytes = io.BytesIO()
        doc.save(pdf_bytes)
        doc.close()
        pdf_bytes.seek(0)
        return pdf_bytes.getvalue()
    except Exception as e:
        logger.error(f"Image to PDF error: {e}")
        return None
    finally:
        gc.collect()

def remove_pdf_password(pdf_bytes, password):
    """Remove password from PDF using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if not doc.is_encrypted:
            return pdf_bytes
        if not doc.authenticate(password):
            raise Exception("Incorrect password")
        
        used_mem, avail_mem, total_mem = get_memory_info()
        mem_percent = (used_mem / total_mem) * 100
        if mem_percent > 80:
            raise Exception("High memory usage detected")
        
        output_buffer = io.BytesIO()
        doc.save(output_buffer, encryption=0)
        doc.close()
        output_buffer.seek(0)
        
        decrypted_bytes = output_buffer.getvalue()
        test_doc = fitz.open(stream=decrypted_bytes, filetype="pdf")
        is_encrypted = test_doc.is_encrypted
        test_doc.close()
        
        if is_encrypted:
            raise Exception("Failed to remove password")
        
        return decrypted_bytes
    except Exception as e:
        logger.error(f"PDF password removal error: {e}")
        return None
    finally:
        gc.collect()




def reorder_pdf_pages(pdf_bytes, page_order):
    """Reorder PDF pages based on user-specified order."""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)
        
        # Validate page_order
        if not page_order or len(page_order) != total_pages:
            raise Exception(f"Invalid page order. Must specify {total_pages} unique pages.")
        if sorted(page_order) != list(range(1, total_pages + 1)):
            raise Exception("Page order must include all pages exactly once.")
        
        pdf_writer = PdfWriter()
        for page_num in page_order:
            pdf_writer.add_page(pdf_reader.pages[page_num - 1])
        
        output_buffer = io.BytesIO()
        pdf_writer.write(output_buffer)
        output_buffer.seek(0)
        return output_buffer.getvalue()
    except Exception as e:
        logger.error(f"Page reordering error: {e}")
        return None
    finally:
        gc.collect()

def add_page_numbers(pdf_bytes, position="bottom", alignment="center"):
    """Add page numbers to PDF at specified position and alignment."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        
        if position not in ["top", "bottom"]:
            raise Exception("Position must be 'top' or 'bottom'.")
        if alignment not in ["left", "center", "right"]:
            raise Exception("Alignment must be 'left', 'center', or 'right'.")
        
        used_mem, avail_mem, total_mem = get_memory_info()
        mem_percent = (used_mem / total_mem) * 100
        if mem_percent > 80:
            raise Exception("High memory usage detected")
        
        for page_num in range(total_pages):
            page = doc[page_num]
            page_rect = page.rect
            font_size = 12
            text = f"Page {page_num + 1}"
            
            # Calculate text position
            text_width = fitz.get_text_length(text, fontsize=font_size)
            margin = 20
            if position == "top":
                y = margin
            else:  # bottom
                y = page_rect.height - margin - font_size
            
            if alignment == "left":
                x = margin
            elif alignment == "center":
                x = (page_rect.width - text_width) / 2
            else:  # right
                x = page_rect.width - text_width - margin
            
            page.insert_text(
                (x, y),
                text,
                fontsize=font_size,
                color=(0, 0, 0),
                fontname="helv",
                overlay=True
            )
        
        output_buffer = io.BytesIO()
        doc.save(output_buffer)
        doc.close()
        output_buffer.seek(0)
        return output_buffer.getvalue()
    except Exception as e:
        logger.error(f"Page numbering error: {e}")
        return None
    finally:
        gc.collect()

def add_signature(pdf_bytes, signature_bytes, page_num, size="medium", position="bottom", alignment="center"):
    """Add a signature image to a specific PDF page."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        
        if page_num < 1 or page_num > total_pages:
            raise Exception(f"Invalid page number. PDF has {total_pages} pages.")
        if size not in ["small", "medium", "large"]:
            raise Exception("Size must be 'small', 'medium', or 'large'.")
        if position not in ["top", "center", "bottom"]:
            raise Exception("Position must be 'top', 'center', or 'bottom'.")
        if alignment not in ["left", "center", "right"]:
            raise Exception("Alignment must be 'left', 'center', or 'right'.")
        
        used_mem, avail_mem, total_mem = get_memory_info()
        mem_percent = (used_mem / total_mem) * 100
        if mem_percent > 80:
            raise Exception("High memory usage detected")
        
        # Load signature image
        img = Image.open(io.BytesIO(signature_bytes))
        if img.format not in ["PNG", "JPEG"]:
            raise Exception("Signature must be PNG or JPEG.")
        
        # Define size dimensions
        size_dims = {"small": 50, "medium": 100, "large": 150}
        max_dim = size_dims[size]
        
        # Scale image
        img_width, img_height = img.size
        scale = min(max_dim / img_width, max_dim / img_height)
        new_width = img_width * scale
        new_height = img_height * scale
        
        # Save signature to buffer
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img.close()
        
        # Insert signature
        page = doc[page_num - 1]
        page_rect = page.rect
        margin = 20
        
        if position == "top":
            y0 = margin
        elif position == "center":
            y0 = (page_rect.height - new_height) / 2
        else:  # bottom
            y0 = page_rect.height - new_height - margin
            
        if alignment == "left":
            x0 = margin
        elif alignment == "center":
            x0 = (page_rect.width - new_width) / 2
        else:  # right
            x0 = page_rect.width - new_width - margin
        
        rect = fitz.Rect(x0, y0, x0 + new_width, y0 + new_height)
        page.insert_image(rect, stream=img_buffer.getvalue())
        
        output_buffer = io.BytesIO()
        doc.save(output_buffer)
        doc.close()
        output_buffer.seek(0)
        return output_buffer.getvalue()
    except Exception as e:
        logger.error(f"Signature addition error: {e}")
        return None
    finally:
        gc.collect()

