#!/usr/bin/env python3
import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime
import pytz  # Install: pip install pytz

# ✅ CORRECT LOG FILE PATH - matches crontab
LOG_FILE = '/home/ec2-user/vishnufastapi/cleanup_cron.log'

# Custom formatter for IST timestamps
class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ist = pytz.timezone('Asia/Kolkata')
        dt = datetime.fromtimestamp(record.created, ist)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

# Configure logging with IST
handler = logging.FileHandler(LOG_FILE)
handler.setFormatter(ISTFormatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ISTFormatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler, console_handler]
)

logger = logging.getLogger(__name__)

def cleanup_orphaned_files():
    """Clean up FILES older than 30 minutes from ALL directories"""
    try:
        current_time = time.time()
        orphan_age_seconds = 360  # 6 minutes
        
        # ✅ ALL directories that need cleaning
        directories_to_clean = [
            # Inside app/temp_processing
            Path('/home/ec2-user/vishnufastapi/app/temp_processing/uploads'),
            Path('/home/ec2-user/vishnufastapi/app/temp_processing/output'),
            Path('/home/ec2-user/vishnufastapi/app/temp_processing/estimation'),
            Path('/home/ec2-user/vishnufastapi/app/temp_processing/word'),
            Path('/home/ec2-user/vishnufastapi/app/temp_processing'),  # root temp dir
            
            # ✅ ALSO clean these legacy directories
            Path('/home/ec2-user/vishnufastapi/input_pdfs'),
            Path('/home/ec2-user/vishnufastapi/output_pdfs'),
        ]
        
        cleaned_count = 0
        cleaned_bytes = 0
        
        for root_dir in directories_to_clean:
            if not root_dir.exists():
                logger.info(f"Directory doesn't exist, skipping: {root_dir}")
                continue
                
            logger.info(f"📂 Scanning: {root_dir}")
            files_found = 0
            files_cleaned_here = 0
            
            # Walk through all files recursively
            for item in root_dir.rglob('*'):
                if not item.is_file():
                    continue
                    
                files_found += 1
                
                try:
                    stat = item.stat()
                    # Check all timestamps
                    most_recent = max(stat.st_mtime, stat.st_ctime, stat.st_atime)
                    
                    # Skip if file was accessed/modified recently
                    if most_recent > (current_time - orphan_age_seconds):
                        continue
                    
                    # Delete orphaned file
                    file_size = stat.st_size
                    item.unlink()
                    cleaned_count += 1
                    cleaned_bytes += file_size
                    files_cleaned_here += 1
                    logger.info(f"🧹 Deleted: {item} ({file_size/1024:.1f}KB)")
                    
                except Exception as e:
                    logger.error(f"❌ Error deleting {item}: {e}")
            
            if files_found > 0:
                logger.info(f"   📊 {root_dir.name}: {files_cleaned_here}/{files_found} files cleaned")
        
        # Summary
        if cleaned_count > 0:
            logger.info(f"🎯 TOTAL: {cleaned_count} files cleaned, {cleaned_bytes/1024/1024:.2f} MB freed")
        else:
            logger.info("ℹ️ No orphaned files found - system is clean!")
            
        return cleaned_count, cleaned_bytes
            
    except Exception as e:
        logger.error(f"💥 Critical error in cleanup: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0, 0

def check_disk_usage():
    """Log disk usage for monitoring"""
    try:
        temp_dir = Path('/home/ec2-user/vishnufastapi/app/temp_processing')
        if temp_dir.exists():
            total_size = 0
            for item in temp_dir.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
            
            logger.info(f"💾 Disk usage: {total_size/1024/1024:.2f} MB in temp_processing")
        
        # Also check input_pdfs and output_pdfs
        for dir_name in ['input_pdfs', 'output_pdfs']:
            dir_path = Path(f'/home/ec2-user/vishnufastapi/{dir_name}')
            if dir_path.exists():
                size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                if size > 0:
                    logger.info(f"💾 {dir_name}: {size/1024/1024:.2f} MB")
                    
    except Exception as e:
        logger.error(f"Error checking disk usage: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting scheduled cleanup...")
    start_time = time.time()
    
    # Log current disk usage before cleanup
    logger.info("📊 BEFORE CLEANUP:")
    check_disk_usage()
    
    files_cleaned, bytes_cleaned = cleanup_orphaned_files()
    
    # Log disk usage after cleanup
    logger.info("📊 AFTER CLEANUP:")
    check_disk_usage()
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"✅ Cleanup completed in {duration:.2f}s | Files: {files_cleaned} | Space freed: {bytes_cleaned/1024/1024:.2f}MB")
    logger.info("="*60)