#!/usr/bin/env python3
"""
BULLETPROOF Cleanup Script for Orphaned Files
Runs every 15 minutes via cron
"""

import sys
import os
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/ubuntu/cleanup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def cleanup_orphaned_files():
    """Clean up FILES older than 15 minutes considering ALL file activities"""
    try:
        current_time = time.time()
        orphan_age_seconds = 900  # 15 minutes
        
        # YOUR EXACT PATHS - based on your directory structure
        BASE_DIR = Path('/home/ubuntu/vishnufastapi/app/temp_processing')
        directories = [
            BASE_DIR / "uploads",
            BASE_DIR / "output", 
            BASE_DIR / "estimation",
            BASE_DIR / "word",
            BASE_DIR  # temp_processing itself
        ]
        
        cleaned_count = 0
        for root_dir in directories:
            if not root_dir.exists():
                logger.info(f"Directory doesn't exist, skipping: {root_dir}")
                continue
                
            logger.info(f"Scanning directory: {root_dir}")
            
            for item in root_dir.iterdir():
                try:
                    if item.is_dir():
                        # Clean files inside subdirectories too
                        for sub_item in item.rglob('*'):
                            if sub_item.is_file():
                                stat = sub_item.stat()
                                most_recent = max(stat.st_mtime, stat.st_ctime, stat.st_atime)
                                
                                if most_recent <= (current_time - orphan_age_seconds):
                                    sub_item.unlink()
                                    cleaned_count += 1
                                    logger.info(f"🧹 Cleaned up orphaned file: {sub_item}")
                        continue
                    
                    if item.is_file():
                        stat = item.stat()
                        most_recent = max(stat.st_mtime, stat.st_ctime, stat.st_atime)
                        
                        # Skip if file had ANY activity in last 15 minutes
                        if most_recent > (current_time - orphan_age_seconds):
                            continue
                            
                        # Delete orphaned file
                        item.unlink()
                        cleaned_count += 1
                        logger.info(f"🧹 Cleaned up orphaned file: {item}")
                        
                except Exception as e:
                    logger.error(f"❌ Error cleaning {item}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"🎯 CLEANUP SUCCESS: {cleaned_count} files removed")
        else:
            logger.info("ℹ️ No orphaned files found for cleanup")
            
        return cleaned_count
            
    except Exception as e:
        logger.error(f"💥 Critical error in orphaned file cleanup: {e}")
        return 0

if __name__ == "__main__":
    logger.info("🚀 Starting scheduled cleanup...")
    start_time = time.time()
    
    files_cleaned = cleanup_orphaned_files()
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"✅ Cleanup completed in {duration:.2f} seconds. Files removed: {files_cleaned}")