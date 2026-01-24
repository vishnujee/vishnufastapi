#!/usr/bin/env python3
"""
One-time script to extract metadata from existing videos
Run: python extract_existing_metadata.py
"""
import boto3
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# AWS Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_PREFIX = "Amazingvideo/"
METADATA_KEY = f"{S3_PREFIX}_videos_metadata.json"

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def extract_existing_metadata():
    """Extract metadata from all existing videos"""
    print("🔍 Scanning S3 for existing videos...")
    
    metadata = {}
    video_count = 0
    processed_count = 0
    
    try:
        # Paginate through all videos (in case you have more than 1000)
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=BUCKET_NAME,
            Prefix=S3_PREFIX
        )
        
        for page in page_iterator:
            if "Contents" not in page:
                continue
                
            for obj in page["Contents"]:
                key = obj["Key"]
                
                # Check if it's a video file
                if key.lower().endswith((".mp4", ".webm", ".ogg", ".mkv", ".avi", ".mov")):
                    video_count += 1
                    
                    try:
                        # Get metadata from S3 object
                        head = s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
                        s3_metadata = head.get('Metadata', {})
                        
                        # Extract video ID (remove prefix)
                        video_id = key[len(S3_PREFIX):] if key.startswith(S3_PREFIX) else key
                        
                        # Get description from S3 metadata or use filename
                        description = s3_metadata.get('description', '')
                        
                        # If no description in S3 metadata, extract from filename
                        if not description:
                            filename = key.split('/')[-1]
                            # Remove hash prefix and extension
                            name_without_ext = filename.rsplit('.', 1)[0]
                            # Remove hash_ prefix if present
                            if '_' in name_without_ext:
                                name_without_ext = name_without_ext.split('_', 1)[1]
                            description = name_without_ext.replace('_', ' ').replace('-', ' ')
                        
                        # Store in metadata
                        metadata[video_id] = {
                            "description": description,
                            "original_filename": key.split('/')[-1],
                            "size": obj.get('Size', 0),
                            "uploaded": obj["LastModified"].isoformat(),
                            "extracted_at": datetime.now().isoformat(),
                            "has_s3_metadata": bool(s3_metadata.get('description'))
                        }
                        
                        processed_count += 1
                        print(f"✅ Extracted: {video_id}")
                        
                    except Exception as e:
                        print(f"❌ Error extracting {key}: {e}")
                        continue
        
        # Save metadata to S3
        if metadata:
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=METADATA_KEY,
                Body=json.dumps(metadata, indent=2, ensure_ascii=False),
                ContentType='application/json',
                CacheControl='no-cache'
            )
            
            print(f"\n🎯 Successfully extracted metadata for {processed_count}/{video_count} videos")
            print(f"📁 Metadata saved to: s3://{BUCKET_NAME}/{METADATA_KEY}")
            
            # Also save locally for backup
            with open('video_metadata_backup.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print("💾 Local backup saved: video_metadata_backup.json")
            
        else:
            print("⚠️ No videos found or no metadata extracted")
            
    except Exception as e:
        print(f"💥 Critical error: {e}")

if __name__ == "__main__":
    extract_existing_metadata()