# ==========================================
# CORRECTED PINECONE EXPORT SCRIPT - FIXED JSON SERIALIZATION
# ==========================================

from pinecone import Pinecone
from datetime import datetime

API_KEY = "pcsk_6PUdni_K929og5qJs9jKxph9VFqeH2HFVAm9jg4L8PkrTNkAdJ7VYFujuD8aBZgH253gJi"
INDEX_NAME = "vishnu-ai-docs"
NAMESPACE = "vishnu_ai_docs"

print("🔍 PINECONE EXPORT - FIXED VERSION")
print("=" * 50)

def convert_to_serializable(obj):
    """Convert Pinecone objects to JSON-serializable format"""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

try:
    # Initialize Pinecone
    pc = Pinecone(api_key=API_KEY)
    index = pc.Index(INDEX_NAME)
    
    print("✅ Connected to Pinecone")
    print(f"📊 Index: {INDEX_NAME}")
    print(f"📂 Namespace: {NAMESPACE}")
    print(f"🎯 Vector dimension: 512")
    
    # Query with correct namespace and dimension
    print("\n🔄 Querying vectors...")
    results = index.query(
        vector=[0.0] * 512,
        top_k=100,
        include_metadata=True,
        include_values=True,
        namespace=NAMESPACE
    )
    
    print(f"✅ Found {len(results['matches'])} vectors")
    
    if results['matches']:
        # Show sample data
        print(f"\n📋 SAMPLE DATA (first 3 vectors):")
        for i, match in enumerate(results['matches'][:3]):
            print(f"\n--- Vector {i+1} ---")
            print(f"ID: {match['id']}")
            print(f"Score: {match['score']:.4f}")
            print(f"Metadata keys: {list(match['metadata'].keys())}")
            
            # Show text content preview
            for key, value in match['metadata'].items():
                if isinstance(value, str):
                    print(f"  {key}: {value[:100]}{'...' if len(value) > 100 else ''}")
        
        # Save complete data (FIXED JSON SERIALIZATION)
        print(f"\n💾 SAVING DATA...")
        

        
        # 2. Extract and save text content
        texts_found = 0
        with open("pinecone_texts.txt", "w", encoding="utf-8") as f:
            f.write(f"PINECONE DATA EXPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total vectors: {len(results['matches'])}\n")
            f.write(f"Namespace: {NAMESPACE}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, match in enumerate(results['matches'], 1):
                metadata = match.get('metadata', {})
                
                # Extract text from page_content field
                text_content = metadata.get('page_content', '')
                
                if text_content:
                    texts_found += 1
                    f.write(f"=== Document {i} ===\n")
                    f.write(f"ID: {match['id']}\n")
                    f.write(f"PDF Source: {metadata.get('pdf_source', 'N/A')}\n")
                    f.write(f"Page: {metadata.get('page_num', 'N/A')}\n")
                    f.write(f"Chunk: {metadata.get('chunk_num', 'N/A')} of {metadata.get('total_chunks_page', 'N/A')}\n")
                    f.write(f"Content Type: {metadata.get('content_type', 'N/A')}\n")
                    f.write(f"Document Type: {metadata.get('document_type', 'N/A')}\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"TEXT CONTENT:\n{text_content}\n")
                    f.write("=" * 80 + "\n\n")
        
        print(f"✅ Saved {texts_found} text documents to: pinecone_texts.txt")
        

        
        # Summary
        print(f"\n📊 EXPORT SUMMARY:")
        print(f"Total vectors: {len(results['matches'])}")
        print(f"Text documents: {texts_found}")
        print(f"Namespace: {NAMESPACE}")
        
        # Show all metadata fields found
        all_metadata_fields = set()
        for match in results['matches']:
            all_metadata_fields.update(match.get('metadata', {}).keys())
        print(f"Metadata fields: {list(all_metadata_fields)}")
        
# File sizes
        import os
        print(f"\n📁 FILE SIZES:")
        for file in ["pinecone_texts.txt"]:  # Remove "pinecone_complete_data.json"
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"  {file}: {size:,} bytes")
        
    else:
        print("❌ No vectors found in the specified namespace")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()