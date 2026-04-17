# test_security_violations.py
import os
import fitz
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_valid_pdf_structure(file_path: str) -> bool:
    """
    Check if file is a valid PDF with proper structure
    Returns True only if:
    - File can be opened by PyMuPDF
    - Has at least 1 page
    - No garbage after %%EOF
    """
    try:
        # Check for garbage after %%EOF
        with open(file_path, 'rb') as f:
            content = f.read()
            last_eof = content.rfind(b'%%EOF')
            if last_eof != -1:
                after_eof = content[last_eof + 5:].strip()
                if after_eof:
                    logger.warning(f"⚠️ PDF has {len(after_eof)} bytes after %%EOF - rejecting")
                    return False
        
        # Check PDF structure
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()
        
        return page_count > 0
        
    except Exception as e:
        logger.debug(f"PDF structure validation failed: {e}")
        return False

def create_test_pdfs():
    """Create test PDFs to verify garbage detection"""
    
    print("\n" + "="*60)
    print("CREATING TEST PDF FILES")
    print("="*60)
    
    # 1. Clean PDF (no garbage)
    c = canvas.Canvas("clean.pdf", pagesize=letter)
    c.drawString(100, 750, "Clean PDF - no garbage")
    c.save()
    print("✅ Created: clean.pdf")
    
    # 2. PDF with garbage after EOF
    c = canvas.Canvas("with_garbage.pdf", pagesize=letter)
    c.drawString(100, 750, "PDF with garbage after EOF")
    c.save()
    
    # Add garbage after EOF
    with open("with_garbage.pdf", "ab") as f:
        f.write(b"\n<script>alert('malicious')</script>\n")
        f.write(b"GARBAGE DATA THAT SHOULD BE DETECTED\n")
        f.write(b"<!-- hidden content -->\n")
    print("✅ Created: with_garbage.pdf")
    
    # 3. PDF with multiple %%EOF (malformed)
    c = canvas.Canvas("multiple_eof.pdf", pagesize=letter)
    c.drawString(100, 750, "PDF with multiple %%EOF markers")
    c.save()
    
    with open("multiple_eof.pdf", "ab") as f:
        f.write(b"\n%%EOF\n")  # Second EOF
        f.write(b"More content after second EOF\n")
    print("✅ Created: multiple_eof.pdf")
    
    # 4. Valid PDF with multiple pages
    c = canvas.Canvas("multi_page_valid.pdf", pagesize=letter)
    for i in range(5):
        c.drawString(100, 750, f"This is page {i+1} of a valid multi-page PDF")
        c.showPage()
    c.save()
    print("✅ Created: multi_page_valid.pdf")
    
    # 5. Fake PDF (text file with header)
    with open("fake_pdf.pdf", "w") as f:
        f.write("%PDF-1.4\n")
        f.write("This is just text, not a real PDF\n")
        f.write("No PDF objects here\n")
    print("✅ Created: fake_pdf.pdf")
    
    print("\n" + "="*60)

def test_validation():
    """Test each PDF with your validation"""
    
    test_files = [
        ("clean.pdf", True, "Should PASS - clean PDF"),
        ("with_garbage.pdf", False, "Should FAIL - has garbage after EOF"),
        ("multiple_eof.pdf", False, "Should FAIL - multiple EOF markers"),
        ("multi_page_valid.pdf", True, "Should PASS - multi-page valid PDF"),
        ("fake_pdf.pdf", False, "Should FAIL - fake PDF (text only)"),
    ]
    
    print("\n" + "="*60)
    print("TESTING GARBAGE DETECTION & PDF VALIDATION")
    print("="*60)
    
    results = []
    
    for filename, should_be_valid, description in test_files:
        if not os.path.exists(filename):
            print(f"\n❌ {filename} not found - skipping")
            continue
        
        print(f"\n{'─'*60}")
        print(f"📄 Testing: {filename}")
        print(f"📝 {description}")
        print(f"🎯 Expected: {'VALID' if should_be_valid else 'INVALID'}")
        
        # Test local validation
        try:
            is_valid = is_valid_pdf_structure(filename)
            
            print(f"📊 Result: {'✅ VALID' if is_valid else '❌ INVALID'}")
            
            if is_valid == should_be_valid:
                print(f"✅ TEST PASSED")
                results.append(True)
            else:
                print(f"❌ TEST FAILED")
                if not should_be_valid and is_valid:
                    print(f"   ⚠️ SECURITY ISSUE: PDF with garbage was marked as valid!")
                    
                    # Show the garbage
                    with open(filename, 'rb') as f:
                        content = f.read()
                        last_eof = content.rfind(b'%%EOF')
                        if last_eof != -1:
                            after_eof = content[last_eof + 5:].strip()
                            if after_eof:
                                print(f"   📝 Found garbage: {after_eof[:100]}")
                results.append(False)
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append(False)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (filename, _, description) in enumerate(test_files[:len(results)]):
        status = "✅" if results[i] else "❌"
        print(f"{status} {filename}: {description}")
    
    print(f"\n{'─'*60}")
    print(f"📊 RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 PERFECT! Your PDF validation is working correctly!")
    else:
        print("⚠️ Some tests failed. Review the output above.")
    
    print("="*60 + "\n")

def test_with_api():
    """Test PDF validation through your API (respects rate limits)"""
    import requests
    import time
    
    print("\n" + "="*60)
    print("TESTING API VALIDATION (with rate limit handling)")
    print("="*60)
    
    test_files = [
        ("clean.pdf", True, "Clean PDF - should be accepted"),
        ("with_garbage.pdf", False, "PDF with garbage - should be rejected"),
        ("fake_pdf.pdf", False, "Fake PDF - should be rejected"),
    ]
    
    results = []
    
    for i, (filename, should_pass, description) in enumerate(test_files):
        if not os.path.exists(filename):
            print(f"\n❌ {filename} not found")
            continue
        
        # Wait between requests to avoid rate limiting
        if i > 0:
            print(f"\n⏳ Waiting 20 seconds to respect rate limits...")
            time.sleep(20)
        
        print(f"\n{'─'*60}")
        print(f"📤 Uploading: {filename}")
        print(f"📝 {description}")
        print(f"🎯 Expected: {'200 OK' if should_pass else '400/403 Error'}")
        
        with open(filename, 'rb') as f:
            files = {'file': (filename, f, 'application/pdf')}
            try:
                response = requests.post(
                    'http://localhost:8080/encrypt_pdf',
                    files=files,
                    data={'password': 'test123'},
                    timeout=30
                )
                
                status_code = response.status_code
                passed = (status_code == 200) == should_pass
                
                print(f"📊 Status Code: {status_code}")
                print(f"✅ Test {'PASSED' if passed else 'FAILED'}")
                
                if status_code != 200:
                    try:
                        error = response.json()
                        print(f"📄 Error: {error.get('detail', error)}")
                    except:
                        print(f"📄 Response: {response.text[:100]}")
                
                results.append(passed)
                
            except Exception as e:
                print(f"❌ Request failed: {e}")
                results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("API TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    if total > 0:
        print(f"📊 {passed}/{total} API tests passed")
        if passed == total:
            print("✅ All API tests passed!")
        else:
            print("⚠️ Some API tests failed")

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     PDF STRUCTURE VALIDATION & GARBAGE DETECTION TEST        ║
    ║                                                               ║
    ║  This will test if your PDF validation correctly detects:    ║
    ║  • Valid PDFs                                                ║
    ║  • PDFs with garbage after %%EOF                             ║
    ║  • Fake PDFs (text files with %PDF header)                   ║
    ║  • Multi-page PDFs                                           ║
    ║                                                               ║
    ║  Make sure your FastAPI server is running on port 8080       ║
    ║  for API tests (optional)                                    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Create test files
    create_test_pdfs()
    
    # Run local validation tests
    test_validation()
    
    # Ask if want to test API
    run_api_tests = input("\nTest API validation? (y/n - requires server running): ").lower()
    if run_api_tests == 'y':
        test_with_api()
    
    # Cleanup
    cleanup = input("\nDelete test files? (y/n): ").lower()
    if cleanup == 'y':
        test_files = ["clean.pdf", "with_garbage.pdf", "multiple_eof.pdf", 
                     "multi_page_valid.pdf", "fake_pdf.pdf"]
        for f in test_files:
            if os.path.exists(f):
                os.remove(f)
        print("✅ Test files deleted")