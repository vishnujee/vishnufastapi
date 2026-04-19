# test_validation.py
import requests
import os
import time
import sys

def test_pdf_file(file_path, should_pass):
    """Test a single PDF file"""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"📄 Testing: {file_path}")
    print(f"📊 File size: {os.path.getsize(file_path)} bytes")
    print(f"🎯 Expected: {'✅ SHOULD PASS' if should_pass else '❌ SHOULD FAIL'}")
    print(f"{'='*60}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'application/pdf')}
            response = requests.post(
                'http://localhost:8080/encrypt_pdf',
                files=files,
                data={'password': 'test123'},
                timeout=30
            )
        
        print(f"\n📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ File was ACCEPTED")
            # Save the encrypted file
            output_name = f"encrypted_{os.path.basename(file_path)}"
            with open(output_name, 'wb') as out:
                out.write(response.content)
            print(f"💾 Saved encrypted file as: {output_name}")
            actual_result = True
        else:
            print("❌ File was REJECTED")
            try:
                error = response.json()
                print(f"📄 Reason: {error.get('detail', 'Unknown error')}")
            except:
                print(f"📄 Response: {response.text[:200]}")
            actual_result = False
        
        # Check if result matches expectation
        if actual_result == should_pass:
            print(f"\n🎉 TEST PASSED! Behavior matches expectation.")
            return True
        else:
            print(f"\n⚠️ TEST FAILED! Expected {'accept' if should_pass else 'reject'} but got {'accept' if actual_result else 'reject'}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running on port 8080")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              PDF VALIDATION API TEST SUITE                    ║
    ║                                                               ║
    ║  Testing your FastAPI server's PDF validation capabilities   ║
    ║                                                               ║
    ║  Server URL: http://localhost:8080                          ║
    ║  Endpoint: /encrypt_pdf                                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check if server is running
    print("🔍 Checking server status...")
    try:
        health = requests.get("http://localhost:8080/security-health", timeout=5)
        if health.status_code == 200:
            print("✅ Server is running and accessible\n")
        else:
            print(f"⚠️ Server returned status {health.status_code}\n")
    except:
        print("❌ Cannot connect to server on port 8080")
        print("Please start your server first:")
        print("  uvicorn main:app --reload --port 8080")
        print("\nOr if server is running on different port, update the URL in this script")
        return
    
    # Test cases
    tests = [
        ("clean.pdf", True, "Valid clean PDF - should be accepted"),
        ("with_garbage.pdf", False, "PDF with garbage after EOF - should be rejected"),
        ("multi_page_valid.pdf", True, "Multi-page valid PDF - should be accepted"),
        ("fake_pdf.pdf", False, "Fake PDF (text only) - should be rejected"),
        ("code_in_pdf.pdf", True, "PDF with code as text - should be accepted (code is just content)"),
    ]
    
    results = []
    
    for i, (filename, should_pass, description) in enumerate(tests, 1):
        print(f"\n{'─'*60}")
        print(f"📋 TEST #{i}: {description}")
        
        # Check if file exists
        if not os.path.exists(filename):
            print(f"⚠️ File '{filename}' not found in current directory")
            print(f"   Current directory: {os.getcwd()}")
            print(f"   Please run 'python create_test_files_for_api.py' first")
            results.append(False)
            continue
        
        result = test_pdf_file(filename, should_pass)
        results.append(result)
        
        # Wait between tests to avoid rate limiting
        if i < len(tests):
            print("\n⏳ Waiting 5 seconds before next test (rate limit)...")
            time.sleep(5)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len([t for t in tests if os.path.exists(t[0])])
    
    for i, (filename, should_pass, description) in enumerate(tests):
        if os.path.exists(filename):
            status = "✅" if results[i] else "❌"
            expected = "PASS" if should_pass else "FAIL"
            print(f"{status} {filename:25} Expected: {expected}")
        else:
            print(f"⚠️ {filename:25} File missing - test skipped")
    
    if total > 0:
        print(f"\n{'─'*60}")
        print(f"📈 RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n🎉 EXCELLENT! All tests passed!")
            print("✅ Your PDF validation is working perfectly in production!")
            print("\nYour system correctly:")
            print("  • Accepts valid PDFs")
            print("  • Rejects PDFs with garbage after EOF")
            print("  • Rejects fake PDFs (text files with PDF header)")
            print("  • Accepts valid PDFs even with code as text content")
        else:
            print("\n⚠️ Some tests failed. Please review the output above.")
    else:
        print("\n⚠️ No tests were executed. Please create test files first.")
    
    print("="*60)

def test_single_file():
    """Test a single file provided as command line argument"""
    if len(sys.argv) < 2:
        print("Usage: python test_validation.py <filename>")
        print("Example: python test_validation.py clean.pdf")
        return
    
    filename = sys.argv[1]
    
    # Ask if the file should pass
    print(f"\nTesting file: {filename}")
    should_pass = input("Should this file be accepted? (y/n): ").lower() == 'y'
    
    test_pdf_file(filename, should_pass)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single file mode
        test_single_file()
    else:
        # Full test suite mode
        run_all_tests()