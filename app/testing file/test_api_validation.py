# create_test_files_for_api.py
import os
import fitz
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_test_files():
    """Create fresh test files for API validation"""
    
    print("Creating fresh test files...")
    print("="*50)
    
    # 1. Clean valid PDF
    c = canvas.Canvas("clean.pdf", pagesize=letter)
    c.drawString(100, 750, "Clean PDF - no garbage, valid structure")
    c.drawString(100, 730, "This is a legitimate PDF document")
    c.save()
    print("✅ Created: clean.pdf")
    
    # 2. PDF with garbage after EOF
    c = canvas.Canvas("with_garbage.pdf", pagesize=letter)
    c.drawString(100, 750, "PDF with garbage after EOF")
    c.drawString(100, 730, "This tests security detection")
    c.save()
    # Add malicious-looking garbage after EOF
    with open("with_garbage.pdf", "ab") as f:
        f.write(b"\n<script>alert('malicious')</script>\n")
        f.write(b"<!-- hidden malware content -->\n")
        f.write(b"GARBAGE DATA THAT SHOULD BE DETECTED\n")
    print("✅ Created: with_garbage.pdf (has garbage after EOF)")
    
    # 3. Multi-page valid PDF
    c = canvas.Canvas("multi_page_valid.pdf", pagesize=letter)
    for i in range(5):
        c.drawString(100, 750, f"This is page {i+1} of a valid multi-page PDF")
        if i == 0:
            c.drawString(100, 700, "This PDF has multiple pages and is completely valid")
        c.showPage()
    c.save()
    print("✅ Created: multi_page_valid.pdf (5 pages)")
    
    # 4. Fake PDF (text file with PDF header)
    with open("fake_pdf.pdf", "w") as f:
        f.write("%PDF-1.4\n")
        f.write("This is NOT a real PDF document\n")
        f.write("It's just a text file pretending to be a PDF\n")
        f.write("No PDF objects, no structure, just text\n")
    print("✅ Created: fake_pdf.pdf (fake PDF)")
    
    # 5. Your fetch.pdf style (valid PDF with code in text)
    c = canvas.Canvas("code_in_pdf.pdf", pagesize=letter)
    c.drawString(100, 750, "Python Code Example")
    c.drawString(100, 730, "import os")
    c.drawString(100, 710, "import sys")
    c.drawString(100, 690, "def test_function():")
    c.drawString(100, 670, "    print('This is code in a PDF')")
    c.save()
    print("✅ Created: code_in_pdf.pdf (valid PDF with code as text)")
    
    print("\n" + "="*50)
    print("All test files created successfully!")
    print("\nFiles available:")
    for f in ["clean.pdf", "with_garbage.pdf", "multi_page_valid.pdf", "fake_pdf.pdf", "code_in_pdf.pdf"]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  • {f} ({size} bytes)")
    
    return True

if __name__ == "__main__":
    create_test_files()