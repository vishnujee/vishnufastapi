# test_clamav_scan.py
import subprocess
import tempfile
import os

def test_clamav():
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
        f.write(b"This is a clean test file")
        temp_path = f.name
    
    try:
        result = subprocess.run(
            ["clamscan", temp_path],  # Removed --infected flag
            capture_output=True,
            text=True,
            timeout=30
        )
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
        # Check return code (0 = clean, 1 = virus found)
        if result.returncode == 0:
            print("✅ ClamAV is working correctly! (Clean file)")
        elif result.returncode == 1:
            print("❌ Virus found!")
        else:
            print("❌ ClamAV error")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    test_clamav()