# verify_installation.py
import importlib

required_modules = [
    'langchain',
    'langchain.chains',
    'langchain.chains.combine_documents',
    'langchain.chains.combine_documents.stuff',
    'langchain.chains.retrieval',
    'langchain_community',
    'langchain_core',
    'langchain_text_splitters',
    'langchain_chroma'
]

print("🔍 Verifying LangChain Installation...")
print("=" * 50)

all_success = True
for module in required_modules:
    try:
        importlib.import_module(module)
        print(f"✅ {module}")
    except ImportError as e:
        print(f"❌ {module} - MISSING: {e}")
        all_success = False

print("=" * 50)
if all_success:
    print("🎉 All modules installed successfully!")
else:
    print("⚠️ Some modules are missing. Run the complete installation command.")