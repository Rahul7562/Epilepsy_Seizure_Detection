import sys
try:
    import pypdf
except ImportError:
    try:
        import PyPDF2 as pypdf
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
        import pypdf

pdf_path = sys.argv[1]
with open(pdf_path, 'rb') as f:
    reader = pypdf.PdfReader(f)
    text = ""
    for i in range(min(5, len(reader.pages))): # read first 5 pages
        text += reader.pages[i].extract_text() + "\n"

with open("pdf_extracted.txt", "w", encoding="utf-8") as out:
    out.write(text)
print("Extracted to pdf_extracted.txt")
