import sys
from pathlib import Path
import pymupdf4llm


# Sec1

# Put backend/ on the path so `import rag...` works when run as a plain script.
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from rag.ingestion.loaders import load_document  # noqa: E402

pdf_path = backend_dir / "data" / "docs" / "85fb23ba2910de45e27f8f40170c0f3576043916.pdf"

# document = load_document(path=str(pdf_path))
pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=True)

print(type(pages), len(pages))          # list, n_pages
print(pages[0].keys())                  # see all fields
print(pages[52]["metadata"]["page_number"])
print(pages[52]["text"][:2000])         # eyeball the markdown for page 53

# With page_chunks=True, pymupdf4llm.to_markdown() returns a list of dicts, one per page (not a string — you only get a single markdown string when page_chunks=False). Each dict looks roughly like:
# [
#   {
#     "metadata": {          # page-level info
#       "page_number": 1,    # 1-based page number (this is the key our loader uses)
#       "file_path": "...",
#       "page_count": 132,
#       # plus title/author/format fields from the PDF metadata
#     },
#     "text": "# Heading\n\nProse...\n\n|Col A|Col B|\n|---|---|\n|Net sales|$688,415|\n",
#     "tables": [ ... ],     # table bbox/row/col info detected on the page
#     "images": [ ... ],
#     "graphics": [ ... ],
#     "words": [],
#     "toc_items": [ ... ],
#   },
#   { ... },  # page 2
# ]


# Sec2
print(f"***" * 50)
print(f"Sec2")
print(f"***" * 50)


def norm(line: str) -> str:
    return " ".join(line.strip())

print(f"split lines:{pages[52]["text"][:2000].splitlines()}")
for line in {norm(ln) for ln in pages[52]["text"][:2000].splitlines() if ln.strip()}:
    print("line:" , line)