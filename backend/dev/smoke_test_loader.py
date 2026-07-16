import sys
from pathlib import Path

# Put backend/ on the path so `import rag...` works when run as a plain script.
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from rag.ingestion.loaders import load_document  # noqa: E402

pdf_path = backend_dir / "data" / "docs" / "85fb23ba2910de45e27f8f40170c0f3576043916.pdf"

document = load_document(path=str(pdf_path))

for i, doc in enumerate(document[:10]):
    print(f'Page No {i} \n')
    print(f"Content: {doc.page_content[:100]}")
