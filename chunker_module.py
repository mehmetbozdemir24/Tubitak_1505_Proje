"""
chunker_module.py — Geriye dönük uyumluluk katmanı.

RBAC tabanlı detect_permission_from_content ve DriveDocumentProcessor
ABAC mimarisine geçildiği için kaldırıldı.

Yeni pipeline için chunker.py kullanın:
    from chunker import chunk_text, chunk_pptx
"""
from chunker import chunk_text, chunk_pptx  # noqa: F401  (re-export)

__all__ = ["chunk_text", "chunk_pptx"]
