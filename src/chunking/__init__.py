"""
Chunking modülü - Çeşitli dosya tiplerinden (PDF, DOCX, PPTX, XLSX) metin çıkarma ve parçalama
"""

from .document_loader import DocumentLoader
from .chunking_processor import ChunkingProcessor

__all__ = ['DocumentLoader', 'ChunkingProcessor']
