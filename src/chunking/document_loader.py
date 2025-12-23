"""
Doküman Yükleyici Modülü
Farklı dosya tiplerinden (PDF, DOCX, PPTX, XLSX) metin çıkarmak için LangChain kullanır.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)


class DocumentLoader:
    """
    Farklı dosya formatlarından doküman yükleme sınıfı.
    Desteklenen formatlar: PDF, DOCX, PPTX, XLSX
    """
    
    def __init__(self):
        """DocumentLoader başlatıcı"""
        self.supported_formats = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.pptx': self._load_pptx,
            '.xlsx': self._load_xlsx
        }
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Dosya yolundan doküman yükler.
        
        Args:
            file_path (str): Yüklenecek dosyanın yolu
            
        Returns:
            List[Document]: Yüklenen doküman listesi
            
        Raises:
            FileNotFoundError: Dosya bulunamadığında
            ValueError: Desteklenmeyen dosya formatı için
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Desteklenmeyen dosya formatı: {file_ext}. "
                f"Desteklenen formatlar: {list(self.supported_formats.keys())}"
            )
        
        loader_func = self.supported_formats[file_ext]
        return loader_func(file_path)
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """PDF dosyası yükler"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """DOCX dosyası yükler"""
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        return documents
    
    def _load_pptx(self, file_path: str) -> List[Document]:
        """PPTX dosyası yükler"""
        loader = UnstructuredPowerPointLoader(file_path)
        documents = loader.load()
        return documents
    
    def _load_xlsx(self, file_path: str) -> List[Document]:
        """XLSX dosyası yükler"""
        loader = UnstructuredExcelLoader(file_path)
        documents = loader.load()
        return documents
    
    def get_supported_formats(self) -> List[str]:
        """Desteklenen dosya formatlarını döndürür"""
        return list(self.supported_formats.keys())
