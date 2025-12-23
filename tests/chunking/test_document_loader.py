"""
DocumentLoader sınıfı için birim testleri
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Proje kök dizinini Python path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.chunking.document_loader import DocumentLoader
from langchain_core.documents import Document


class TestDocumentLoader(unittest.TestCase):
    """DocumentLoader sınıfı için test sınıfı"""
    
    def setUp(self):
        """Her test öncesi çalışır"""
        self.loader = DocumentLoader()
    
    def test_initialization(self):
        """DocumentLoader başlatma testi"""
        self.assertIsNotNone(self.loader)
        self.assertEqual(len(self.loader.supported_formats), 4)
    
    def test_supported_formats(self):
        """Desteklenen formatları kontrol eder"""
        formats = self.loader.get_supported_formats()
        self.assertIn('.pdf', formats)
        self.assertIn('.docx', formats)
        self.assertIn('.pptx', formats)
        self.assertIn('.xlsx', formats)
    
    def test_file_not_found_error(self):
        """Var olmayan dosya için hata fırlatılması testi"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_document('/nonexistent/file.pdf')
    
    def test_unsupported_format_error(self):
        """Desteklenmeyen format için hata testi"""
        # Geçici bir dosya oluştur
        test_file = '/tmp/test.txt'
        with open(test_file, 'w') as f:
            f.write('test')
        
        try:
            with self.assertRaises(ValueError) as context:
                self.loader.load_document(test_file)
            self.assertIn('Desteklenmeyen dosya formatı', str(context.exception))
        finally:
            # Temizlik
            if os.path.exists(test_file):
                os.remove(test_file)
    
    @patch('src.chunking.document_loader.PyPDFLoader')
    def test_load_pdf(self, mock_loader):
        """PDF yükleme testi (mock ile)"""
        # Mock loader ayarla
        mock_instance = MagicMock()
        mock_instance.load.return_value = [
            Document(page_content="Test PDF content", metadata={"page": 1})
        ]
        mock_loader.return_value = mock_instance
        
        # Geçici PDF dosyası oluştur
        test_file = '/tmp/test.pdf'
        with open(test_file, 'w') as f:
            f.write('dummy pdf')
        
        try:
            documents = self.loader.load_document(test_file)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "Test PDF content")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    @patch('src.chunking.document_loader.Docx2txtLoader')
    def test_load_docx(self, mock_loader):
        """DOCX yükleme testi (mock ile)"""
        # Mock loader ayarla
        mock_instance = MagicMock()
        mock_instance.load.return_value = [
            Document(page_content="Test DOCX content", metadata={"source": "test.docx"})
        ]
        mock_loader.return_value = mock_instance
        
        # Geçici DOCX dosyası oluştur
        test_file = '/tmp/test.docx'
        with open(test_file, 'w') as f:
            f.write('dummy docx')
        
        try:
            documents = self.loader.load_document(test_file)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "Test DOCX content")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    @patch('src.chunking.document_loader.UnstructuredPowerPointLoader')
    def test_load_pptx(self, mock_loader):
        """PPTX yükleme testi (mock ile)"""
        # Mock loader ayarla
        mock_instance = MagicMock()
        mock_instance.load.return_value = [
            Document(page_content="Test PPTX content", metadata={"source": "test.pptx"})
        ]
        mock_loader.return_value = mock_instance
        
        # Geçici PPTX dosyası oluştur
        test_file = '/tmp/test.pptx'
        with open(test_file, 'w') as f:
            f.write('dummy pptx')
        
        try:
            documents = self.loader.load_document(test_file)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "Test PPTX content")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    @patch('src.chunking.document_loader.UnstructuredExcelLoader')
    def test_load_xlsx(self, mock_loader):
        """XLSX yükleme testi (mock ile)"""
        # Mock loader ayarla
        mock_instance = MagicMock()
        mock_instance.load.return_value = [
            Document(page_content="Test XLSX content", metadata={"source": "test.xlsx"})
        ]
        mock_loader.return_value = mock_instance
        
        # Geçici XLSX dosyası oluştur
        test_file = '/tmp/test.xlsx'
        with open(test_file, 'w') as f:
            f.write('dummy xlsx')
        
        try:
            documents = self.loader.load_document(test_file)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "Test XLSX content")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)


if __name__ == '__main__':
    unittest.main()
