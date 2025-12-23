"""
ChunkingProcessor sınıfı için birim testleri
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Proje kök dizinini Python path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.chunking.chunking_processor import ChunkingProcessor
from langchain_core.documents import Document


class TestChunkingProcessor(unittest.TestCase):
    """ChunkingProcessor sınıfı için test sınıfı"""
    
    def setUp(self):
        """Her test öncesi çalışır"""
        self.processor = ChunkingProcessor(
            chunk_size=100,
            chunk_overlap=20
        )
    
    def test_initialization(self):
        """ChunkingProcessor başlatma testi"""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.chunk_size, 100)
        self.assertEqual(self.processor.chunk_overlap, 20)
    
    def test_initialization_with_custom_separators(self):
        """Özel ayırıcılarla başlatma testi"""
        custom_separators = ["\n", ". "]
        processor = ChunkingProcessor(
            chunk_size=100,
            chunk_overlap=20,
            separators=custom_separators
        )
        self.assertIsNotNone(processor)
    
    @patch('src.chunking.chunking_processor.DocumentLoader')
    def test_process_file(self, mock_loader_class):
        """Tek dosya işleme testi"""
        # Mock loader oluştur
        mock_loader = MagicMock()
        mock_loader.load_document.return_value = [
            Document(
                page_content="Bu bir test metnidir. " * 20,  # Uzun metin
                metadata={"source": "test.pdf"}
            )
        ]
        mock_loader_class.return_value = mock_loader
        
        # Processor'ı mock loader ile oluştur
        processor = ChunkingProcessor(chunk_size=100, chunk_overlap=20)
        processor.document_loader = mock_loader
        
        # Geçici dosya oluştur
        test_file = '/tmp/test.pdf'
        with open(test_file, 'w') as f:
            f.write('dummy')
        
        try:
            chunks = processor.process_file(test_file)
            
            # Chunk'ların oluşturulduğunu kontrol et
            self.assertGreater(len(chunks), 0)
            
            # Her chunk'ın metadata'sını kontrol et
            for chunk in chunks:
                self.assertIn('chunk_id', chunk.metadata)
                self.assertIn('chunk_size', chunk.metadata)
                self.assertIn('source_file', chunk.metadata)
                self.assertEqual(chunk.metadata['source_file'], test_file)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
    
    @patch('src.chunking.chunking_processor.DocumentLoader')
    def test_process_multiple_files(self, mock_loader_class):
        """Çoklu dosya işleme testi"""
        # Mock loader oluştur
        mock_loader = MagicMock()
        mock_loader.load_document.side_effect = [
            [Document(page_content="İlk dosya içeriği. " * 10, metadata={"source": "file1.pdf"})],
            [Document(page_content="İkinci dosya içeriği. " * 10, metadata={"source": "file2.pdf"})]
        ]
        mock_loader_class.return_value = mock_loader
        
        # Processor'ı mock loader ile oluştur
        processor = ChunkingProcessor(chunk_size=100, chunk_overlap=20)
        processor.document_loader = mock_loader
        
        # Geçici dosyalar oluştur
        test_files = ['/tmp/test1.pdf', '/tmp/test2.pdf']
        for f in test_files:
            with open(f, 'w') as file:
                file.write('dummy')
        
        try:
            all_chunks = processor.process_multiple_files(test_files)
            
            # Chunk'ların oluşturulduğunu kontrol et
            self.assertGreater(len(all_chunks), 0)
            
            # Her iki dosyanın da işlendiğini kontrol et
            self.assertEqual(mock_loader.load_document.call_count, 2)
        finally:
            for f in test_files:
                if os.path.exists(f):
                    os.remove(f)
    
    def test_get_chunk_statistics_empty(self):
        """Boş chunk listesi için istatistik testi"""
        stats = self.processor.get_chunk_statistics([])
        
        self.assertEqual(stats['total_chunks'], 0)
        self.assertEqual(stats['avg_chunk_size'], 0)
        self.assertEqual(stats['min_chunk_size'], 0)
        self.assertEqual(stats['max_chunk_size'], 0)
        self.assertEqual(stats['total_characters'], 0)
    
    def test_get_chunk_statistics(self):
        """Chunk istatistikleri testi"""
        # Test chunk'ları oluştur
        chunks = [
            Document(page_content="A" * 50, metadata={}),
            Document(page_content="B" * 100, metadata={}),
            Document(page_content="C" * 75, metadata={})
        ]
        
        stats = self.processor.get_chunk_statistics(chunks)
        
        self.assertEqual(stats['total_chunks'], 3)
        self.assertEqual(stats['avg_chunk_size'], 75.0)
        self.assertEqual(stats['min_chunk_size'], 50)
        self.assertEqual(stats['max_chunk_size'], 100)
        self.assertEqual(stats['total_characters'], 225)
    
    def test_chunk_size_respected(self):
        """Chunk boyutunun ayarlara uygun olduğunu test eder"""
        processor = ChunkingProcessor(chunk_size=50, chunk_overlap=10)
        
        # Uzun bir metin oluştur
        long_text = "Bu bir test metnidir. " * 100
        documents = [Document(page_content=long_text, metadata={"source": "test"})]
        
        # Chunk'lara ayır
        chunks = processor.text_splitter.split_documents(documents)
        
        # Her chunk'ın boyutunun chunk_size'dan küçük veya eşit olduğunu kontrol et
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), processor.chunk_size + 10)  # Küçük tolerans


if __name__ == '__main__':
    unittest.main()
