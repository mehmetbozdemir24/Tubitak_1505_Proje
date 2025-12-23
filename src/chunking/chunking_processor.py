"""
Chunking İşlemcisi Modülü
Yüklenen dokümanları anlamlı parçalara ayırır.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .document_loader import DocumentLoader


class ChunkingProcessor:
    """
    Dokümanları yükleyip anlamlı parçalara ayıran ana sınıf.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        ChunkingProcessor başlatıcı.
        
        Args:
            chunk_size (int): Her bir chunk'ın maksimum karakter sayısı
            chunk_overlap (int): Chunk'lar arasındaki örtüşme miktarı
            separators (List[str], optional): Metni bölmek için kullanılacak ayırıcılar
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_loader = DocumentLoader()
        
        # Varsayılan ayırıcılar: paragraf, satır, nokta, boşluk
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        Dosyayı yükler ve chunk'lara ayırır.
        
        Args:
            file_path (str): İşlenecek dosyanın yolu
            
        Returns:
            List[Document]: Chunk'lanmış doküman listesi
        """
        # Dosyayı yükle
        documents = self.document_loader.load_document(file_path)
        
        # Chunk'lara ayır
        chunks = self.text_splitter.split_documents(documents)
        
        # Metadata'ya chunk bilgilerini ekle
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
            chunk.metadata['source_file'] = file_path
        
        return chunks
    
    def process_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """
        Birden fazla dosyayı yükler ve chunk'lara ayırır.
        
        Args:
            file_paths (List[str]): İşlenecek dosya yolları listesi
            
        Returns:
            List[Document]: Tüm dosyalardan oluşturulan chunk listesi
        """
        all_chunks = []
        
        for file_path in file_paths:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
                print(f"✅ {file_path}: {len(chunks)} chunk oluşturuldu")
            except Exception as e:
                print(f"❌ {file_path}: Hata - {str(e)}")
        
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """
        Chunk'lar hakkında istatistik bilgisi döndürür.
        
        Args:
            chunks (List[Document]): Analiz edilecek chunk listesi
            
        Returns:
            dict: İstatistik bilgileri içeren sözlük
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_characters': 0
            }
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        }
