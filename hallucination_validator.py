import re
from typing import List, Tuple
from langchain_core.documents import Document


class HallucinationValidator:
    
    MINIMUM_OVERLAP_RATIO = 0.50
    RISK_PHRASES = [
        "genel olarak", "muhtemelen", "sanırım", "sanırsam",
        "tahmin ediyorum", "olabilir ki", "genellikle", "büyük ihtimalle",
        "sanki", "gibi görünüyor", "belki de", "her halde"
    ]
    
    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> set:
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if len(w) >= min_length}
    
    @staticmethod
    def validate_response(query: str, response: str, source_docs: List[Document]) -> Tuple[bool, str]:
        
        if not source_docs:
            return False, "Kaynak doküman yok"
        
        if not response or not response.strip():
            return False, "Boş yanıt"
        
        response_keywords = HallucinationValidator.extract_keywords(response)
        if not response_keywords:
            return False, "Yanıt analiz edilemedi"
        
        context_text = " ".join([doc.page_content for doc in source_docs])
        context_keywords = HallucinationValidator.extract_keywords(context_text)
        
        if not context_keywords:
            return False, "Kaynak içerik boş"
        
        overlap = response_keywords & context_keywords
        overlap_ratio = len(overlap) / len(response_keywords)
        
        if overlap_ratio < HallucinationValidator.MINIMUM_OVERLAP_RATIO:
            return False, f"Düşük benzerlik: {overlap_ratio:.2%}"
        
        response_lower = response.lower()
        detected_risks = [
            phrase for phrase in HallucinationValidator.RISK_PHRASES 
            if phrase in response_lower
        ]
        
        if detected_risks:
            return False, f"Spekülasyon tespit edildi: {', '.join(detected_risks)}"
        
        return True, "Geçerli"
    
    @staticmethod
    def validate_context_quality(docs: List[Document], min_docs: int = 2, min_score: float = 0.60) -> Tuple[bool, str]:
        
        if not docs:
            return False, "Doküman bulunamadı"
        
        high_quality = [
            doc for doc in docs 
            if doc.metadata.get("score", 0) >= min_score
        ]
        
        if len(high_quality) < min_docs:
            return False, f"Yetersiz kaliteli doküman: {len(high_quality)}/{min_docs}"
        
        total_length = sum(len(doc.page_content) for doc in high_quality)
        if total_length < 100:
            return False, "Context çok kısa"
        
        return True, "Kaliteli"
