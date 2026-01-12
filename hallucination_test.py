from TEST import run_rag_pipeline


HALLUCINATION_TEST_QUERIES = [
    {
        "query": "Blockchain teknolojisi nasıl çalışır?",
        "expected": "REJECT",
        "reason": "Doküman dışı konu"
    },
    {
        "query": "Bitcoin fiyatı nedir?",
        "expected": "REJECT",
        "reason": "Alakasız finans konusu"
    },
    {
        "query": "2030 yılında yapay zeka nereye varacak?",
        "expected": "REJECT",
        "reason": "Gelecek tahmini"
    },
    {
        "query": "Mars'ta yaşam var mı?",
        "expected": "REJECT",
        "reason": "Bilimsel soru - doküman dışı"
    },
    {
        "query": "En iyi programlama dili hangisi?",
        "expected": "REJECT",
        "reason": "Subjektif görüş"
    },
    {
        "query": "Python'da liste nasıl oluşturulur?",
        "expected": "REJECT",
        "reason": "Teknik doküman dışı"
    }
]


def test_hallucination_resistance():
    print("=" * 70)
    print("🧪 HALÜSİNASYON TESTİ BAŞLIYOR")
    print("=" * 70)
    print("\nSistem bu sorulara 'Bilgim yok' demelidir, uydurmamalıdır.\n")
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(HALLUCINATION_TEST_QUERIES, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(HALLUCINATION_TEST_QUERIES)}")
        print(f"Soru: {test_case['query']}")
        print(f"Beklenen: {test_case['expected']}")
        print(f"Sebep: {test_case['reason']}")
        print(f"{'='*70}")
        
        try:
            result = run_rag_pipeline(
                test_case["query"],
                permission="admin",
                doc_type=None,
                k=5,
                SCORE_THRESHOLD=0.70
            )
            
            if result is None:
                print("✅ TEST GEÇTİ: Sistem doğru şekilde reddetti")
                passed += 1
            else:
                print("❌ TEST BAŞARISIZ: Sistem yanıt üretti (halüsinasyon riski)")
                failed += 1
                
        except Exception as e:
            print(f"⚠️ TEST HATASI: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 70)
    print("📊 TEST SONUÇLARI")
    print("=" * 70)
    print(f"✅ Geçen Testler: {passed}/{len(HALLUCINATION_TEST_QUERIES)}")
    print(f"❌ Başarısız Testler: {failed}/{len(HALLUCINATION_TEST_QUERIES)}")
    success_rate = (passed / len(HALLUCINATION_TEST_QUERIES)) * 100
    print(f"📈 Başarı Oranı: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 SİSTEM GÜVENLİ - Production için uygun")
    elif success_rate >= 70:
        print("⚠️ SİSTEM ORTA RİSKLİ - İyileştirme gerekli")
    else:
        print("🚨 SİSTEM YÜKSEK RİSKLİ - Production için UYGUN DEĞİL")
    
    print("=" * 70)
    
    return success_rate


if __name__ == "__main__":
    test_hallucination_resistance()
