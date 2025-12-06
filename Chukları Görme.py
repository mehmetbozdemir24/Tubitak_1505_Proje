import pickle
import os
from collections import defaultdict


def pkl_to_organized_txts(pkl_path, base_output_folder="Organize_Veriler"):
    # 1. Pickle dosyasını yükle
    print("Pickle dosyası yükleniyor...")
    try:
        with open(pkl_path, "rb") as f:
            all_chunks = pickle.load(f)
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return

    print(f"Toplam {len(all_chunks)} chunk işleniyor...")

    # 2. Chunkları kaynak dosya ismine (source) göre grupla
    files_map = defaultdict(list)
    for chunk in all_chunks:
        source_name = chunk.metadata.get("source", "bilinmeyen.txt")
        files_map[source_name].append(chunk)

    # 3. Dosyaları türlerine göre klasörleyerek yaz
    for source_filename, chunks in files_map.items():

        # Dosya adını ve uzantısını ayrıştır
        base_name = os.path.basename(source_filename)  # örn: rapor.pdf
        name_without_ext, ext = os.path.splitext(base_name)  # örn: ("rapor", ".pdf")
        ext = ext.lower()

        # Uzantıya göre klasör belirle
        sub_folder = "DIGER"  # Bilinmeyen uzantılar için
        if ext == ".pdf":
            sub_folder = "PDF"
        elif ext in [".docx", ".doc"]:
            sub_folder = "WORD"
        elif ext in [".xlsx", ".xls", ".csv"]:
            sub_folder = "EXCEL"
        elif ext in [".pptx", ".ppt"]:
            sub_folder = "POWERPOINT"

        # Hedef klasör yolunu oluştur: "Organize_Veriler/PDF" gibi
        target_folder = os.path.join(base_output_folder, sub_folder)

        # Klasör yoksa oluştur
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Yazılacak dosya yolu: "Organize_Veriler/PDF/rapor.txt"
        output_path = os.path.join(target_folder, f"{name_without_ext}.txt")

        # Dosyayı yaz
        with open(output_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                content = chunk.page_content
                meta = chunk.metadata

                # Metadata bilgilerini listele
                meta_str_list = []
                for key, value in meta.items():
                    meta_str_list.append(f"   - {key}: {value}")
                formatted_metadata = "\n".join(meta_str_list)

                # Format bloğu
                block = (
                    f"================================================================================\n"
                    f"CHUNK NO: {i + 1}\n"
                    f"================================================================================\n"
                    f"[METADATA DETAYLARI]\n"
                    f"{formatted_metadata}\n"
                    f"--------------------------------------------------------------------------------\n"
                    f"[İÇERİK]\n"
                    f"{content}\n"
                    f"\n\n"
                )
                f.write(block)

        print(f"-> Kaydedildi: {sub_folder}/{name_without_ext}.txt ({len(chunks)} chunk)")

    print(f"\nİşlem tamamlandı! Dosyalar '{base_output_folder}' klasöründe türlerine göre ayrıldı.")


if __name__ == "__main__":
    pkl_file = "tum_dokumanlar_final_last.pkl"
    pkl_to_organized_txts(pkl_file)

