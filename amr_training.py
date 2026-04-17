"""
AMR (Abstract Meaning Representation) Model Eğitim Scripti
==========================================================

Bu script, metin → AMR grafı üreten bir seq2seq model eğitimi için
temel bir çerçeve sunar.

Kullanılan Model:
    - facebook/bart-large  (SPRING yaklaşımı - en yaygın AMR modeli)
    - Alternatif: google/mt5-base (çok dilli, Türkçe dahil)
    - Alternatif: t5-base / t5-large (İngilizce için güçlü)

Kullanılan Veri Setleri:
    - LDC2020T02 (AMR 3.0)  — Lisanslı, en kapsamlı İngilizce AMR
    - LDC2017T10 (AMR 2.0)  — Lisanslı, yaygın kullanılan önceki sürüm
    - bio_amr               — Biyomedikal alan AMR corpus (açık erişim)
    - little_prince         — Küçük Prens AMR corpus (açık erişim, 1562 cümle)
    - Türkçe için           — BOUN-TUPA projesi veya çok-dilli transfer öğrenme

Kurulum:
    pip install transformers datasets torch accelerate smatch penman
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# 1. YAPILANDIRMA
# ==========================================

@dataclass
class AMRTrainingConfig:
    """AMR model eğitimi için tüm ayarlar."""

    # --- Model ---
    # BART: Tek dilli (İngilizce) ama AMR için en güçlü model
    # mT5 : Çok dilli (Türkçe dahil), BART'tan biraz daha zayıf ama Türkçe için öneri
    model_name: str = "facebook/bart-large"
    # model_name: str = "google/mt5-base"  # Türkçe AMR için alternatif

    # --- Veri ---
    # LDC verisine erişiminiz varsa: train/dev/test dosya yollarını güncelleyin.
    # Açık erişim alternatifi olarak Little Prince veya Bio AMR kullanılabilir.
    train_file: str = "data/amr/train.jsonl"   # Her satır: {"sentence": ..., "amr": ...}
    dev_file: str = "data/amr/dev.jsonl"
    test_file: str = "data/amr/test.jsonl"

    # --- Eğitim ---
    output_dir: str = "amr_model_output"
    num_train_epochs: int = 30
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8   # Efektif batch = 4 * 8 = 32
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    fp16: bool = True                       # GPU varsa True, yoksa False
    max_source_length: int = 512
    max_target_length: int = 1024           # AMR grafı uzun olabilir
    early_stopping_patience: int = 5
    save_total_limit: int = 2


# ==========================================
# 2. VERİ SETİ
# ==========================================

class AMRDataset(Dataset):
    """
    JSONL formatındaki AMR veri setini yükler.
    Her satır: {"sentence": "Kedi uyudu.", "amr": "(z0 / sleep-01 :ARG0 (z1 / cat))"}
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.examples = self._load(file_path)

    def _load(self, file_path: str) -> List[Dict]:
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        logger.info(f"{len(examples)} örnek yüklendi: {file_path}")
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        sentence = item["sentence"]
        amr_str = item["amr"]

        # Kaynak: cümle
        source = self.tokenizer(
            sentence,
            max_length=self.max_source_length,
            padding=False,
            truncation=True,
        )

        # Hedef: AMR grafı (linearized)
        with self.tokenizer.as_target_tokenizer():
            target = self.tokenizer(
                amr_str,
                max_length=self.max_target_length,
                padding=False,
                truncation=True,
            )

        return {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "labels": target["input_ids"],
        }


# ==========================================
# 3. ÖRNEK VERİ OLUŞTURMA (DEMO)
# ==========================================

def create_sample_data(output_dir: str = "data/amr"):
    """
    Gerçek LDC verisi yoksa test için küçük bir örnek veri seti oluşturur.
    Üretim için LDC2020T02 veya bio_amr kullanın.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Küçük örnek AMR verileri (İngilizce)
    samples = [
        {
            "sentence": "The cat sat on the mat.",
            "amr": "(z0 / sit-01\n   :ARG0 (z1 / cat)\n   :location (z2 / mat))"
        },
        {
            "sentence": "The boy wants to go.",
            "amr": "(z0 / want-01\n   :ARG0 (z1 / boy)\n   :ARG1 (z2 / go-02\n      :ARG0 z1))"
        },
        {
            "sentence": "The girl likes the dog.",
            "amr": "(z0 / like-01\n   :ARG0 (z1 / girl)\n   :ARG1 (z2 / dog))"
        },
    ]

    # Train / dev / test split
    splits = {"train": samples * 10, "dev": samples * 2, "test": samples}
    for split_name, data in splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Örnek veri oluşturuldu: {out_path} ({len(data)} satır)")


# ==========================================
# 4. MODEL EĞİTİMİ
# ==========================================

def train(config: AMRTrainingConfig):
    """Modeli eğitir ve output_dir'e kaydeder."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Cihaz: {device}")

    # --- Tokenizer & Model ---
    logger.info(f"Model yükleniyor: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    # --- Veri Setleri ---
    train_dataset = AMRDataset(
        config.train_file, tokenizer,
        config.max_source_length, config.max_target_length
    )
    dev_dataset = AMRDataset(
        config.dev_file, tokenizer,
        config.max_source_length, config.max_target_length
    )

    # --- Eğitim Argümanları ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        fp16=config.fp16 and torch.cuda.is_available(),
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=config.save_total_limit,
        logging_steps=50,
        report_to="none",
    )

    # --- Data Collator ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # --- Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    logger.info("Eğitim başlıyor...")
    trainer.train()

    # --- Kaydet ---
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"Model kaydedildi: {config.output_dir}")


# ==========================================
# 5. ÇIKARSAMA (INFERENCE)
# ==========================================

def predict_amr(sentence: str, model_dir: str) -> str:
    """
    Eğitilmiş modelle bir cümleden AMR grafı üretir.

    Args:
        sentence:  Giriş cümlesi
        model_dir: Eğitilmiş modelin kaydedildiği klasör

    Returns:
        Linearize edilmiş AMR string
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=5,
            early_stopping=True,
        )

    amr_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return amr_str


# ==========================================
# 6. ANA PROGRAM
# ==========================================

if __name__ == "__main__":
    config = AMRTrainingConfig()

    # Gerçek veri yoksa örnek veri oluştur
    if not os.path.exists(config.train_file):
        logger.info("Gerçek veri bulunamadı — örnek veri oluşturuluyor...")
        create_sample_data()

    # Eğit
    train(config)

    # Test çıkarsaması
    logger.info("\n--- Örnek Çıkarsama ---")
    test_sentence = "The boy wants to go to school."
    amr_output = predict_amr(test_sentence, config.output_dir)
    logger.info(f"Girdi : {test_sentence}")
    logger.info(f"AMR   : {amr_output}")
