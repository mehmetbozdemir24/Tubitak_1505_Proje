# Temel Python imajı (Hafif sürüm)
FROM python:3.11-slim

# Çalışma klasörünü ayarla
WORKDIR /app

# Gereksinimleri kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kodları kopyala (embedding_script.py senin python dosyanın adı neyse o olmalı)
COPY rag_test.py .

# Konteyner açılınca çalışacak komut
# (Eğer dosyanın adı farklıysa burayı da değiştirmelisin)
CMD ["python", "rag_test.py"]