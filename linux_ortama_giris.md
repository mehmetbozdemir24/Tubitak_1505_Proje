# WSL'e Yani Linux'a Giriş ve Ortam Aktive

wsl -d Ubuntu-24.04 -u ogrenci

cd \~

source vllm-workspace/bin/activate

export HF\_TOKEN="hf\_LBDtNLcdnYalXufTGkQeLLSmZEnfvjkfpS"



(vllm-workspace) ogrenci@ULUNLP-5090:\~$ cd /mnt/c/Users/ogrenci/Desktop/MinerU



# VLLM server Başlat

VLLM\_USE\_FLASHINFER\_SAMPLER=0 vllm serve google/gemma-4-12B-it \\

&#x20; --max-model-len 8192 \\

&#x20; --gpu-memory-utilization 0.92 \\

&#x20; --kv-cache-dtype fp8 \\

&#x20; --reasoning-parser gemma4 \\

&#x20; --enable-auto-tool-choice \\

&#x20; --tool-call-parser gemma4 \\

&#x20; --host 0.0.0.0 --port 8000





# Ollama server Başlat

"/mnt/c/Users/ogrenci/AppData/Local/Programs/Ollama/ollama app.exe" \&



# Curl ile bir  istek at

(vllm-workspace) ogrenci@ULUNLP-5090:\~$ curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{

&#x20;   "model": "google/gemma-4-12B-it",

&#x20;   "messages": \[{"role": "user", "content": "Fatih sultlan mehmet kimdir."}],

&#x20;   "max\_tokens": 200,

&#x20;   "temperature": 0.7

&#x20; }'





# Docker conteiner Başlatma

docker run -d --name qdrant-wsl -p 6333:6333 -p 6334:6334 -v "$PWD/qdrant\_data:/qdrant/storage" qdrant/qdrant

