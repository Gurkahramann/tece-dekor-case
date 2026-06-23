# TECE DEKOR — Kenar Bandı Görsel Karşılaştırma Sistemi

## Proje Hakkında

Kenar bandı üretim numunelerini referans (master) görsellerle karşılaştıran hibrit bir kalite kontrol aracıdır. CV katmanı (SSIM + histogram korelasyonu) ile Gemini multimodal LLM katmanını birleştirerek desen yönü, budak varlığı ve renk farklarını tespit eder.

## Kurulum

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Ortam Değişkenleri

Proje kök dizininde bir `.env` dosyası oluşturun:

```
GEMINI_API_KEY=your_api_key_here
```

API anahtarını [Google AI Studio](https://aistudio.google.com/app/apikey) üzerinden alabilirsiniz.

## Çalıştırma

```bash
python main.py --img1 test_images/pair3_master.jpg --img2 test_images/saha_testi.jpg
```

## Örnek Çıktı

```json
{"similarity_score": 18, "explanation": "Renk tonu belirgin farklı, desen uyumsuz."}
```

LLM erişilemez durumdaysa skor yalnızca CV katmanından hesaplanır; hata `stderr`'e yazılır, `stdout` yine temiz JSON döndürür.
