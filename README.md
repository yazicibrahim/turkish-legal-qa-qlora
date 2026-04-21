# ⚖️ Turkish Legal QA — LLaMA-2 + QLoRA + RAG

> **TÜBİTAK 2209-Araştırma Projesi destekli lisans tez projesi | Karadeniz Teknik Üniversitesi

Türk iş hukuku alanında soru-cevap yapabilen, **LLaMA-2-7B-Chat** üzerine **QLoRA** ile fine-tune edilmiş ve **FAISS tabanlı RAG pipeline** ile güçlendirilmiş bir yapay zeka sistemi.

🤗 **Model:** [huggingface.co/ibrhmyzc/lawqamodel](https://huggingface.co/ibrhmyzc/lawqamodel)

---

## 📌 Proje Bilgisi

| | |
|---|---|
| **Tez Başlığı** | Yapay Zeka Destekli Hukuk Chatbot Geliştirilmesi ve Hukuki Sorunların Çözümünde Etkinliği |
| **Destek** | TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri |
| **Kurum** | Karadeniz Teknik Üniversitesi — Yönetim Bilişim Sistemleri |
| **Ekip** | İbrahim Yazıcı, İsmail Cem Özkan |

---

## 🏗️ Sistem Mimarisi

```
[PDF Belgeler + Web Kaynakları]
        ↓
[OCR (pytesseract) / Web Scraping (Selenium)]
        ↓
[Embedding — sentence-transformers/all-MiniLM-L6-v2]
        ↓
[FAISS Vektör İndeksi]
        ↓ top-k=2 retrieval
[LLaMA-2-7B-Chat + QLoRA Adapter]
        ↓
[Yanıt + Yargıtay Kaynağı]
```

---

## 🔧 Teknoloji Yığını

| Kategori | Teknolojiler |
|----------|-------------|
| **LLM & Fine-tuning** | `transformers`, `peft`, `trl`, `bitsandbytes` |
| **Vektör Arama** | `faiss-cpu`, `sentence-transformers` |
| **Veri Toplama** | `selenium`, `pytesseract`, `pdf2image` |
| **Altyapı** | `PyTorch`, `datasets`, `accelerate` |

---

## 📦 Veri Pipeline'ı

### 1. PDF → Metin (OCR)
Türkçe iş hukuku PDF belgeleri `pdf2image` ile görüntüye, ardından `pytesseract` ile metne dönüştürüldü.

### 2. Web Scraping
`Selenium` ile iş hukuku soru-cevap siteleri tarandı, veriler LLaMA chat formatına dönüştürüldü:

```json
{
  "id": "ishukuku-1",
  "conversations": [
    {"from": "user", "value": "İş hukuku nedir?"},
    {"from": "assistant", "value": "İş hukuku, iş sözleşmesinden doğan haklardır..."}
  ]
}
```

---

## 🤖 Fine-Tuning Detayları

### QLoRA Konfigürasyonu
```python
# 4-bit Quantization
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=bfloat16
)

# LoRA Adapter
LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

### Eğitim Parametreleri
```python
TrainingArguments(
    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    fp16=True
)
```

### Training Loss
| Step | Loss |
|------|------|
| 10 | 2.3201 |
| 20 | 1.9510 |
| 30 | 1.6496 |
| 40 | 1.6100 |

---

## 🔍 RAG Pipeline

```python
def generate_answer(question, faiss_index, texts, embed_model):
    # 1. Soruyu embedding'e dönüştür
    vec = embed_model.encode([question]).astype("float32")
    
    # 2. FAISS'te en yakın 2 belgeyi bul
    _, indices = faiss_index.search(vec, top_k=2)
    chunks = [texts[i][:400] for i in indices[0]]
    
    # 3. Context + soru → model
    context = "\n\n".join(chunks)
    messages = [{"role": "user", "content": f"{context}\n\nSoru: {question}"}]
    
    # 4. Yanıt üret
    ...
```

---

## 🧪 Örnek Çıktılar

**Soru:** İş hukuku nedir?

**Yanıt:** İş hukuku, iş sözleşmesinden doğan hakların yanı sıra işçilerin işverene karşı haklarıdır.

**Kaynak:** Yargıtay 10. Hukuk Dairesi, 2021/11829E., 2021/17004K.

---

**Soru:** Bir işçi işini evinden yaparsa, işçinin evi işyeri sayılır mı?

**Yanıt:** İşçi evden çalıştığı halde, işçinin evi işyeridir.

**Kaynak:** Yargıtay 9. Hukuk Dairesi, 2021/12055E., 2021/16455K.

---

## ⚠️ Sınırlılıklar

- Model yalnızca Türk iş hukuku domain'i için optimize edilmiştir
- Üretilen yanıtlar hukuki tavsiye niteliği taşımaz, yalnızca bilgilendirme amaçlıdır
- Profesyonel hukuki danışmanlık için bir avukata başvurunuz

---

## 👥 Geliştirici Ekip

**İbrahim Yazıcı** — Model geliştirme, RAG pipeline, fine-tuning  
[github.com/yazicibrahim](https://github.com/yazicibrahim) · [linkedin.com/in/ibrahimyzc](https://linkedin.com/in/ibrahimyzc)

**İsmail Cem Özkan** — Veri toplama, sistem tasarımı

---

🤗 **Fine-tuned model:** [huggingface.co/ibrhmyzc/lawqamodel](https://huggingface.co/ibrhmyzc/lawqamodel)
