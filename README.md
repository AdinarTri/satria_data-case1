# CASE 1 BDC 2026 — Klasifikasi Tweet MBG

**Nama Tim:** _FOTOIN QR ABSENNYA WOI!!_
**Anggota:** _Adinar Tri Panuntun, Muhammad Caesar Rivaldo, Darvesh Gladwin Musyaffa_
**Institusi:** _Telkom University_

Submission untuk Case 1 BDC 2026 — klasifikasi multikelas wacana publik di platform X mengenai program Makan Bergizi Gratis (MBG) ke dalam 8 kelas: Anggaran, Kualitas Pangan, Distribusi, Ekonomi, Tata Kelola, Sasaran Penerima, Politik, dan Lainnya.

---

## Ringkasan Pendekatan

Submission ini menggunakan **soft-vote ensemble** dari dua *encoder-only pretrained language model* yang di-*fine-tune* pada data berlabel panitia:

1. **IndoBERTweet** (`indolem/indobertweet-base-uncased`) — BERT yang di-pretrain pada korpus tweet berbahasa Indonesia, sangat cocok dengan domain data MBG yang berasal dari platform X.
2. **IndoRoBERTa** (`flax-community/indonesian-roberta-base`) — RoBERTa yang di-pretrain pada korpus OSCAR Indonesia. Menggunakan tokenizer byte-level BPE dan objective pretraining yang berbeda dari IndoBERTweet, sehingga menghasilkan error pattern yang komplementer dalam ensemble.

Setiap model dilatih menggunakan **stratified 5-fold cross-validation**, sehingga total 10 model fine-tuned (5 fold × 2 backbone). Prediksi akhir adalah hasil soft-vote: rata-rata probabilitas dari seluruh 10 model, lalu argmax untuk mendapatkan label.

### Hasil OOF (Out-of-Fold)
| Model | Balanced Accuracy (5-fold OOF) |
|---|---|
| IndoBERTweet (single) | 0.6893 |
| IndoRoBERTa (single) | 0.6593 |
| **Ensemble (IndoBERTweet + IndoRoBERTa)** | **0.6914** |

Skor OOF diukur dengan menggabungkan prediksi setiap fold pada validation set masing-masing, lalu menghitung balanced accuracy global. Karena setiap tweet dievaluasi tepat sekali pada fold di mana ia menjadi validation, skor ini merupakan estimasi yang stabil terhadap performa di data uji.

---

## Kepatuhan terhadap Aturan Panitia

Submission ini patuh sepenuhnya terhadap seluruh batasan pada Petunjuk Teknis:

- **(a) Data:** Hanya menggunakan `case_1_labeled_data.xlsx` (5.000 tweet) untuk training. Tidak ada data eksternal.
- **(b) Anotasi manual:** Tidak ada. Seluruh label data uji 100% berasal dari output model.
- **(c) Generative AI / LLM:** Tidak digunakan untuk prediksi. Tidak ada akses ke ChatGPT, Claude, Gemini, atau LLM lain pada saat inference.
- **(d) Anotasi ulang / augmentasi label / pseudo-labeling:** Tidak digunakan. Label data latih tidak diubah, tidak ditambah, tidak di-augment. Tidak ada pseudo-labeling terhadap data uji.
- **(e) Pretrained LM untuk fine-tuning:** Diizinkan secara eksplisit. IndoBERTweet dan IndoRoBERTa adalah model encoder-only yang di-fine-tune pada data panitia, **bukan** generative AI dan **tidak** digunakan dalam kapasitas zero-shot atau few-shot.

---

## Struktur Repository

```
├── README.md                         
├── requirements.txt                  
├── indobert_multi_backbone_(1).ipynb           
└── Adinar Tri Panuntun_FOTOIN QR ABSENNYA WOI!!.xlsx                   
```

Catatan: file dataset (`case_1_labeled_data.xlsx`, `case_1_text_to_predict.xlsx`, `case_1_template_sheet.xlsx`) tidak disertakan dalam repository karena merupakan properti panitia. Untuk reproduksi, file-file tersebut harus ditempatkan di folder yang sesuai (lihat bagian "Cara Menjalankan").

---

## Persyaratan Sistem

- **Python:** 3.10 atau lebih baru
- **GPU:** NVIDIA dengan minimum 12 GB VRAM (T4 di Google Colab cukup)
- **Runtime estimasi:** ~30–40 menit pada T4 GPU untuk training kedua backbone + inference

---

## Dependencies

Lihat `requirements.txt`. Versi utama:

```
transformers==4.44.2
accelerate==0.34.2
torch>=2.0
scikit-learn>=1.3
pandas>=2.0
openpyxl>=3.1
numpy>=1.24
```

Untuk instalasi:
```bash
pip install -r requirements.txt
```

---

## Cara Menjalankan

Notebook ini dirancang untuk dijalankan di **Google Colab dengan T4 GPU**.

### Langkah 1: Setup
1. Buka `notebooks/train_and_predict.ipynb` di Google Colab.
2. Ubah runtime ke GPU: `Runtime → Change runtime type → T4 GPU`.
3. Upload tiga file dataset panitia ke folder kerja, atau (seperti pada pengembangan) mount Google Drive dan letakkan ketiga file di `/content/drive/MyDrive/satria/`:
   - `case_1_labeled_data.xlsx`
   - `case_1_text_to_predict.xlsx`
   - `case_1_template_sheet.xlsx`

### Langkah 2: Urutan Eksekusi Cell
Jalankan cell secara berurutan dari atas ke bawah:

| Cell | Fungsi |
|---|---|
| 1 | Install dependencies (transformers, accelerate, dll) |
| 2 | Import library, set random seed (=42), inisialisasi device |
| 3 | Load dataset, text cleaning, label encoding, hitung class weights |
| 4 | Definisi class Dataset + fungsi training/inference/evaluation |
| 7 | Train IndoBERTweet (5-fold CV) |
| 12 | Train IndoRoBERTa (5-fold CV) |
| 14 | Tampilkan summary table semua model yang sudah dilatih |
| 15 | Tampilkan semua kombinasi ensemble dan skor OOF-nya |
| 16 | Generate file prediksi final (`Adinar Tri Panuntun_FOTOIN QR ABSENNYA WOI!!.xlsx`) |

Cell 8, 9, 10, 11, 13 (backbone tambahan dan stacking) bersifat opsional — tidak dipakai dalam submission final. Boleh di-skip.

### Langkah 3: Output
Setelah Cell 16 selesai, file `Adinar Tri Panuntun_FOTOIN QR ABSENNYA WOI!!.xlsx` akan tersimpan di lokasi yang ditentukan oleh variabel `OUT_PATH` di dalam cell. Default: `/content/drive/MyDrive/satria/Adinar Tri Panuntun_FOTOIN QR ABSENNYA WOI!!.xlsx`.

---

## Detail Teknis

### Preprocessing
- Unescape HTML entities (`&amp;` → `&`, dst.)
- URL diganti dengan token `[URL]`
- Mention `@user` diganti dengan token `[USER]`
- Whitespace dinormalisasi
- Tidak ada lowercasing manual (ditangani oleh tokenizer model)
- Duplikasi `full_text` di-drop (1 baris)

### Training Configuration

**IndoBERTweet:**
- `max_len=128`, `batch_size=16`, `learning_rate=2e-5`, `epochs=5`
- `warmup_ratio=0.1`, `weight_decay=0.05`
- Optimizer: AdamW dengan decoupled weight decay (skip bias & LayerNorm)
- Loss: Cross-entropy dengan inverse-frequency class weights
- Mixed precision: FP16
- Gradient clipping: 1.0

**IndoRoBERTa:**
- `max_len=160`, `batch_size=16`, `learning_rate=2e-5`, `epochs=4`
- `warmup_ratio=0.1`, `weight_decay=0.01`
- Konfigurasi lain identik dengan IndoBERTweet

### Cross-Validation
- 5-fold StratifiedKFold dengan `random_state=42` dan `shuffle=True`
- Per fold: model di-fine-tune ulang dari pretrained weights, snapshot epoch dengan balanced accuracy validasi terbaik disimpan
- OOF prediction terbentuk dari prediksi validation tiap fold
- Test prediction = rata-rata softmax probability dari 5 fold

### Ensemble Strategy
- **Soft voting** (averaging softmax probabilities), bukan hard voting
- **Equal weights** (50/50 antara IndoBERTweet dan IndoRoBERTa)
- Argmax dari rata-rata probabilitas untuk label final

### Class Imbalance Handling
- Distribusi kelas tidak seimbang (Ekonomi hanya ~3%, Kualitas Pangan ~23%)
- Karena metrik evaluasi adalah **balanced accuracy**, kelas minoritas memberikan kontribusi yang sama besarnya dengan kelas mayoritas
- Solusi: weighted cross-entropy loss dengan bobot = `n_samples / (n_classes × class_count)` (inverse frequency)

### Reproduktibilitas
- `random_state=42` digunakan di seluruh komponen: numpy, torch, random, sklearn
- `torch.cuda.manual_seed_all(42)` untuk GPU
- Catatan: operasi GPU non-deterministik (cuDNN) dapat menyebabkan variasi minor (<0.005 balanced accuracy) antar run meskipun seed identik. Ini umum pada training neural network dan tidak menandakan masalah metodologi.

---

## Hasil Eksperimen Tambahan

Selama pengembangan, beberapa backbone dan strategi lain diuji namun tidak dipilih untuk submission final:

| Backbone / Strategi | OOF Balanced Accuracy | Catatan |
|---|---|---|
| IndoBERTweet (single) | 0.6893 | Tetap dipakai dalam ensemble |
| IndoRoBERTa (single) | 0.6593 | Tetap dipakai dalam ensemble |
| **IndoBERTweet + IndoRoBERTa ensemble** | **0.6914** | **Dipilih untuk submission** |
| Stacking (IndoBERTweet [CLS] + XGBoost) | 0.6577 | Tidak dipakai — di bawah single IndoBERTweet |
| Stacking + ensemble 3-arah | 0.6825 | Tidak dipakai — stacking menurunkan skor ensemble |

Backbone yang sempat direncanakan namun belum di-train karena keterbatasan waktu/compute: IndoBERT base, IndoBERT large, XLM-R base, XLM-R large. Cell untuk training model-model ini tetap tersedia di notebook bagi yang ingin mereproduksi atau mengembangkan lebih lanjut.

---

## Lisensi & Kredit

Source code original dibuat oleh tim _[nama tim]_ untuk keperluan kompetisi BDC 2026.

Model pretrained yang digunakan:
- IndoBERTweet — Koto, F., et al. (2021). _IndoBERTweet: A Pretrained Language Model for Indonesian Twitter with Effective Domain-Specific Vocabulary Initialization._ EMNLP 2021.
- IndoRoBERTa — Flax Community (2021). Dipublikasikan di Hugging Face: `flax-community/indonesian-roberta-base`.

