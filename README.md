# Çekme Dayanımı (RmA) Tahmini – Kimyasal Kompozisyondan

Bu depo, çelik numunelerin **kimyasal kompozisyonundan** (ör. C, Si, Mn, P, S, Cr, Ni, Mo, V, Nb, Ti, B, N, Al, Cu…) **çekme dayanımı (RmA)** değerini tahmin etmeye yönelik bir makine öğrenmesi (ML) hattı içerir. Model; veri hazırlama, özellik mühendisliği, eğitim, değerlendirme ve tek satırlık CSV’den tahmin akışlarını kapsar.

> Not: Sanal ortam klasörü (`.venv/`) **depoya eklenmez.** Her kullanıcı kendi makinesinde yaratıp gereksinimleri yüklemelidir.

---

## 1) Önkoşullar

* **Python**: 3.10, 3.11 (öneri: 3.11)
* **Git**: 2.40+
* (İsteğe bağlı) **Conda/Miniconda** veya yalnızca `python -m venv`
* **pip**: 23+

---

## 2) Hızlı Başlangıç

### 2.1 Depoyu edin

```bash
# HTTPS
git clone <bu-deponun-url'si>.git
cd <depo-klasoru>
```

### 2.2 Sanal ortam (.venv) oluştur ve etkinleştir

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```bat
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS / Linux (bash/zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3 Gereksinimleri yükle

Depoda bulunan `requirements.txt` dosyasını kullanın:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Eğer `requirements.txt` yoksa, geliştirme makinenizden `pip freeze > requirements.txt` ile üretebilir ve repoya ekleyebilirsiniz.

---

## 3) Proje Yapısı (önerilen)

```
.
├── data/
│   ├── raw/                # Ham girdiler (laboratuvar/kimyasal analiz CSV’leri vb.)
│   └── processed/          # Temizlenmiş, özelliklendirilmiş veriler
├── notebooks/
│   └── eda.ipynb           # Keşifsel veri analizi (opsiyonel)
├── src/
│   ├── config.py | yaml    # Yol, hedef, model parametreleri
│   ├── data.py             # Okuma/temizleme/birleştirme fonksiyonları
│   ├── features.py         # Özellik mühendisliği (oranlar, log, ölçekleme, vb.)
│   ├── model.py            # Model tanımı ve kaydetme/yükleme
│   ├── train.py            # Eğitim/Değerlendirme CLİ’si
│   └── predict.py          # Tek satır veya toplu tahmin CLİ’si
├── models/                 # Kaydedilen modeller/artefaktlar (\.pkl, \*.joblib)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 4) Veri Beklentileri

* **Girdi sütunları**: En azından ana alaşım elementleri yüzdesel kompozisyonları (kütlece % veya ppm). Örn: `C, Si, Mn, P, S, Cr, Ni, Mo, V, Nb, Ti, B, N, Al, Cu`.
* (Varsa) proses ısıl bilgileri, döküm/ısı numarası, numune tipi vb. ek değişkenler modele girdi olabilir.
* **Hedef (label)**: `RmA` veya kuruluşunuzda kullanılan çekme dayanımı sütun adı. **Tahmin dosyasında hedef sütunu bulunmamalıdır.**

> Birleştirme için genellikle `IsıNo / HeatNo`, `NumuneNo / SampleID` gibi anahtarlar kullanılır. Sütun isimleri sizde farklıysa, `src/data.py` içindeki eşleştirme eşiklerini/haritalarını güncelleyin.

---

## 5) Kullanım

### 5.1 Eğitim

```bash
# Örnek: config ile eğitim
python -m src.train --config configs/train.yaml
# veya
python src/train.py --train data/processed/train.csv --target RmA --out models/rma_lgbm.pkl
```

Parametreler tipik olarak:

* `--train`: Eğitim veri yolu
* `--target`: Hedef sütun adı (örn. `RmA`)
* `--out`: Kaydedilecek model/artefakt yolu
* `--cv` ve `--metric`: Çapraz doğrulama ve metrik (MAE/RMSE/R²)

### 5.2 Tek satırlık CSV’den tahmin

```bash
python src/predict.py --model models/rma_lgbm.pkl \
                      --input data/only_row.csv \
                      --output predictions.csv
```

* `only_row.csv` **tek satır** ve **eğitimde kullanılanla aynı özellik adlarına** sahip olmalıdır.
* Çıktı dosyası `pred` veya `RmA_pred` alanı ile oluşturulur.

---

## 6) .gitignore

`.venv/` ve büyük geçici klasörler repoya dahil edilmez:

```gitignore
.venv/
venv/
__pycache__/
*.py[cod]
*.egg-info/
models/*.tmp
.ipynb_checkpoints/
build/
dist/
```

---

## 7) Çoğaltılabilirlik (Reproducibility)

* Tüm rasgelelik kaynakları için `random_state=42` vb. sabitleyin.
* Eğitimde kullanılan **özellik listesi**, **ölçekleyiciler** ve **dönüşümler** (`MinMaxScaler`, `PowerTransformer` vb.) `joblib` ile kaydedilip `predict.py` içinde yeniden yüklenmelidir.

---

## 8) Sık Karşılaşılan Sorunlar

* **`fatal: not a git repository`**: Proje kökünde `git init` ve ardından `git remote add origin ...` çalıştırın.
* **`.venv` repoya eklenmiyor**: Beklenen davranış. Herkes lokalinde kurmalı. Kopyalamak gerekiyorsa `robocopy`/`xcopy /H /E` ile klasörü taşıyın; ancak repoya itmeyin.
* **`ModuleNotFoundError`**: Sanal ortam etkin mi? (`which python`, `where python`). `pip install -r requirements.txt` uygulandı mı?
* **`pip` çok yavaş / hata**: `python -m pip install -r requirements.txt --default-timeout=100` deneyin. Kurumsal ağda proxy ayarlarını kontrol edin.

---

## 9) Lisans

Kuruluş politikanıza göre lisans ekleyin (örn. MIT, Apache-2.0). Kurumsal iç kullanım ise lisans dosyasında belirtin.

---

## 10) Katkı

* PR açmadan önce lütfen `pre-commit`/`flake8`/`black` (kullanıyorsanız) kontrollerini çalıştırın.
* Yeni veri kaynağı eklerken `data/processed` şemasını ve `features.py` içerisindeki dönüşümleri güncel tutun.

---

### Ek: Minimal `requirements.txt` (eğer `pip freeze` yoksa)

Aşağıdaki liste başlangıç için uygundur; sürümleri ihtiyacınıza göre sabitleyebilirsiniz.

```
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
xgboost>=2.0
lightgbm>=4.3
catboost>=1.2
pyyaml>=6.0
joblib>=1.3
matplotlib>=3.8
```

> Üretimde kullanılan kesin sürümler için geliştirme ortamınızda `pip freeze > requirements.txt` üretip bu dosyayı repoya ekleyin.
