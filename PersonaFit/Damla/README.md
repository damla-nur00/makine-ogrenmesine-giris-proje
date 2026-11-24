# PersonaFit — Yorgunluk Tahmini (damla branch)

Bu klasörde PMData'dan **yorgunluk puanı regresyonu** ve **yüksek yorgunluk (fatigue >= 4) sınıflandırması** deneyleri bulunur. Egzersiz önerisi, simülasyon ve takviyeli öğrenme bu aşamaya dahil değildir.

## Yapı

- `01_veri_hazirlama/`: PMData dosyalarından günlük özellik üretimi.
- `02_yorgunluk_tahmini/`: model ve eğitim kodları; yeni 13 özellik ve 9 çekirdek + 4 uyku adayları.
- `06_deneyler/`: soğuk başlangıç, ablation, E3 yüksek yorgunluk, LOPO ve hata analizleri.
- `experiment_protocol.json`, `e3_candidate_config.json`: sabit protokol ve aday yapılandırması.
- `run_fatigue_experiments.py`: deneyleri sırayla çalıştırır ve sonuçları kontrol eder.

## Kurulum

Komutları **PersonaFit/Damla** dizininde çalıştır:

```powershell
python -m pip install -r requirements.txt
```

Ham PMData klasörleri (`p01`–`p16`) yerelde varsa günlük özellikleri hazırlamak için:

```powershell
python 01_veri_hazirlama/prepare_features.py --data-root . --features-out outputs/fatigue_daily.csv --audit-out outputs/data_audit.json
```

**Zorunlu:** Daha önce mühürlenmiş **orijinal** `outputs/split.json` dosyasını yerelde bulundur. GitHub'da yoktur ve yeniden oluşturulmamalıdır. Beklenen SHA256:

`8f053e7dc1ee6a4fd7d34a7a590079a11fe53cc9036328b27a63f2fc1a76437e`

Kontrol (PowerShell): `(Get-FileHash outputs/split.json -Algorithm SHA256).Hash`

## Çalıştırma

```powershell
python run_fatigue_experiments.py
python run_fatigue_experiments.py --check
```

Yalnızca yeni iki aday:

```powershell
python run_fatigue_experiments.py --only experiment_high_fatigue_new_13_features,experiment_high_fatigue_core9_sleep4
```

Tüm deneyler kişi bazlı geliştirme doğrulamasıdır; **mühürlü test kişilerinin sonuçları değerlendirilmez**. Çalıştırıcı başarısız deneyde durur; çıkan dosyaları ve çalışma durumunu `outputs/experiment_run_status.json` içinde kaydeder.

## Sonuçların isimlendirilmesi

Her betik kendi adıyla `outputs/<betik_adı>/` klasörüne sonuç üretir:

| Deney | Başlıca sonuç dosyası |
| --- | --- |
| `train_grouped.py` | `outputs/train_grouped/validation_metrics.json` |
| `experiment_cold_start.py` | `outputs/experiment_cold_start/summary.json` |
| `experiment_ablation.py` | `outputs/experiment_ablation/results.csv` |
| `experiment_high_fatigue.py` | `outputs/experiment_high_fatigue/results.csv` |
| `experiment_lopo.py` | `outputs/experiment_lopo/comparison.csv` |
| `experiment_fatigue_change_errors.py` | `outputs/experiment_fatigue_change_errors/summary.json` |
| `experiment_high_fatigue_new_13_features.py` | `outputs/experiment_high_fatigue_new_13_features/metrics.csv` |
| `experiment_high_fatigue_core9_sleep4.py` | `outputs/experiment_high_fatigue_core9_sleep4/metrics.csv` |

Günlük tablo ve ham katılımcı kayıtları, `split.json`, üretilen çıktılar ve eğitilmiş modeller **yerelde kalır** ve `.gitignore` nedeniyle GitHub'a yüklenmez. Çıktıların adlandırılması için kaynak betiklerin varsayılan `--out-dir` yolları eşleştirildi; ayrıca komut satırında `--out-dir` verilebilir.


Önceki geliştirme deneylerinde regresyonda kişisel medyan MAE yaklaşık **0,426**, GBM MAE yaklaşık **0,447**: ML, kişisel medyanı geçemedi. Sonraki keşif deneylerinde yeni 13 özellikli L2 lojistik regresyon AP yaklaşık **0,187**; 9 çekirdek + 4 uyku adayında Recall@10% yaklaşık **%32,7**. Bu değerler bu branch üzerinde **yeniden çalıştırılmış/onaylanmış sayılmaz** ve iki yeni aday henüz kesin seçim değildir.

Aynı 14 geliştirme katılımcısının tekrar kullanılması seçim yanlılığı oluşturabilir. Anket zaman damgaları bağımsız doğrulanmadı; model skorları kalibre klinik olasılıklar değildir. Kapalı test sonuçları açılmadı.

## Kapsam notu

25 özellik, L1/elastic net, XGBoost ve LightGBM ile yapılan bazı son keşif deneyleri önceki çalışmanın yerel betikleri/raporlarıdır; **bu branch'e henüz taşınmadı**. Sunumda bunları burada çalıştırılmış gibi göstermeyin.


## Tüm yorgunluk tahmini çalışmalarının arşivi (güncellendi)

Orijinal `ml` reposundaki ilk regresyon ve yorgunluk betikleri de aktarıldı: `train_fatigue_model.py`, `train_gbm_a_integration.py`, `real_predictor_v2.py` ve `07_testler/test_grouped.py`. **`test_grouped.py` mühürlü testi açacağı için çalıştırılmamalı.** Eski `train_fatigue_model.py` farklı bir kronolojik protokoldür; yeni kişi-bazlı sonuçlarla karşılaştırırken bu fark belirtilmelidir. Simülatöre bağımlı `evaluate_fixed_fatigue_v1.py` sadece tarihsel referanstır, bu sunumdaki tahmin çalıştırıcısına dahil değildir.

`06_deneyler` klasörüne ek olarak **8 keşif betiği** aktarıldı: `individual_feature_ablation.py`, `feature_combination_followup.py`, `experiment_25_feature_groups.py`, `compare_recommended_algorithms.py`, `nested_model_selection.py`, `daily_change_exploratory.py`, `weighted_gbm_exploratory.py`, `compare_logistic_class_weight.py`. Araştırma metotlarında bir değişiklik yapılmadı; kişisel `/mnt/data` dosya yolları yerel `outputs/fatigue_daily.csv` ve kendi `outputs/<betik_adı>/` çıktı klasörlerine uyarlandı.

Bu ek çalışmaları kurup çalıştırmak için:

```powershell
python -m pip install -r requirements_exploratory.txt
python run_fatigue_experiments.py --exploratory
python run_fatigue_experiments.py --check --exploratory
```

**Sonuç dosyaları** GitHub'da `outputs/` altında deney bazında arşivlendi. Kaynak, tamlık ve kişisel veri güvenliği notları için [outputs/README_RESULTS.md](outputs/README_RESULTS.md) dosyasına bak. Bu arşivdeki tablolar geçmiş geliştirme deneylerine aittir; **burada yeniden üretilmiş oldukları iddia edilmez**. Resmî E1, E2, E3 ve LOPO'nun tam günlük/katsal CSV çıktıları henüz yeni branch'te çalıştırılmadı; bunların kaydı ancak orijinal donmuş split ve PMData dosyalarıyla yerelde üretilebilir. `outputs/` gitignore kapsamındadır; açık GitHub'da özellikle toplu sonuç özetleri izlenir, kişi/gün bazlı tahminler izlenmez.
