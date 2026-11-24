# Yorgunluk tahmini sonuç arşivi

**Kaynak:** Önceki PMData geliştirme deneylerinin yerel CSV/JSON özetleri. Bunlar bu `damla` branch'inde tekrar çalıştırılmış sonuçlar **değildir**.

**Kapsam:** 14 geliştirme katılımcısı (1.512 kişi-gün, 113 yüksek yorgunluk günü). İki mühürlü test katılımcısı modele uygulanmadı. Bu depoya tekil kişi-gün tahminleri, ham yorgunluk kayıtları, `fatigue_daily.csv` veya `split.json` yüklenmedi.

## Hangi çıktı nerede?

| Çalışma | Kod | GitHub'a kaydedilen özet çıktı |
|---|---|---|
| Tarihsel kronolojik regresyon (eski protokol) | `02_yorgunluk_tahmini/train_fatigue_model.py` | `train_fatigue_model/historical_chronological_metrics.csv` (orijinal ml README özeti; yeni grup CV ile karşılaştırma yapmayın) |
| İlk grup doğrulama ve regresyon | `02_yorgunluk_tahmini/train_grouped.py` | `train_grouped/historical_validation_metrics.json`, `historical_validation_summary.csv` |
| Önceki 13 ile yeni 13 | `02_yorgunluk_tahmini/experiment_high_fatigue_new_13_features.py` | `experiment_high_fatigue_new_13_features/historical_metrics.csv` |
| 9 çekirdek + 4 uyku | `02_yorgunluk_tahmini/experiment_high_fatigue_core9_sleep4.py` | `experiment_high_fatigue_core9_sleep4/historical_metrics.csv` |
| Tek tek özellik çıkarma | `06_deneyler/individual_feature_ablation.py` | `individual_feature_ablation/individual_feature_ablation_metrics.csv` (orijinal), ayrıca `metrics.csv` (yuvarlanmış özet) |
| Özellik kombinasyonları | `06_deneyler/feature_combination_followup.py` | `feature_combination_followup/feature_combination_metrics.csv` (orijinal), `metrics.csv` (özet), `feature_combination_bootstrap.json` |
| İç/dış özellik karşılaştırması | Geçmiş ayrı deney; aynı kaynaktaki kodun birebir kopyası bu repoda yok | `feature_nested/feature_nested_metrics.csv`, `metrics.csv` |
| 25 özellik / uyku / nabız / antrenman | `06_deneyler/experiment_25_feature_groups.py` | `experiment_25_feature_groups/classification_metrics_exact.csv`, `regression_metrics_exact.csv`, `feature_missingness.csv`, ayrıca okunabilir özet CSV'ler |
| L1, elastic net, XGBoost, LightGBM, RF | `06_deneyler/compare_recommended_algorithms.py` | `compare_recommended_algorithms/metrics.csv` (orijinal) |
| Nested model karşılaştırması (CatBoost dâhil) | `06_deneyler/nested_model_selection.py` | `nested_model_selection/nested_metrics.csv`, ayrıca `metrics.csv` (özet) |
| Günlük değişim / başlangıç | `06_deneyler/daily_change_exploratory.py` | `daily_change_exploratory/daily_change_metrics.csv`, `metrics.csv` (özet) |
| Sensör özelliği denemeleri | Önceki yerel tarama; çalıştırma betiği korunmamış | `feature_trial/feature_trial_metrics.csv`, `metrics.csv` (özet) |
| Group7 / LOPO sağlamlık taraması | Önceki yerel tarama; çalıştırma betiği korunmamış | `robustness/robustness_metrics.csv`, `metrics.csv` (özet) |
| Weighted GBM | `06_deneyler/weighted_gbm_exploratory.py` | `weighted_gbm_exploratory/metrics.csv` |
| Sınıf ağırlığı deneyi | `06_deneyler/compare_logistic_class_weight.py` | `compare_logistic_class_weight/metrics.csv` |

`historical_*.csv` dosyaları geçmiş çalışmalardan kopyalanmıştır. `*_exact.csv` / kaynakla aynı adla yüklenen dosyalar orijinal hassasiyettedir; bazı `metrics.csv` dosyaları önce hazırlanmış **4 ondalıklı özet** niteliğindedir. Aynı grupta iki sonuç dosyası varsa orijinal hassasiyetli olanı raporlamada tercih et.

## Henüz yeniden üretilmesi gereken standart raporlar

`experiment_cold_start` (E1), `experiment_ablation` (E2), `experiment_high_fatigue` (E3), `experiment_lopo` ve `experiment_fatigue_change_errors` betikleri GitHub'dadır; ancak **bu branch'te yeniden çalıştırılmış tam CSV/JSON çıktıları yoktur**. Eski yerel çıktıların tamamı elimizde olmadığından bunları varmış gibi eklemedik. Aynı resmî split ile yerelde çalıştırılırsa kendi adları altına yazılacaktır.

**Orijinal** `outputs/split.json` (SHA256: `8f053e7dc1ee6a4fd7d34a7a590079a11fe53cc9036328b27a63f2fc1a76437e`) ve `outputs/fatigue_daily.csv` yerel makinede bulunmalıdır. Yeni bir split üretmeyin.

## Çalıştırma

```powershell
python -m pip install -r requirements.txt
python run_fatigue_experiments.py
python run_fatigue_experiments.py --check
```

İsteğe bağlı keşif deneyleri için:

```powershell
python -m pip install -r requirements_exploratory.txt
python run_fatigue_experiments.py --exploratory
python run_fatigue_experiments.py --check --exploratory
```

Bu ek deneyler önceki geliştirme verisinde defalarca özellik/model denemesinden sonra yapılmıştır; bunların sonuçları bağımsız doğrulama değildir. Yeni ankete dayalı özelliklerin zaman damgaları gerçek kullanım için ayrıca doğrulanmalıdır. Tüm tahmin skorları araştırma amaçlıdır; klinik kullanım veya egzersiz güvenlik kararı için onaylanmış değildir.

`outputs/` gitignore kapsamındadır, çünkü çalıştırıldığında bireysel tahmin kayıtları ve modeller oluşabilir. **Açık repoya yalnızca kişisel veri içermeyen toplulaştırılmış sonuçlar bilinçli olarak eklenmiştir.** Gelecekte yeni özet sonuçları eklemek için seçili dosyaları `git add -f outputs/<deney>/metrics.csv` benzeri açık bir komutla aşamalayın; ham günlük tahminleri paylaşmayın.
