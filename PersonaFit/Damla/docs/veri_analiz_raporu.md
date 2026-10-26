# PersonaFit – PMData Veri Kalitesi ve Modelleme Planı

## Kaynak ve kapsam
Bu analiz kullanıcı tarafından yüklenen 3 parçalı orijinal 7z arşivindeki p01–p16 klasörlerine dayanmaktadır. Tarih eşleşmeleri dosyalardaki gün alanlarına göre yapılmıştır. Günlük satır sayısı steps ve calories tarih birleşimidir. Nabız dosyaları arşiv manifestosunda 16 kişinin tamamında doğrulandı; bazı büyük nabız JSON'ları çıkarılmadığından bunların iç veri kalitesine ilişkin inceleme ayrıca gereklidir.

**Toplam:** 16 kişi; **2.395 kişi-gün**, **1.728 günlük aktiviteyle eşleşen, hedefi bulunan kişi-gün**. Wellness dosyalarında toplam **1.747** etiketli anket günü vardır; aradaki **19** kayıt gün bazında aktivite takvimiyle eşleşmemektedir (başta zaman dilimi/gün hizalaması kontrol edilmelidir). Bu sayı, birleştirme için kullanılan tarih yorumuna bağlıdır.

## Katılımcı düzeyinde kayıtlar

| Kişi | Günlük satır | Etiketli gün | Uyku günü | Uyku puanı günü | Dinlenik nabız günü | sRPE günü | Önerilen bölüm |
|---|---:|---:|---:|---:|---:|---:|---|
| p01 | 152 | 138 | 150 | 150 | 152 | 31 | Eğitim |
| p02 | 152 | 99 | 142 | 138 | 91 | 84 | Eğitim |
| p03 | 152 | 78 | 78 | 73 | 152 | 2 | Eğitim |
| p04 | 152 | 144 | 144 | 139 | 35 | 82 | Test |
| p05 | 152 | 134 | 118 | 117 | 95 | 8 | Eğitim |
| p06 | 152 | 147 | 150 | 146 | 152 | 46 | Eğitim |
| p07 | 148 | 132 | 142 | 140 | 148 | 103 | Doğrulama |
| p08 | 143 | 103 | 133 | 132 | 143 | 107 | Eğitim |
| p09 | 152 | 102 | 136 | 131 | 152 | 3 | Doğrulama |
| p10 | 148 | 95 | 99 | 98 | 148 | 26 | Test |
| p11 | 152 | 107 | 119 | 119 | 98 | 28 | Eğitim |
| p12 | 152 | 83 | 8 | 1 | 0 | 39 | Eğitim |
| p13 | 152 | 68 | 51 | 47 | 0 | 10 | Eğitim |
| p14 | 139 | 123 | 121 | 114 | 140 | 55 | Eğitim |
| p15 | 145 | 93 | 143 | 140 | 145 | 95 | Eğitim |
| p16 | 152 | 82 | 147 | 145 | 152 | 11 | Eğitim |

## Eksik dosyalar

| Dosya | Mevcut katılımcı | Dosyası olmayan katılımcılar |
|---|---:|---|
| `fitbit/heart_rate.json` | 6/16 | p01, p02, p04, p05, p06, p07, p08, p11, p15, p16 |
| `fitbit/lightly_active_minutes.json` | 15/16 | p12 |
| `fitbit/resting_heart_rate.json` | 14/16 | p12, p13 |
| `googledocs/reporting.csv` | 15/16 | p15 |
| `pmsys/injury.csv` | 15/16 | p08 |

**Not:** `heart_rate.json` her 16 kişinin arşivinde bulunuyor. Bu raporda dosyanın varlığı arşiv manifestosundan kontrol edildi; boyut nedeniyle bütün nabız kayıtları satır satır analiz edilmedi.

## Önerilen sabit kişi bazlı ayrım (başlangıç planı)
- **Eğitim (12):** p01, p02, p03, p05, p06, p08, p11, p12, p13, p14, p15, p16. **1.255** eşleşen yorgunluk etiketi; **1.795** kişi-gün.
- **Doğrulama (2):** p07, p09. **234** etiketli gün; **300** kişi-gün.
- **Nihai test (2):** p04, p10. **239** etiketli gün; **300** kişi-gün.
Bu öneri kişi gruplarını birbirinden ayırır ve etiket sayılarını makul dengeler. Bölümlemeyi model sonuçlarına bakarak değiştirmeyin. Küçük katılımcı sayısı nedeniyle ek olarak *GroupKFold / Leave-One-Participant-Out* grup çapraz doğrulaması raporlayın; nihai test katılımcılarına özellik seçimi için dokunmayın.

## Eksik değer kuralları
1. **Hedef `wellness.fatigue`:** eksik olanı **asla impute etmeyin**; denetimli öğrenmede etiket bulunmayan satırları eğitim/puanlamadan çıkarın; günlük gözlem tablosunda tutabilirsiniz.
2. **Adım/kalori:** sensör kaydının olmaması ile gerçekten sıfır aktiviteyi ayırın. Günlük toplam yanında `steps_minutes_observed` / `calories_minutes_observed` sayın. Yetersiz kapsamlı günleri bayraklayın ve eğitim katılımcılarından öğrenilen kurallarla temizleyin. 1,440 dakikayı otomatik tamamlanmış kabul etmeyin.
3. **Uyku:** son gecenin uyku oturumunu uyandığı sabahla eşleştirin; bir günde birden fazla oturum varsa `mainSleep` kullanın, gerekirse toplam gece uykusu kuralını açıkça belirleyin. Eksik süre/puanı kişiye özgü veya eğitim verisinden elde edilen medyanla doldurmayı deneyin; `sleep_missing` ekleyin. p12 ve p13'te eksiklik çok büyük: imputasyonun etkisini ayrıca ölçün.
4. **Dinlenik nabız:** p12/p13 için dosya yok; eksik gösterge (`rhr_missing`) ve sadece eğitim grubunda öğrenilmiş medyan kullanılabilir. Kalıcı eksikliği gizlemeyin. p04'ün ölçümü yalnızca 35 gün; testte eksikliğe karşı dayanıklılığı özellikle raporlayın.
5. **sRPE:** kayıt yoksa otomatik 0 yük varsaymayın. Exercise kaydını karşılaştırıp gözlenen 'antrenman yok' durumu ile 'anket yok' durumunu ayırın. Ayrım net değilse `srpe_missing` bayrağı ve hassasiyet analizi yapın.
6. **Injury:** p08'de dosya yok; bunu 'sakatlık yok' olarak kabul etmeyin. Güvenlik mekanizması bilinmeyeni ayrı durum olarak taşımalı; geçmişte raporlanan sakatlık kaydının geçerlilik süresi tanımlanmalı.
7. **Beslenme/reporting:** p15 dosyası yok; diğerlerinde kapsam değişiyor. Temel modele zorunlu özellik yapmayın. Beslenmeli/beslenmesiz kıyaslamayı aynı kişi-günlerde, aynı bölünmeyle ayrı deney olarak yapın.
8. **Nabız:** günlük ortalama hesabında ölçüm sayısını, örnekleme aralığını ve güvenilirlik bilgisini göz önünde bulundurun. Eksik saatler varsa tam günlük ortalama gibi göstermeyin; ham veriyi ileriye dönük simülatör kalibrasyonu için yedekleyin.
9. **Genel:** sayısal dönüştürme, imputer, scaler, özellik seçimi ve hiperparametre ayarı yalnızca eğitim kişileriyle fit edilmeli, doğrulama/teste yalnızca uygulanmalı.

## Günlük tablo şeması
**Kimlik:** participant_id, date. **Hedef:** fatigue.
**Sabah tahmin girdileri:** prev_day_steps, steps_7d_mean, prev_day_calories, calories_7d_mean, prev_day_srpe_load, srpe_load_7d_mean, previous_day_activity_intensity, last_night_sleep_minutes, last_night_sleep_score, morning_resting_hr (gerçekte o saate kadar mevcutsa), missingness_flags.
**Aday ekler:** heart_rate_zones (sadece geçmiş), light/moderate/sedentary minutes, previous-day HR stats, nutrition variables.
**Sadece arşiv/sonuç:** bugünün tüm gün adım/kalori/egzersiz yükü (sabah tahmin girdisi olarak kullanılmaz).

### Teknik kontroller
- Tüm katılımcılar için tam günlük takvim oluşturun; `reindex` ile eksik günleri görünür yapın. `groupby(participant_id).shift(1)` kişi geçişindeki sızıntıyı engeller; takvim oluşturmazsanız `shift(1)` bazı kişilerde *bir önceki kaydedilmiş günü* verir, dünü değil.
- `wellness`/`srpe` UTC, Fitbit yerel olabilir. `tz_localize(None)` ile zaman dilimini gelişigüzel silmeden önce örnek günleri kontrol edin. Özellikle gece yarısı kayıtlarını denetleyin.
- Uyku bitiş anı ile sabah önerisinin saatini tutarlı hizalayın; aynı gün anketi öneri saatinden sonra doldurulmuşsa hedefle ilgili bilgi sızıntısı riskini işaretleyin.
- Eğitime giren etiketli satır sayısı nihai öznitelik kurallarından sonra değişir; 1.728 üst sınır/başlangıç sayısıdır, otomatik garantili eğitim satırı sayısı değildir.

## Deney tasarımı
1. Baseline-1: Ridge + Pipeline(imputer, scaler). Baseline-2: RandomForest + Pipeline(imputer).
2. Çekirdek özellik setiyle MAE/RMSE/R²; ardından ek yoğunluk ve nabız bölgesi gruplarıyla ablation. Aynı validation katılımcıları kullanılmalı.
3. Beslenme verisini isteğe bağlı üçüncü ablation olarak karşılaştırın; eksikliği yüksek kişilerden kaynaklanan yanlılığa dikkat edin.
4. Nihai modeli eğitim + doğrulama grubu üzerinde yeniden eğitip tek seferlik test sonuçlarını alın; simülatör parametrelerinin kalibrasyonunda test kişisi kullanılmamalı.
5. Gymnasium simülasyonunda Random, Rule-Based, Q-Learning, DQN; üç profil, 60 günlük bölümler, çoklu seed, güvenlik filtreli/filtersiz karşılaştırma. Gözlemsel veri egzersizin nedensel etkisini kanıtlamaz: geçiş/ödül fonksiyonu varsayımlarını yazın.

## Hemen uygulanacak sıra
1. Kaynak dosya envanterini sabitleyin ve bu CSV özetini proje deposuna koyun.
2. UTC/gün hizalamasını düzeltin; 19 eşleşmeyen wellness kaydını inceleyin.
3. Günlük takvim, sensör kapsama bayrakları, sabah kullanılabilirlik kontrolü ve eksik değer bayraklarını oluşturun.
4. Sabit kişi bazlı split'i kaydedin, imputasyon pipeline'larını yalnızca train ile fit edin.
5. Ridge, Random Forest ve ablation sonuçlarını doğrulamada karşılaştırın.
6. Ayrı Gymnasium/RL değerlendirmesine geçin.
