# FordLogo_250801017 — Ford Logosu Tespit Veri Seti

Ford logosunu sınırlayıcı kutu (bounding box) ile işaretleyen tek sınıflı bir nesne tespit veri seti.
Görsel düzeyindeki `has_logo` sütunu sayesinde aynı veri seti logo var/yok **sınıflandırma** için de kullanılabilir.

- **Sınıf:** `0 = ford_logo` (tek sınıf)
- **Etiket formatı:** YOLO (`<sinif> <x_merkez> <y_merkez> <genislik> <yukseklik>`, 0–1 normalize, 6 ondalık)
- **Ek format:** COCO JSON (`annotations/instances_{train,val,test}.json`)
- **Etiketleme aracı:** Roboflow (YOLOv8 export)
- **Toplama aracı:** `icrawler` (BingImageCrawler) — `scripts/download_images.py`
- Görseller yalnızca ders amaçlıdır; veri seti herkese açık bir yerde paylaşılmamalıdır. Her görselin kaynak URL'si `metadata.csv` içindedir.

## İstatistikler

<!-- STATS -->
| Bölüm | Görsel | Pozitif | Negatif | Kutu |
|---|---:|---:|---:|---:|
| train | 489 | 255 | 234 | 377 |
| val | 105 | 56 | 49 | 91 |
| test | 105 | 55 | 50 | 93 |
| **Toplam** | **699** | **366** | **333** | **561** |

- Görsel başına kutu (pozitiflerde): ort. 1.53, maks. 24
- Kutu alanı / görsel alanı: medyan %4.82, min %0.010, maks %99.4
- Kutu kısa kenarı < 32 px olan küçük logo sayısı: 58
- Çözünürlük: genişlik 369-5472 px, yükseklik 320-6000 px, medyan 1140x956

Kanal bazında ortalama / std (RGB, 0–1 ölçek, yalnızca train kümesi, 835,114,169 piksel):

```python
mean = [0.4822, 0.4749, 0.4899]
std  = [0.3474, 0.3452, 0.3410]
```
<!-- /STATS -->

Grafikler: `istatistikler/` (kutu sayısı histogramı, kutu boyutu dağılımı, çözünürlük dağılımı, pozitif/negatif dağılımı).

## Klasör yapısı

```
FordLogo_250801017/
├── images/{train,val,test}/      ford_XXXX.jpg, neg_XXXX.jpg (orijinal boyut)
├── labels/{train,val,test}/      aynı adlı .txt (negatiflerde boş dosya)
├── data.yaml                     YOLO veri tanımı
├── classes.txt                   ford_logo
├── metadata.csv                  file_name,split,has_logo,num_boxes,width,height,source_url,notes
├── annotations/                  COCO JSON (ek puan)
├── scripts/                      tüm toplama / ön işleme / bölme / kontrol scriptleri
├── augmentation_ornekleri/       veri artırma örnekleri (sadece rapor, eğitimde kullanılmaz)
├── letterbox_ornekleri/          224x224 letterbox denemesi (sadece rapor)
├── kalite_kontrol/               kutuları çizilmiş 40 rastgele görsel
└── istatistikler/                rapor grafikleri
```

## Veri toplama

`icrawler` ile Bing'den sorgu bazlı indirme yapıldı (her sorgu ayrı alt klasöre, kaynak URL'ler `raw_urls.csv`'ye):

- **Pozitif sorgular (farklı bağlam/açı/ölçek/ışık):** araç ön ızgara amblemi, arka bagaj rozeti, bayi tabelası (gece dahil), direksiyon logosu, anahtarlık, F-150 ön amblemi, Mustang + Ford rozeti, eski (vintage) Ford rozetleri.
- **Negatif sorgular (zor negatifler):** Subaru / Hyundai / Toyota (benzer oval amblemler), Kia, rozetsiz (debadged) araç ön ızgaraları, mavi/oval şirket logoları.
- **Küçük/uzak logo turu (`scripts/download_small_logos.py`):** ilk iki turdan sonra kısa kenarı < 32 px olan logo yalnızca 19 taneydi (çoğu görsel yakın çekim/ürün fotoğrafı). Bu yüzden geniş açılı sahneler için 10 ek sorgu yapıldı: bayi binası dış cephesi, cadde/otopark, yoldaki F-150 / Transit / Ranger / Explorer / Focus / Kuga, polis aracı, fuar ve araba şovu kalabalığı. 292 ham görselden aynı ön işleme kurallarıyla (≥320 px, phash tekrar kontrolü mevcut tüm görsellere karşı) 280 aday kaldı; bunlar elle incelenip logonun küçük göründüğü 50 sahne seçildi (yandan çekilmiş/logo görünmeyen ve stüdyo ürün fotoğrafları alınmadı). Sonuçta küçük logo sayısı **19 → 58** oldu.
- Ford logosu içermeyen pozitif sorgu sonuçları (yalnızca Mustang atı, F-150 / Raptor / ST model yazısı, BMW bayisi vb.) **zor negatif** olarak negatiflere aktarıldı (`notes` sütununda belirtilir).

## Ön işleme (`scripts/preprocess.py`, `scripts/reindex.py`)

1. EXIF yön bilgisi `PIL.ImageOps.exif_transpose` ile görsele uygulandı; kayıt sırasında EXIF yazılmadığı için silindi. Bu işlem etiketlemeden **önce** yapıldı.
2. Kısa kenarı 320 px'den küçük görseller elendi.
3. Tüm görseller RGB'ye dönüştürülüp `.jpg` (kalite 95) olarak kaydedildi (gri/CMYK/RGBA/PNG/WebP dahil).
4. Tekrarlar perceptual hash (`imagehash.phash`, Hamming mesafesi ≤ 4) ile temizlendi; farklı boyuttaki kopyalar da yakalanır.
5. Aşırı bulanık / filigranı logoyu kapatan görseller elle silinir; `reindex.py` silinenleri atlayıp kalanlara sıralı adlar verir ve negatifler için boş etiket dosyası üretir.
6. Görseller **orijinal boyutlarında** saklanır; yeniden boyutlandırma ve veri artırma eğitim sırasında yapılacaktır.

### Etiketlerin aktarılması (`scripts/build_dataset.py`)

Roboflow'a ön işlenmiş görseller yerine ham görseller yüklendiğinden export'taki dosya adları (`000001_jpg.rf.<hash>.jpg`) ve bazı görseller (elenmiş küçük/tekrar eden kopyalar) farklıydı. Bu script her export görselini perceptual hash ile ön işlenmiş karşılığına eşleştirir, en-boy oranını doğrular ve normalize YOLO etiketlerini orijinal boyutlu görsele aktarır. Elenmiş görsellere ait etiketler kullanılmaz. Son adlandırma (`ford_0001.jpg …`, `neg_0001.jpg …`) bu adımda yapılır.

## Etiketleme kuralları

Tüm veri setinde tutarlı uygulanan kurallar:

- Kutu logoya **sıkı (tight)** oturur; logonun hiçbir parçası dışarıda kalmaz.
- Oval amblem görünüyorsa kutu **ovalin dış sınırını** kapsar. Yalnızca "Ford" yazısı (oval olmadan) varsa kutu yazıyı kapsar.
- Görseldeki **her** Ford logosu ayrı kutu ile işaretlenir.
- Kısmen kapanmış veya görsel kenarında kesilmiş logolar insan tarafından tanınabiliyorsa etiketlenir; kutu yalnızca görünen kısmı kapsar.
- Kısa kenarı ~10 px'den küçük, tanınamayan logolar etiketlenmez.
- Yansımalar, ekran/poster/tabela üzerindeki logolar tanınabiliyorsa etiketlenir.
- Negatif görsellerde kutu yoktur; boş `.txt` dosyası bulunur.

**Özel kararlar:**

1. **Oval içinde ikinci kutu yok:** Oval görünüyorsa yalnızca dış oval kutulanır; ovalin içindeki "Ford" yazısı ayrıca kutulanmaz. (İlk etiketlemede bu kurala uymayan iç kutular `build_dataset.py` ile otomatik temizlendi.)
2. **Mustang atı, F-150 / Raptor / ST / Focus gibi model rozetleri Ford logosu sayılmaz.** Bu görsellerde başka bir Ford oval/yazısı yoksa görsel zor negatiftir.
3. **Büyük harfli "FORD" yazısı** (tabela harfleri, eski arma üzerindeki FORD yazısı, bagaj kapağı harfleri) logo kabul edilir ve yazı kutulanır.
4. **Logo yazısı olmayan boş oval amblem** (ör. sadece siyah/mavi oval kapak) etiketlenmez.
5. **Logo ızgaraları** (çok sayıda şirket logosunun yer aldığı görseller) içindeki küçük Ford logosu kuralda belirtilen 10 px eşiğini geçiyorsa etiketlenir.
6. **Ürün fotoğraflarındaki üst üste dizilmiş amblemler:** her biri görünen kısmı kadar ayrı kutulanır.
7. Stok fotoğraf filigranları (alamy, dreamstime, shutterstock vb.) logoyu kapatmıyorsa görsel tutuldu.
8. **~10 px sınırı:** Kural "tanınamayacak kadar küçük" logoları dışarıda bırakır. Kısa kenarı 8–9 px olup oval şekli ve konumu (araç ızgarası/bagaj kapağı) sayesinde açıkça tanınan 5 kutu etiketlendi; `check_dataset.py` bunları hata değil **uyarı** olarak raporlar. Tanınamayan uzak logolar (ör. otoparkın arka sırasındaki araçlar) etiketlenmedi; hiç tanınabilir logosu kalmayan 2 görsel zor negatife geçti.
9. **Yönergedeki örnek görseller ile yazılı kural farkı:** Yönergenin sonundaki örnek görsellerde kutu yalnızca ovalin içindeki "Ford" yazısını kapsıyor; yazılı kural ise "oval görünüyorsa kutu ovalin dış sınırını kapsar" diyor. Tüm veri setinde **yazılı kural** uygulandı (kutu ovalin dış sınırı), çünkü kural açık ve ölçülebilir; tutarlılık için tek bir tanım seçildi.

## Train / Val / Test bölmesi (`scripts/split.py`)

- Oran: %70 train / %15 val / %15 test, sabit tohum `seed = 42`.
- **Tabakalı:** pozitif ve negatifler ayrı ayrı bölündü, oran her bölümde korunur.
- **Grup korumalı:** perceptual hash mesafesi ≤ 10 olan (aynı fotoğraf serisinden / çok benzer) görseller union-find ile gruplanıp grubun tamamı aynı bölüme atandı → veri sızıntısı önlenir.
- Bu bölme sonraki tüm ödevlerde aynen kullanılacaktır. Test kümesi yalnızca son değerlendirmede kullanılır.

## Normalizasyon

Kanal bazında ortalama/std yukarıdaki istatistik bloğundadır. Değerler yalnızca **train** kümesinden piksel bazında hesaplanmıştır (val/test bilgisinin modele sızmaması için).

## Ek denemeler (sadece rapor için)

- `letterbox_ornekleri/`: oran korunarak 224x224 letterbox (gri dolgu). Kutu dönüşümü: `x' = (x·W·r + dx) / 224`, `w' = w·W·r / 224` (r = ölçek, dx/dy = dolgu kayması). Sol: orijinal, sağ: dönüştürülmüş görsel ve kutu.
- `augmentation_ornekleri/`: albumentations ile yatay çevirme, ±15° döndürme, parlaklık/kontrast, ölçekleme, kutu korumalı kırpma, renk tonu, bulanıklık ve karma dönüşüm; kutular görselle birlikte dönüştürülür. Bu görseller train/val/test'e **karıştırılmaz** (veri sızıntısı).

## YOLO ve COCO farkı

| | YOLO | COCO |
|---|---|---|
| Dosya | görsel başına bir `.txt` | bölüm başına tek `.json` |
| Kutu | `x_merkez y_merkez genişlik yükseklik` | `[x_sol_üst, y_sol_üst, genişlik, yükseklik]` |
| Birim | görsel boyutuna bölünmüş (0–1) | piksel |
| Sınıf no | 0'dan başlar | `category_id` 1'den başlar |

Dönüşüm: `x_sol_üst = (x_merkez − w/2)·W`, `genişlik_px = w·W` (y/h için H ile).

## Kalite kontrol

- `scripts/visualize_labels.py`: 40 rastgele görsel kutularıyla çizildi (`kalite_kontrol/`). Ayrıca tüm pozitifler pano halinde gözden geçirildi.
- `scripts/check_dataset.py`: ders kapsamında resmi bir kontrol scripti paylaşılmadığı için yönergede listelenen tüm kontroller bu scriptte uygulandı: klasör yapısı, `data.yaml` / `classes.txt` içeriği, her görselin etiket dosyası (ve her etiketin görseli), dosya adı formatı, JPEG + RGB + EXIF yönü + ≥320 px, satır başına 5 değer, sınıf no = 0, koordinatların 0–1 aralığında olması ve kutunun görsel dışına taşmaması, negatif etiketlerin boş / pozitiflerin dolu olması, aynı dosyanın (ad veya MD5 içerik) birden fazla bölümde olmaması, `metadata.csv` sütunları ve her satırın dosyalarla tutarlılığı (split, has_logo, num_boxes, boyut, URL). Bağımsız çalışır: `python check_dataset.py [veri_seti_klasoru]`; hata varsa çıkış kodu 1. Bilerek bozulmuş bir kopya üzerinde denenerek hataları yakaladığı doğrulandı. Veri seti: **0 hata**, 5 uyarı (kısa kenarı 8–9 px olan, özel karar 8 kapsamındaki kutular).
- Yapılan düzeltmeler: negatif klasöründe Ford logosu içeren görseller (Ford oval tabelaları, logo ızgaraları) ayıklandı; Ford logosu içermeyen "pozitif" görseller zor negatife aktarıldı; oval içi ikinci kutular silindi; F-150 rozetine çizilmiş kutu kaldırıldı.
- İkinci etiketleme turunda (176 görsel) yalnızca Mustang atı / Dark Horse, F-150, Raptor, ST, FOCUS rozetlerine ve yazısız boş ovale çizilmiş 78 görseldeki kutular özel karar 2 ve 4 gereği kaldırıldı (`build_dataset.py` → `LABEL_OVERRIDES`); bu görseller zor negatif oldu. Aynı görselde ayrıca gerçek bir Ford ovali/yazısı bulunanlarda (ör. köşede küçük Ford ovali) yalnızca o kutu korundu.
- **Eski armalar (özel karar 3):** Yeniden etiketlemede `ford_0252` tutarsızlığı bulununca tüm eski arma görselleri (56) tarandı; 15 görselde kutunun armanın tamamını (ford_0288'de "F-100" yazısını da) kapsadığı görüldü. Bu görseller Roboflow'da yeniden etiketlendi ve yalnızca "FORD"/"Ford" yazısı kutulandı (`fordlogo7` export'u, `build_dataset.py` → `EXPORTS`).
- Üçüncü turda (50 küçük logo görseli) tüm kutular çizdirilip tek tek kontrol edildi; 64 kutu kurallara uygun bulundu, düzeltme gerekmedi.
- `scripts/relabel_check.py`: pozitiflerin %10'u yeniden etiketlenip IoU ile karşılaştırılır.

### %10 yeniden etiketleme sonucu

Pozitiflerin %10'u eski kutular görülmeden ayrı Roboflow projelerinde yeniden etiketlendi ve ilk etiketlerle greedy IoU eşleştirmesiyle karşılaştırıldı (`relabel_check.py compare`). Önce 32 görsel seçildi (seed 42). Küçük logo turu pozitif sayısını 366'ya çıkarınca oranı korumak için yeni turdan 5 görsel daha seçildi (`sample_extra 319 5`, seed 42). Toplam 37/366 = **%10.1**.

| Ölçüt (düzeltme öncesi) | İlk örnek | Ek örnek | Toplam |
|---|---:|---:|---:|
| Karşılaştırılan görsel | 32 | 5 | 37 |
| İlk etiketlemedeki kutu | 49 | 8 | 57 |
| Eşleşen kutu (IoU > 0.1) | 48 | 7 | 55 |
| Ortalama IoU (eşleşenler) | 0.976 | 0.937 | 0.971 |
| IoU ≥ 0.75 oranı | %100 | %100 | %100 |
| Tutarsız görsel | 1 | 1 | 2 |

Tutarsızlıklar:

1. `ford_0252` (eski Ford arması): ilk etiketlemede kutu tüm armayı kapsıyordu, ikinci etiketlemede yalnızca "FORD" yazısı kutulandı (IoU 0.07). Özel karar 3'e göre doğru olan ikinci etikettir; ilk etiket düzeltildi (`LABEL_OVERRIDES`).
2. `ford_0366` (otomobil fuarı): ikinci etiketlemede sol üstteki, arkadan aydınlatılmış düşük kontrastlı "Ford" tabelası (48×36 px) atlandı. Logo tanınabilir olduğundan ilk etiket (2 kutu) doğrudur ve korundu. Bu tür bir hata, düşük kontrastlı ve kalabalık sahnelerde gözden kaçma riskini gösteriyor.

Düzeltme sonrası ilk etiketlerin 57 kutusunun 56'sı eşleşiyor (tek fark ikinci etiketlemedeki atlanan kutu), ortalama IoU 0.972. Kalan küçük farklar (IoU 0.91–0.99) kutu kenarının 1–3 px oynamasından kaynaklanıyor.

## Bilinen sorunlar

- Bazı görsellerde stok fotoğraf filigranları vardır (logoyu kapatmayan).
- Görsellerin bir kısmı ürün/katalog fotoğrafıdır (beyaz arka plan); gerçek sahne görsellerine göre daha kolaydır.
- Küçük logolar (kısa kenar < 32 px) ek toplama turuyla 58'e çıkarıldı ama toplam kutuların hâlâ ~%10'u; uzak mesafe tespiti veri setinin en zayıf kısmıdır.
- Bazı görseller birden fazla logoyu (ör. 24 küçük amblem) içerir; kutu sayısı dağılımı çarpıktır.

## Scriptlerin çalıştırma sırası

```bash
# FordLogo1/Project klasöründe
python download_images.py      # ham görseller + raw_urls.csv
python preprocess.py           # EXIF, RGB/JPG, >=320px, phash tekrar temizliği
python reindex.py              # elle silme sonrası sıralı adlandırma
python download_small_logos.py download   # 3. tur: küçük/uzak logolu sahneler
python download_small_logos.py process    # elle seçimden sonra
# Roboflow'da etiketleme -> YOLOv8 export (FordLogo1, fordlogo, fordlogo3, fordlogo7)
# FordLogo_250801017/scripts klasöründe
python build_dataset.py        # etiketleri orijinal görsellere aktar
python split.py                # 70/15/15, seed 42, tabakalı + grup korumalı
python check_dataset.py        # yönerge kontrolleri, 0 hata olmalı
python visualize_labels.py
python letterbox_demo.py
python augment_examples.py
python export_coco.py
python dataset_stats.py        # istatistik + grafik + README güncelleme
python relabel_check.py sample              # %10 yeniden etiketleme örneği
python relabel_check.py sample_extra 319 5  # sonradan eklenen pozitiflerden ek örnek
python relabel_check.py compare ../../fordlogo2 ../../fordlogo5
```
