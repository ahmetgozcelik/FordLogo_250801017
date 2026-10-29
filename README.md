# Ford Logosu Veri Seti — Derin Öğrenme Ödev 1

Derin Öğrenme Yüksek Lisans dersi, "Temel Derin Öğrenme Süreci ile Etiketleme Projesi" ödevi.

| Klasör / dosya | İçerik |
|---|---|
| `FordLogo_250801017/` | Veri seti (YOLO + COCO), `metadata.csv`, scriptler ve veri seti kartı (`README.md`) |
| `Rapor_250801017/` | IEEE formatında rapor (`main.tex`), şekiller ve şekilleri üreten `make_figures.py` |
| `Odev1_250801017_AhmedGurkanOzcelik.pdf` | Raporun PDF hali |

Veri setinin ayrıntıları, etiketleme kuralları, istatistikler ve scriptlerin çalıştırma sırası
`FordLogo_250801017/README.md` dosyasındadır.

Görseller web aramasıyla toplanmıştır ve yalnızca ders amaçlı kullanılmaktadır; kaynak adresleri
`metadata.csv` içindedir. Bu nedenle depo herkese açık değildir.

Doğrulama:

```bash
cd FordLogo_250801017/scripts
python check_dataset.py
```
