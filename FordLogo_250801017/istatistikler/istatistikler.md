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
