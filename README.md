# Mini Project 2 — Object Counting: Deteksi dan Penghitungan Mobil pada Citra Area Parkiran

**Mata Kuliah:** Pengolahan Citra dan Video
**Nama:** Athaya Khairani Adi  
**NRP:** 5024241007  

---

## Jumlah Mobil Terdeteksi

Program mendeteksi **91 mobil** pada citra input `parking_ori.jpg`.

> Nilai ini dihasilkan secara otomatis berdasarkan pipeline thresholding dan filtering kontur dengan batasan area MIN\_AREA = 100 piksel dan MAX\_AREA = 2000 piksel.

---

## Penjelasan Pipeline

Pipeline yang digunakan dalam program ini adalah pendekatan **threshold-based** yang dikombinasikan dengan operasi morfologi. Setiap tahapan dirancang secara berurutan untuk menghasilkan segmentasi yang bersih sebelum dilakukan penghitungan objek.

### Tahap 1: Eksplorasi Color Space

Citra asli dikonversi ke tiga ruang warna berbeda untuk analisis awal karakteristik visual:

- **Grayscale** (`cv2.cvtColor` dengan `COLOR_BGR2GRAY`): Menghasilkan representasi intensitas piksel yang paling sederhana dan komputasional ringan.
- **Channel V dari HSV** (`COLOR_BGR2HSV`): Channel Value merepresentasikan kecerahan (brightness) piksel, berguna untuk membedakan objek terang (mobil) dari latar gelap (aspal).
- **Channel L dari LAB** (`COLOR_BGR2LAB`): Channel Lightness pada ruang warna LAB bersifat lebih perseptual dan seragam dibanding grayscale biasa.

**Alasan pemilihan:** Eksplorasi awal ini dilakukan untuk menentukan representasi terbaik yang memisahkan mobil dari aspal. Setelah analisis visual, channel **grayscale** dipilih karena menghasilkan kontras yang cukup baik antara badan mobil dan permukaan jalan.

---

### Tahap 2: Preprocessing dengan Gaussian Blur

```python
blur = cv2.GaussianBlur(gray, (5, 5), 0)
```

Gaussian Blur diterapkan pada citra grayscale menggunakan kernel berukuran 5x5. Operasi ini meratakan variasi intensitas piksel akibat noise, bayangan kecil, atau tekstur permukaan yang tidak relevan.

**Alasan:** Tanpa blurring, thresholding cenderung menghasilkan banyak bintik putih kecil (noise) yang nantinya akan salah dihitung sebagai objek. Gaussian Blur mengurangi frekuensi tinggi pada citra sehingga thresholding bekerja lebih stabil.

---

### Tahap 3: Thresholding dengan Metode Otsu

```python
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

Metode Otsu digunakan untuk binarisasi otomatis. Algoritma ini menghitung nilai threshold optimal secara statistik berdasarkan histogram citra, dengan memaksimalkan variansi antar kelas (piksel objek dan latar belakang).

**Alasan:** Metode Otsu tidak memerlukan penentuan nilai threshold secara manual, sehingga lebih adaptif terhadap variasi pencahayaan pada citra aerial. Hasilnya adalah citra biner di mana piksel putih merepresentasikan area objek (mobil) dan piksel hitam merepresentasikan latar belakang (aspal).

---

### Tahap 4: Operasi Morfologi

#### 4a. Morphological Closing

```python
morph_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
```

Closing merupakan kombinasi dilasi diikuti erosi. Operasi ini menutup celah atau lubang kecil di dalam area objek yang muncul akibat variasi warna permukaan kendaraan (kaca depan, atap, dst.).

**Alasan:** Badan mobil yang memiliki warna dan tekstur tidak seragam sering kali menghasilkan area putih yang tidak utuh setelah thresholding. Closing memastikan tiap kendaraan terbaca sebagai satu blob yang solid.

#### 4b. Morphological Opening

```python
morph_clean = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel, iterations=1)
```

Opening merupakan kombinasi erosi diikuti dilasi. Operasi ini menghilangkan bintik putih kecil sisa noise yang tersebar di area aspal.

**Alasan:** Setelah closing, masih mungkin terdapat artefak kecil yang tidak merepresentasikan kendaraan. Opening membersihkan artefak tersebut tanpa merusak struktur blob utama yang berukuran lebih besar.

---

### Tahap 5: Deteksi Kontur dan Filtering Berdasarkan Area

```python
contours, _ = cv2.findContours(morph_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

Fungsi `findContours` dengan mode `RETR_EXTERNAL` mendeteksi seluruh kontur luar (batas terluar) pada citra biner hasil morfologi. Setiap kontur kemudian difilter berdasarkan luas area:

| Parameter | Nilai | Keterangan |
|-----------|-------|-----------|
| `MIN_AREA` | 100 piksel | Batas bawah untuk mengabaikan noise kecil |
| `MAX_AREA` | 2000 piksel | Batas atas untuk mengabaikan area besar yang bukan mobil individu |

Kontur yang memenuhi kriteria area digambar sebagai bounding box berwarna hijau pada citra output, disertai nomor urut berwarna merah.

**Alasan pemilihan range area:** Mobil pada citra aerial memiliki ukuran piksel yang relatif seragam. Rentang 100 hingga 2000 piksel dipilih setelah pengamatan visual terhadap ukuran tiap kendaraan pada citra, sehingga noise dan objek non-kendaraan (misalnya marka jalan, bayangan besar) dapat dieksklusi.

---

## Visualisasi Tahapan Pipeline

Seluruh hasil antarpipeline disimpan di folder `output/steps/`. Berikut ringkasan visualisasi tiap tahapan:

| No. | Nama File | Deskripsi |
|-----|-----------|-----------|
| 1 | `1_color_space_gray.png` | Citra grayscale hasil konversi dari BGR |
| 2 | `1_color_space_hsv_v.png` | Channel V dari ruang warna HSV |
| 3 | `1_color_space_lab_l.png` | Channel L dari ruang warna LAB |
| 4 | `2_gaussian_blur.png` | Hasil setelah Gaussian Blur (kernel 5x5) |
| 5 | `3_otsu_threshold.png` | Citra biner hasil thresholding Otsu |
| 6 | `4_morphology_close.png` | Setelah Morphological Closing (2 iterasi) |
| 7 | `5_morphology_open_clean.png` | Setelah Morphological Opening (1 iterasi) |
| 8 | `output/result.png` | Hasil akhir dengan bounding box dan nomor urut |

---

## Analisis

### Kendala yang Dihadapi

1. **Mobil berdekatan atau bersentuhan:** Setelah thresholding dan morfologi, dua atau lebih kendaraan yang parkir berdekatan dapat bergabung menjadi satu blob tunggal. Akibatnya, program menghitung kelompok tersebut sebagai satu objek, sehingga jumlah deteksi menjadi lebih rendah dari jumlah sebenarnya.

2. **Variasi warna kendaraan:** Kendaraan berwarna gelap (hitam atau abu tua) memiliki intensitas piksel yang mendekati aspal, sehingga sulit dibedakan setelah thresholding sederhana berbasis grayscale.

3. **Bayangan dan pantulan cahaya:** Bayangan panjang yang jatuh dari kendaraan dapat memperbesar ukuran blob dan memengaruhi akurasi bounding box maupun hitungan area.

4. **Sensitivitas parameter MIN\_AREA dan MAX\_AREA:** Rentang area yang digunakan bersifat statis. Apabila resolusi citra berbeda, rentang ini perlu dikalibrasi ulang secara manual.

### Perkiraan Akurasi

Akurasi program dipengaruhi langsung oleh kondisi parkiran dalam citra:

- Kendaraan yang terparkir rapi dengan jarak cukup cenderung terdeteksi dengan baik.
- Kendaraan yang berimpitan atau berwarna gelap berpotensi tidak terdeteksi (false negative).
- Noise besar yang lolos dari morfologi berpotensi terhitung sebagai kendaraan (false positive).

### Potensi Peningkatan

1. **Watershed Algorithm:** Algoritma watershed (`cv2.watershed`) dapat memisahkan kendaraan yang berdekatan dan bergabung menjadi satu blob, sehingga penghitungan menjadi lebih akurat.

2. **Segmentasi berbasis warna (HSV/LAB):** Menambahkan filter warna untuk mengeksklusi area aspal secara eksplisit dapat meningkatkan kualitas biner awal sebelum morfologi.

3. **Distance Transform:** Kombinasi distance transform dengan thresholding lokal dapat membantu memisahkan objek yang saling berdekatan.

4. **Adaptive Thresholding:** Menggunakan `cv2.adaptiveThreshold` dapat menangani variasi pencahayaan lokal yang tidak merata pada citra aerial.

5. **Parameter area berbasis persentase resolusi:** Mengubah MIN\_AREA dan MAX\_AREA menjadi nilai relatif terhadap resolusi citra agar program lebih adaptif terhadap berbagai ukuran gambar.

---

## Cara Menjalankan Program

### Prasyarat

Pastikan Python versi 3.7 atau lebih baru sudah terpasang. Instal seluruh dependensi yang diperlukan menggunakan perintah berikut:

```bash
pip install opencv-python numpy matplotlib
```

### Struktur Direktori

Pastikan struktur folder berada dalam kondisi berikut sebelum menjalankan program:

```
mp2-object-counting/
├── README.md
├── counting.py
└── TUGAS/
    └── parking_ori.jpg
```

### Menjalankan Program

Jalankan perintah berikut dari direktori utama proyek:

```bash
python counting.py
```

### Output yang Dihasilkan

Setelah program selesai berjalan, hasil akan tersimpan secara otomatis pada:

```
TUGAS/
└── output/
    ├── result.png                       
    └── steps/
        ├── 1_color_space_gray.png
        ├── 1_color_space_hsv_v.png
        ├── 1_color_space_lab_l.png
        ├── 2_gaussian_blur.png
        ├── 3_otsu_threshold.png
        ├── 4_morphology_close.png
        └── 5_morphology_open_clean.png
```

Selain itu, program akan menampilkan dua jendela visualisasi matplotlib secara langsung: satu untuk perbandingan color space dan satu untuk alur kerja pipeline lengkap dari awal hingga deteksi akhir.
