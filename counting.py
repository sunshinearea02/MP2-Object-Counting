import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Membuat folder output dan steps secara otomatis jika belum tersedia
os.makedirs("TUGAS/output/steps", exist_ok=True)

# ==========================================
# 1. LOAD CITRA INPUT
# ==========================================
image_path = "TUGAS/parking_ori.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Gambar tidak ditemukan di {image_path}. Periksa folder input kamu!")
    exit()

# Salinan gambar asli untuk menggambar bounding box di akhir nanti
img_out = img.copy()

# ==========================================
# 2. EKSPLORASI COLOR SPACE & PENYIMPANAN
# ==========================================
# Mengubah citra ke berbagai color space untuk analisis awal
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Konversi ke HSV dan ambil channel V (Value / Kecerahan)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
_, _, v_channel = cv2.split(hsv)

# Konversi ke LAB dan ambil channel L (Lightness / Luminositas)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel, _, _ = cv2.split(lab)

# Menyimpan hasil eksplorasi ruang warna ke folder output/steps/
cv2.imwrite("TUGAS/output/steps/1_color_space_gray.png", gray)
cv2.imwrite("TUGAS/output/steps/1_color_space_hsv_v.png", v_channel)
cv2.imwrite("TUGAS/output/steps/1_color_space_lab_l.png", l_channel)

# ==========================================
# 3. PREPROCESSING (BLURRING) & PENYIMPANAN
# ==========================================
# Menerapkan Gaussian Blur untuk mengurangi noise bintik kecil sebelum thresholding
blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite("TUGAS/output/steps/2_gaussian_blur.png", blur)

# ==========================================
# 4. THRESHOLDING (METODE OTSU) & PENYIMPANAN
# ==========================================
# Menggunakan Otsu's thresholding untuk binarisasi otomatis
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("TUGAS/output/steps/3_otsu_threshold.png", thresh)

# ==========================================
# 5. OPERASI MORFOLOGI & PENYIMPANAN
# ==========================================
kernel = np.ones((3, 3), np.uint8)

# Langkah A: Closing untuk menutup lubang di dalam badan mobil
morph_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imwrite("TUGAS/output/steps/4_morphology_close.png", morph_close)

# Langkah B: Opening untuk menghilangkan bintik putih sisa noise di aspal
morph_clean = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imwrite("TUGAS/output/steps/5_morphology_open_clean.png", morph_clean)

# ==========================================
# 6. DETEKSI KONTUR & FILTERING AREA
# ==========================================
contours, _ = cv2.findContours(morph_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Parameter threshold luas area objek (dalam satuan piksel)
MIN_AREA = 100  
MAX_AREA = 2000 

car_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    
    # Seleksi kondisi: Hanya memproses kontur yang berukuran menyerupai mobil
    if MIN_AREA < area < MAX_AREA:
        car_count += 1
        
        # Mengambil koordinat kotak pembatas (Bounding Box)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Menggambar kotak hijau (BGR: 0, 255, 0) pada gambar output dengan ketebalan 2 piksel
        cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Menuliskan teks nomor urutan mobil di atas kotak pembatas
        cv2.putText(img_out, str(car_count), (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


print(f"Jumlah mobil yang berhasil terdeteksi: {car_count}")


# ==========================================
# 7. MENYIMPAN HASIL AKHIR OUTPUT
# ==========================================
# Menyimpan gambar hasil akhir tracking utama sesuai permintaan struktur direktori
cv2.imwrite("TUGAS/output/result.png", img_out)
print("Semua berkas visualisasi tahapan berhasil disimpan ke folder 'TUGAS/output/' dan 'TUGAS/output/steps/'")

# ==========================================
# 8. VISUALISASI PIPELINE LENGKAP (MATPLOTLIB)
# ==========================================
# BAGIAN A: Menampilkan Eksplorasi Awal Color Space
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1), plt.imshow(gray, cmap='gray'), plt.title('1. Grayscale Channel')
plt.subplot(1, 3, 2), plt.imshow(v_channel, cmap='gray'), plt.title('2. HSV - Channel V')
plt.subplot(1, 3, 3), plt.imshow(l_channel, cmap='gray'), plt.title('3. LAB - Channel L')
plt.tight_layout()

# BAGIAN B: Menampilkan Alur Kerja Utama dari Awal hingga Deteksi Akhir
titles = ['Original Image', 'After Blur', 'Otsu Threshold', 'Morphology Clean', 'Final Counting']
images = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
    blur, 
    thresh, 
    morph_clean, 
    cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
]

plt.figure(figsize=(15, 10))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    if i in [1, 2, 3]:  
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()