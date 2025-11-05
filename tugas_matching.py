import streamlit as st
import cv2
import numpy as np
from scipy import stats
from PIL import Image

# --- Fungsi Bantuan ---

def load_image_from_uploader(uploader):
    """Membaca file yang diupload menjadi format OpenCV BGR."""
    if uploader is not None:
        file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img_bgr
    return None

def get_image_stats(image_bgr):
    """Menghitung semua statistik yang diperlukan untuk satu gambar."""
    stats_dict = {}
    
    # Konversi ke Grayscale untuk Skewness, Kurtosis, Entropy, Chi-Square
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    pixels_flat = img_gray.flatten()
    
    # 1. Skewness
    stats_dict['Skewness'] = stats.skew(pixels_flat)
    
    # 2. Kurtosis
    stats_dict['Kurtosis'] = stats.kurtosis(pixels_flat)
    
    # 3. Entropy
    # Hitung histogram dulu untuk entropy
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist_prob = hist.ravel() / hist.sum() # Normalisasi jadi probabilitas
    stats_dict['Entropy'] = stats.entropy(hist_prob)
    
    # 4. Pearson Correlation (antara channel R dan G)
    r_channel, g_channel, _ = cv2.split(image_bgr)
    corr_rg, _ = stats.pearsonr(r_channel.flatten(), g_channel.flatten())
    stats_dict['Pearson (R vs G)'] = corr_rg
    
    # 5. Chi-Square (Goodness-of-fit vs distribusi uniform)
    hist_observed = hist.flatten()
    total_pixels = pixels_flat.shape[0]
    # Buat distribusi "expected" (seragam)
    expected_freq = total_pixels / 256.0
    hist_expected = np.full_like(hist_observed, expected_freq)
    
    # Menghindari pembagian nol jika ada bin kosong di expected (meski di sini tidak)
    # dan memastikan tidak ada frekuensi 0 di observed
    hist_observed[hist_observed == 0] = 1 
    
    chi_stat, p_val = stats.chisquare(f_obs=hist_observed, f_exp=hist_expected)
    stats_dict['Chi-Square (vs Uniform)'] = chi_stat
    
    return stats_dict

def compare_histograms(img_bgr1, img_bgr2):
    """Menghitung nilai matching histogram antara dua gambar."""
    
    # Ubah ke Grayscale
    img_gray1 = cv2.cvtColor(img_bgr1, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2GRAY)
    
    # Hitung histogram untuk kedua gambar
    hist1 = cv2.calcHist([img_gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img_gray2], [0], None, [256], [0, 256])
    
    # Normalisasi histogram agar perbandingan adil
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Bandingkan menggunakan metode yang diminta
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    
    return {
        "Korelasi Histogram": correlation,
        "Chi-Square Histogram": chi_square
    }

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide", page_title="Analisis Statistik Citra")
st.title("Tugas Analisis & Matching Citra")

# --- Panel Upload ---
st.sidebar.header("Panel Input")
uploader1 = st.sidebar.file_uploader("Upload Citra Pertama", type=["jpg", "png", "jpeg"])
uploader2 = st.sidebar.file_uploader("Upload Citra Kedua (untuk matching)", type=["jpg", "png", "jpeg"])

# --- Panel Utama ---
col1, col2 = st.columns([1, 1])

img1_bgr = load_image_from_uploader(uploader1)
img2_bgr = load_image_from_uploader(uploader2)

if img1_bgr is not None:
    # Konversi BGR ke RGB untuk ditampilkan di Streamlit
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    
    with col1:
        st.header("Analisis Citra Pertama")
        st.image(img1_rgb, caption="Citra 1", use_container_width=True)
    
    with col2:
        st.header("Hasil Statistik Citra 1")
        try:
            image_stats = get_image_stats(img1_bgr)
            st.metric(label="Skewness", value=f"{image_stats['Skewness']:.4f}")
            st.metric(label="Kurtosis", value=f"{image_stats['Kurtosis']:.4f}")
            st.metric(label="Entropy", value=f"{image_stats['Entropy']:.4f}")
            st.metric(label="Pearson Correlation (R vs G)", value=f"{image_stats['Pearson (R vs G)']:.4f}")
            st.metric(label="Chi-Square (vs Uniform Dist.)", value=f"{image_stats['Chi-Square (vs Uniform)']:.2f}")

        except Exception as e:
            st.error(f"Gagal menghitung statistik: {e}")

else:
    st.info("Silakan upload Citra Pertama untuk melihat analisis statistik.")

st.divider()

if img1_bgr is not None and img2_bgr is not None:
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    
    with col1:
        st.header("Analisis Citra Kedua")
        st.image(img2_rgb, caption="Citra 2", use_container_width=True)
        
    with col2:
        st.header("Hasil Matching (Citra 1 vs Citra 2)")
        st.info("Perbandingan ini didasarkan pada kemiripan histogram (distribusi kecerahan) kedua gambar.")
        
        try:
            matching_values = compare_histograms(img1_bgr, img2_bgr)
            st.metric(label="Korelasi Histogram", 
                      value=f"{matching_values['Korelasi Histogram']:.4f}",
                      help="Semakin dekat ke 1, semakin mirip. 1 = identik.")
            
            st.metric(label="Chi-Square Histogram", 
                      value=f"{matching_values['Chi-Square Histogram']:.2f}",
                      help="Semakin dekat ke 0, semakin mirip. 0 = identik.",
                      delta_color="inverse") # Agar nilai rendah terlihat bagus
                      
        except Exception as e:
            st.error(f"Gagal melakukan matching: {e}")

elif img1_bgr is not None and img2_bgr is None:
    st.info("Upload Citra Kedua untuk melihat hasil perbandingan/matching.")
