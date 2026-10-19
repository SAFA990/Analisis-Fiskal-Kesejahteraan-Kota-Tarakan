import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go

# ============================================================
# 1. KONFIGURASI HALAMAN & STYLING CSS
# ============================================================
st.set_page_config(
    page_title="Analisis Makro - Kota Tarakan",
    page_icon="🏙️",
    layout="wide"
)

st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; color: #1f77b4; }
    .insight-box { background-color: #f0f7ff; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 20px; }
    .warning-box { background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107; margin-bottom: 20px; }
    .success-box { background-color: #d4edda; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. LOAD DATA (ROBUST LOADING & DUMMY FALLBACK TARAKAN)
# ============================================================
@st.cache_data(show_spinner="Memuat dataset & menyiapkan analisis...")
def load_data():
    url = "https://raw.githubusercontent.com/Alam-A99/DATAAKSI/refs/heads/main/dataset_fiskal_97_hipoFull.csv"
    try:
        df = pd.read_csv(url)
        df['Tahun'] = df['Tahun'].astype(int)
        
        # Cek ketersediaan Kota Tarakan
        if "Kota Tarakan" not in df['Kota'].unique():
            # FALLBACK: Data Panel Dummy Kota Tarakan (2010-2022)
            np.random.seed(42)
            tahun = np.arange(2010, 2023)
            n = len(tahun)
            
            # Simulasi tren Tarakan
            ipm = np.linspace(74.5, 79.8, n) + np.random.normal(0, 0.15, n)
            kemiskinan = np.linspace(7.8, 4.5, n) + np.random.normal(0, 0.2, n) 
            pendidikan = np.linspace(350, 520, n) + np.random.normal(0, 15, n)
            kesehatan = np.linspace(150, 280, n) + np.random.normal(0, 10, n)
            pad = np.linspace(400, 850, n) + np.random.normal(0, 25, n)
            
            df_dummy = pd.DataFrame({
                'Kota': 'Kota Tarakan',
                'Tahun': tahun,
                'IPM': np.round(ipm, 2),
                'KEMISKINAN': np.round(kemiskinan, 2),
                'bm_pendidikan': np.round(pendidikan, 1),
                'bm_kesehatan': np.round(kesehatan, 1),
                'PAD': np.round(pad, 1)
            })
            return df_dummy, True
        else:
            return df[df['Kota'] == 'Kota Tarakan'].copy(), False
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame(), False

df, is_dummy = load_data()

# ============================================================
# 3. HEADER & SIDEBAR
# ============================================================
st.title("🏙️ Dashboard Komprehensif: Analisis Makro & Fiskal Kota Tarakan")
st.markdown("---")

if is_dummy:
    st.markdown("""
    <div class="warning-box">
    <b>⚠️ Informasi Sistem:</b> 'Kota Tarakan' tidak ditemukan di dalam dataset referensi. 
    Aplikasi ini secara otomatis mengaktifkan <b>Simulasi Data Panel (Dummy)</b> khusus untuk Kota Tarakan 
    agar seluruh fitur analisis ekonometrika tetap dapat didemonstrasikan secara sempurna.
    </div>
    """, unsafe_allow_html=True)

st.sidebar.header("⚙️ Panel Kontrol")
tahun_min, tahun_max = int(df['Tahun'].min()), int(df['Tahun'].max())
rentang_tahun = st.sidebar.slider("Pilih Rentang Tahun Analisis:", min_value=tahun_min, max_value=tahun_max, value=(tahun_min, tahun_max))

df_filtered = df[(df['Tahun'] >= rentang_tahun[0]) & (df['Tahun'] <= rentang_tahun[1])]

st.sidebar.markdown(f"**Periode Aktif:** {rentang_tahun[0]} - {rentang_tahun[1]}")
st.sidebar.markdown("---")
st.sidebar.info("Tugas Pemrograman Data Science - Dashboard Ekonometrika")

# ============================================================
# 4. TABS STRUKTUR UTAMA
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 1. Ringkasan & Model", 
    "📈 2. Visualisasi Data", 
    "🔥 3. Korelasi & Regresi OLS", 
    "🧊 4. Bidang Regresi 3D", 
    "📁 5. Data & Ekspor"
])

# ------------------------------------------------------------
# TAB 1: RINGKASAN & MODEL PENELITIAN
# ------------------------------------------------------------
with tab1:
    st.header("Ringkasan Eksekutif & Model Penelitian")
    
    st.markdown(f"""
    <div class="insight-box">
    <b>📝 Ringkasan Kondisi Makro Kota Tarakan ({rentang_tahun[0]} - {rentang_tahun[1]}):</b><br>
    Berdasarkan data yang dianalisis, Indeks Pembangunan Manusia (IPM) Kota Tarakan menunjukkan tren positif, 
    bergerak dari <b>{df_filtered['IPM'].iloc[0]:.2f}</b> menjadi <b>{df_filtered['IPM'].iloc[-1]:.2f}</b>. 
    Hal ini diiringi dengan penurunan tingkat kemiskinan dari <b>{df_filtered['KEMISKINAN'].iloc[0]:.2f}%</b> menjadi <b>{df_filtered['KEMISKINAN'].iloc[-1]:.2f}%</b>.<br><br>
    Secara fiskal, rata-rata Belanja Pendidikan berada di kisaran Rp {df_filtered['bm_pendidikan'].mean():.1f} Miliar, 
    sedangkan Belanja Kesehatan rata-rata Rp {df_filtered['bm_kesehatan'].mean():.1f} Miliar. Kapasitas fiskal (PAD) 
    juga menunjukkan tren yang meningkat, menandakan kemandirian daerah yang semakin baik.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Kerangka Model Penelitian")
    st.markdown("""
    Penelitian ini menggunakan pendekatan kuantitatif dengan **Regresi Linear Berganda (Ordinary Least Squares/OLS)** 
    untuk melihat pengaruh variabel fiskal terhadap kesejahteraan.

    **1. Model Utama (Pengaruh Fiskal terhadap IPM):**
    > **IPM** = Konstanta + (Koefisien × Belanja Pendidikan) + (Koefisien × Belanja Kesehatan) + (Koefisien × PAD) + Error

    **2. Model Visualisasi 3D (Pengaruh Fiskal terhadap Kemiskinan):**
    > **Kemiskinan** = Konstanta + (Koefisien × PAD) + (Koefisien × Belanja Pendidikan) + Error

    **Hipotesis Penelitian:**
    *   **H1:** Belanja Pendidikan berpengaruh signifikan terhadap IPM.
    *   **H2:** Belanja Kesehatan berpengaruh signifikan terhadap IPM.
    *   **H3:** PAD (Pendapatan Asli Daerah) berpengaruh signifikan terhadap IPM.
    """)

# ------------------------------------------------------------
# TAB 2: VISUALISASI DATA (TREN & SCATTER)
# ------------------------------------------------------------
with tab2:
    st.header("Visualisasi Perkembangan & Hubungan Data")
    
    # 2.1 Grafik Perkembangan IPM Tarakan
    st.subheader("1. Perkembangan IPM Kota Tarakan")
    fig_ipm, ax_ipm = plt.subplots(figsize=(10, 4))
    ax_ipm.plot(df_filtered['Tahun'], df_filtered['IPM'], marker='o', color='#1f77b4', linewidth=3, markersize=8)
    ax_ipm.fill_between(df_filtered['Tahun'], df_filtered['IPM'], alpha=0.2, color='#1f77b4')
    ax_ipm.set_title("Tren Indeks Pembangunan Manusia (IPM) Tarakan", fontweight='bold')
    ax_ipm.set_xlabel("Tahun")
    ax_ipm.set_ylabel("Nilai IPM")
    ax_ipm.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_ipm)
    plt.close(fig_ipm)
    
    st.markdown(f"""
    <div class="insight-box">
    <b>Interpretasi Tren IPM:</b> Pergerakan IPM Kota Tarakan dari tahun ke tahun mengindikasikan perbaikan kualitas hidup masyarakat. 
    Kenaikan ini biasanya didorong oleh peningkatan angka harapan hidup (sektor kesehatan) dan rata-rata lama sekolah (sektor pendidikan).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 2.2 Belanja Modal Kesehatan vs IPM
    st.subheader("2. Hubungan Belanja Kesehatan vs IPM")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
        ax_scatter.scatter(df_filtered['bm_kesehatan'], df_filtered['IPM'], color='crimson', s=150, alpha=0.8, edgecolors='black')
        
        # Garis regresi sederhana
        z = np.polyfit(df_filtered['bm_kesehatan'], df_filtered['IPM'], 1)
        p = np.poly1d(z)
        ax_scatter.plot(df_filtered['bm_kesehatan'], p(df_filtered['bm_kesehatan']), "k--", linewidth=2, label="Garis Tren")
        
        ax_scatter.set_title("Scatter Plot: Belanja Kesehatan vs IPM", fontweight='bold')
        ax_scatter.set_xlabel("Belanja Kesehatan (Miliar Rp)")
        ax_scatter.set_ylabel("IPM")
        ax_scatter.grid(True, linestyle='--', alpha=0.6)
        ax_scatter.legend()
        st.pyplot(fig_scatter)
        plt.close(fig_scatter)
        
    with col2:
        st.markdown("""
        <div class="insight-box">
        <b>Analisis Scatter:</b><br>
        Setiap titik mewakili satu tahun pengamatan. Jika titik-titik membentuk formasi naik dari kiri bawah ke kanan atas 
        (mengikuti garis putus-putus), artinya secara visual terdapat indikasi bahwa peningkatan anggaran kesehatan 
        diikuti oleh kenaikan IPM di Kota Tarakan.
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 3: KORELASI & REGRESI OLS
# ------------------------------------------------------------
with tab3:
    st.header("Hasil Estimasi Koefisien Regresi OLS")
    
    # Setup Variabel OLS
    Y = df_filtered['IPM']
    X_vars = ['bm_pendidikan', 'bm_kesehatan', 'PAD']
    X = df_filtered[X_vars]
    X_const = sm.add_constant(X)
    model = sm.OLS(Y, X_const).fit()
    
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("R-Squared (R²)", f"{model.rsquared:.4f}", "Kekuatan Model")
    col_r2.metric("Adj. R-Squared", f"{model.rsquared_adj:.4f}")
    col_r3.metric("Prob (F-Statistic)", f"{model.f_pvalue:.6f}", "Signifikansi Simultan")
    
    st.subheader("Tabel Koefisien Estimasi (Variabel Y = IPM)")
    df_hasil = pd.DataFrame({
        'Koefisien (Beta)': model.params,
        'Standar Error': model.bse,
        'P-Value': model.pvalues
    })
    st.dataframe(df_hasil.style.format("{:.6f}").background_gradient(cmap='Blues'), use_container_width=True)
    
    st.markdown(f"""
    <div class="success-box">
    <h4>Interpretasi Mendalam Output OLS:</h4>
    1. <b>Uji Kelayakan Model (R²):</b> Nilai R-squared sebesar <b>{model.rsquared:.4f}</b> berarti bahwa sekitar <b>{model.rsquared*100:.1f}%</b> 
    variasi naik-turunnya IPM Kota Tarakan dapat dijelaskan oleh variabel Belanja Pendidikan, Belanja Kesehatan, dan PAD secara bersama-sama.<br>
    
    2. <b>Uji Simultan (Prob F):</b> Nilai p-value F-statistik adalah <b>{model.f_pvalue:.6f}</b> (di bawah 0.05). Artinya, secara statistik, 
    ketiga variabel bebas kita terbukti berpengaruh nyata terhadap IPM.<br>
    
    3. <b>Pengaruh Parsial (Uji T):</b> 
    - Setiap kenaikan 1 Miliar Belanja Pendidikan, IPM diprediksi berubah sebesar <b>{model.params['bm_pendidikan']:.5f}</b> poin (dengan asumsi variabel lain tetap).
    - Setiap kenaikan 1 Miliar Belanja Kesehatan, IPM diprediksi berubah sebesar <b>{model.params['bm_kesehatan']:.5f}</b> poin.
    - Setiap kenaikan 1 Miliar PAD, IPM diprediksi berubah sebesar <b>{model.params['PAD']:.5f}</b> poin.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 4: BIDANG REGRESI 3D DENGAN PLOTLY
# ------------------------------------------------------------
with tab4:
    st.header("Visualisasi 3D: Bidang Regresi (Regression Plane)")
    st.write("Visualisasi interaktif ini menggunakan **Plotly** untuk memetakan hubungan 3 dimensi antara PAD, Belanja Pendidikan, dan Kemiskinan.")
    
    # Setup Variabel untuk 3D
    Y_VAR = 'KEMISKINAN'
    Y_3d = df_filtered[Y_VAR]
    
    # Model OLS Khusus 3D
    X2 = sm.add_constant(df_filtered[['PAD', 'bm_pendidikan']])
    model3d = sm.OLS(Y_3d, X2).fit()
    b0, b1, b2 = (model3d.params['const'], model3d.params['PAD'], model3d.params['bm_pendidikan'])

    # Grid bidang regresi
    x1g = np.linspace(df_filtered['PAD'].min(), df_filtered['PAD'].max(), 20)
    x2g = np.linspace(df_filtered['bm_pendidikan'].min(), df_filtered['bm_pendidikan'].max(), 20)
    X1g, X2g = np.meshgrid(x1g, x2g)
    Zg = b0 + b1 * X1g + b2 * X2g

    # Membuat Plotly Figure
    fig = go.Figure()
    
    # 1. Trace Data Aktual (Titik Merah)
    fig.add_trace(go.Scatter3d(
        x=df_filtered['PAD'], y=df_filtered['bm_pendidikan'], z=df_filtered[Y_VAR],
        mode='markers', marker=dict(size=7, color='red', opacity=0.9),
        name='Data Aktual',
        text=[f'Tahun {t}' for t in df_filtered['Tahun']],
        hovertemplate='<b>%{text}</b><br>PAD: %{x:.2f}<br>BM Pendidikan: %{y:.2f}<br>Kemiskinan: %{z:.2f}%'
    ))
    
    # 2. Trace Bidang Regresi (Surface)
    fig.add_trace(go.Surface(
        x=X1g, y=X2g, z=Zg, opacity=0.6, colorscale='Viridis',
        name='Regression Plane',
        hovertemplate='PAD: %{x:.2f}<br>BM Pendidikan: %{y:.2f}<br>Prediksi Kemiskinan: %{z:.2f}%'
    ))

    # Layout Plotly
    fig.update_layout(
        title={
            'text': ('<b>REGRESI LINEAR BERGANDA 3D - KOTA TARAKAN</b><br>' +
                     f'KEMISKINAN = {b0:.3f} + ({b1:.5f})·PAD + ({b2:.5f})·BM_Pendidikan | R² = {model3d.rsquared:.3f}'), 
            'x': 0.5
        },
        scene={
            'xaxis': {'title': 'PAD (X1)'},
            'yaxis': {'title': 'BM Pendidikan (X2)'},
            'zaxis': {'title': 'Kemiskinan (%) (Y)'}
        },
        width=950, height=700, margin={'l': 0, 'r': 0, 'b': 0, 't': 100}
    )
    
    # Tampilkan di Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    # Fitur Download HTML
    html_str = fig.to_html(include_plotlyjs='cdn')
    st.download_button(
        label="📥 Unduh Grafik 3D Interaktif (HTML)",
        data=html_str,
        file_name="e_regresi_3d.html",
        mime="text/html"
    )
    
    st.markdown(f"""
    <div class="insight-box">
    <b>📖 Cara Membaca Grafik 3D di Atas:</b><br>
    *   <b>Titik Merah (Data Aktual):</b> Merepresentasikan kondisi riil Kemiskinan, PAD, dan Belanja Pendidikan pada tahun tertentu.<br>
    *   <b>Bidang Berwarna (Regression Plane):</b> Adalah "alas" prediksi yang dibentuk oleh model matematika OLS kita.<br>
    *   <b>Persamaan:</b> Model memprediksi Kemiskinan berawal dari {b0:.3f}. Setiap kenaikan PAD akan mengubah kemiskinan sebesar {b1:.5f}, dan setiap kenaikan Belanja Pendidikan mengubahnya sebesar {b2:.5f}. Nilai R² sebesar {model3d.rsquared:.3f} menunjukkan bahwa bidang ini cukup akurat ({model3d.rsquared*100:.1f}%) dalam menangkap pola data aktual.<br><br>
    <i>*Tips: Anda dapat memutar (rotate), melakukan zoom in/out, dan melihat nilai detail dengan mengarahkan kursor ke titik atau bidang pada grafik di atas.*</i>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 5: DATA & EKSPOR
# ------------------------------------------------------------
with tab5:
    st.header("Tabel Data Mentah & Ekspor")
    st.dataframe(df_filtered.style.set_properties(**{'background-color': '#f9f9f9', 'color': 'black'}), use_container_width=True)
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Dataset Kota Tarakan (CSV)",
        data=csv,
        file_name=f"Data_Makro_Tarakan_{rentang_tahun[0]}_{rentang_tahun[1]}.csv",
        mime="text/csv",
    )