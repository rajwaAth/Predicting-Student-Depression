# Laporan Proyek Machine Learning

## Domain Proyek: Mental Health Analytics dalam Lingkungan Pendidikan

Kesehatan mental merupakan aspek penting dalam mendukung keberhasilan akademik dan kehidupan sosial siswa maupun mahasiswa. Dalam beberapa tahun terakhir, terdapat peningkatan signifikan terhadap kasus gangguan mental, khususnya depresi, di kalangan pelajar. Organisasi Kesehatan Dunia (WHO) melaporkan bahwa lebih dari 264 juta orang di dunia menderita depresi, dan sebagian besar kasus dimulai pada usia remaja dan dewasa muda – usia yang identik dengan masa sekolah dan kuliah.

Dalam konteks pendidikan tinggi, tekanan akademik, ekspektasi sosial, kesepian, gaya hidup yang tidak seimbang, dan masalah finansial merupakan pemicu utama gangguan psikologis. Menurut sebuah studi oleh American College Health Association (2021), lebih dari 40% mahasiswa di Amerika Serikat melaporkan mengalami gejala depresi dalam satu tahun terakhir. Hal serupa juga terjadi di Indonesia, sebagaimana diungkapkan oleh Kementerian Kesehatan RI, bahwa sekitar 6,1% remaja usia 15–24 tahun mengalami gangguan mental emosional, dan angka ini cenderung meningkat setiap tahunnya.

Meningkatnya prevalensi depresi pada pelajar menuntut adanya solusi berbasis teknologi untuk mengidentifikasi risiko secara dini. Pendekatan konvensional seperti konseling tatap muka atau survei manual sering kali bersifat reaktif, tidak efisien, dan sulit menjangkau semua siswa secara merata. Oleh karena itu, diperlukan pendekatan yang lebih sistematis dan prediktif dengan memanfaatkan machine learning dan analitik data.

**Referensi:**
- [World Health Organization. (2020). Depression](https://www.who.int/news-room/fact-sheets/detail/depression)
- [American College Health Association. (2021). ACHA-National College Health Assessment III](https://www.sjsu.edu/wellness/docs/ncha-spring-2021-executive-summary.pdf)
- [Kementerian Kesehatan RI. (2018). Laporan Nasional Riskesdas 2018](https://repository.badankebijakan.kemkes.go.id/id/eprint/3514/1/Laporan%20Riskesdas%202018%20Nasional.pdf)

## Business Understanding

### Problem Statements

1. Belum adanya sistem prediktif yang dapat mengidentifikasi siswa yang berisiko mengalami depresi
2. Kurangnya pemahaman terkait faktor-faktor utama yang memengaruhi kondisi mental siswa
3. Kebutuhan institusi pendidikan terhadap pendekatan berbasis data dalam pengambilan keputusan terkait kebijakan kesehatan mental masih belum terpenuhi

### Goals

1. Mengembangkan model prediktif untuk mengidentifikasi status depresi siswa berdasarkan data demografis, akademik, dan gaya hidup
2. Menentukan fitur atau faktor yang paling signifikan dalam memengaruhi risiko depresi
3. Memberikan rekomendasi berbasis data kepada institusi pendidikan untuk membantu dalam perancangan program intervensi dini kesehatan mental

### Solution Statements

1. Melakukan Exploratory Data Analysis (EDA) untuk memahami pola belajar, kebiasaan istirahat, gaya hidup, dan pengalaman siswa yang berkaitan dengan risiko depresi
2. Menerapkan dan membandingkan beberapa algoritma klasifikasi seperti `Logistic Regression`, `Decision Tree`, dan `XGBoost` untuk menemukan model prediktif terbaik
3. Mengevaluasi performa setiap model menggunakan metrik **F1-Score** dan **ROC AUC** guna memilih model yang paling seimbang antara precision dan recall

## Data Understanding

Dataset yang digunakan merupakan kumpulan data terkait kondisi kesehatan mental siswa, khususnya berfokus pada identifikasi kemungkinan depresi. Dataset ini bersifat publik dan dapat digunakan untuk penelitian edukasi maupun eksperimen machine learning.

- Jumlah data: 27.901 baris
- Jumlah fitur: 18 kolom (termasuk target)
- Tipe data: Gabungan antara 9 data numerik dan 9 data kategorikal
- Sumber data: [Student Depression Dataset on Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)

### Variabel-variabel pada Student Depression Dataset adalah sebagai berikut:

| No. | Nama Kolom | Deskripsi |
|-----|------------|-----------|
| 1 | `id` | ID unik untuk setiap responden |
| 2 | `Gender` | Jenis kelamin responden (`Male` atau `Female`) |
| 3 | `Age` | Usia responden dalam tahun |
| 4 | `City` | Kota atau lokasi tempat tinggal responden |
| 5 | `Profession` | Pekerjaan atau profesi responden (contoh: Mahasiswa, Engineer, dll) |
| 6 | `Academic Pressure` | Tingkat tekanan akademik yang dirasakan (skala numerik) |
| 7 | `Work Pressure` | Tingkat tekanan dari pekerjaan atau tugas akademik |
| 8 | `CGPA` | Indeks Prestasi Kumulatif (IPK) responden |
| 9 | `Study Satisfaction` | Tingkat kepuasan terhadap pengalaman belajar |
| 10 | `Job Satisfaction` | Tingkat kepuasan terhadap pekerjaan (jika ada) |
| 11 | `Sleep Duration` | Durasi tidur rata-rata (kategori: contoh: '5-6 hours', 'Less than 5 hours') |
| 12 | `Dietary Habits` | Pola makan responden (`Healthy`, `Moderate`, `Unhealthy`, atau `Others`) |
| 13 | `Degree` | Gelar pendidikan yang sedang ditempuh atau telah diperoleh |
| 14 | `Have you ever had suicidal thoughts ?` | Pernahkah responden memiliki pikiran untuk bunuh diri (`Yes` atau `No`) |
| 15 | `Work/Study Hours` | Rata-rata jam kerja atau belajar per hari |
| 16 | `Financial Stress` | Tingkat stres keuangan yang dirasakan (skala 1.0 – 5.0) |
| 17 | `Family History of Mental Illness` | Riwayat keluarga dengan gangguan mental (`Yes` atau `No`) |
| 18 | `Depression` | Variabel target: status depresi (`1` = Ya, `0` = Tidak) |

Dari hasil analisis awal deskripsi data, ditemukan bahwa kolom `id` hanya berisi nilai unik untuk setiap pelanggan dan **tidak memberikan kontribusi prediktif**, sehingga kolom ini nantinya akan **dihapus** kerena tidak dibutuhkan untuk analisis lebih lanjut. Selain itu terdapat pula kolom `Financial Stress` yang bertipe data **object** yang seharusnya bertipe data **float**, sehingga perlu dilakukan konversi tipe data pada tahap selanjutnya.

### Distribusi Data Target (Depression)

<br>
<image src='image/distribusi_target.png' width= 500/>
<br>
Distribusi pada data target (Depression) sedikit **imbalance** (tidak seimbang), yang mungkin dapat menyebabkan model cenderung memprediksi kelas mayoritas. Oleh karena itu, untuk mengantisipasi permasalahan ini, akan dilakukan percobaan menggunakan teknik **oversampling** pada **data train** (*setelah proses pembagian data menjadi data latih dan data uji*).

#### Analisis Univariat

**1. Distribusi Data Kategorikal**
<br>
<image src='image/distribusi_data_kategorik.png' width= 500/>
<br>

Berdasarkan distribusi nilai pada kolom-kolom kategorikal, ditemukan bahwa kolom `City` dan `Profession` memiliki beberapa kategori dengan jumlah data yang sangat sedikit (dominan pada satu kategori saja). Selain itu, kolom City juga memiliki terlalu banyak kategori, yang ***dapat menyebabkan curse of dimensionality***. Oleh karena itu, kedua kolom tersebut akan dihapus dari dataset.

Selain itu, pada kolom `Sleep Duration`, `Dietary Habits`, dan `Degree`, terdapat kategori bernilai "Others" yang tidak merepresentasikan informasi yang jelas serta jumlahnya sangat sedikit. Maka dari itu, baris data yang memiliki nilai "Others" pada fitur-fitur tersebut akan dihapus dari dataset.

**2. Distribusi Data Numerik**
<br>
<image src='image/distribusi_data_numerik.png' width= 500/>
<image src='image/boxplot_data_numerik.png' width= 500/>
<br> 
Kolom `Work Pressure` dan `Job Satisfaction` juga menunjukkan dominasi pada satu nilai tertentu, sehingga tidak memberikan variasi yang signifikan untuk analisis. Oleh karena itu, kedua kolom tersebut akan dihapus dari dataset.

Sementara itu, kolom `Age` dan `CGPA` teridentifikasi **memiliki nilai outlier** yang dapat memengaruhi hasil analisis. Outlier pada kedua kolom tersebut akan dihapus pada tahap praproses selanjutnya.

#### Analisis Multivariat

**1. Distribusi Jenis Kelamin dan Pengaruhnya terhadap Status Depresi**
<br>
<image src='image/gender.png' width= 500/>
<br>
Baik pria maupun wanita memiliki proporsi yang hampir sama dalam hal mengalami depresi, dengan sekitar 58% dari masing-masing gender tercatat mengalami depresi. Persentase pria yang mengalami depresi sedikit lebih tinggi (58,62%) dibanding wanita (58,47%). Ini mengindikasikan bahwa dalam data ini, status depresi tidak terlalu dipengaruhi oleh perbedaan gender.

**2. Pengaruh Pemikiran Bunuh Diri Terhadap Status Depresi**
<br>
<image src='image/pemikiran_bunuh_diri.png' width= 500/>
<br>
Seperti yang dapat diduga, data menunjukkan bahwa responden yang memiliki pemikiran untuk melakukan bunuh diri memiliki kemungkinan jauh lebih besar untuk mengalami depresi.

**3. Pengaruh Tekanan Akademik Terhadap Status Depresi**
<br>
<image src='image/tekanan_akademik.png' width= 500/>
<br>
Tekanan akademik yang tinggi dapat menjadi salah satu faktor yang meningkatkan risiko seseorang mengalami depresi. Semakin besar beban dan stres yang dirasakan, semakin tinggi pula kemungkinan individu mengalami gangguan kesehatan mental seperti depresi.

**4. Umur Pelajar Dengan Kemungkinan Status Depresi**
<br>
<image src='image/umur.png' width= 500/>
<br> 
Grafik menunjukkan bahwa responden dengan status depresi cenderung berusia lebih muda, dengan rata-rata usia 24 tahun, dibandingkan yang tidak depresi dengan rata-rata 27 tahun. Hal ini mengindikasikan bahwa depresi lebih banyak dialami oleh kelompok usia muda.

**5. Jam Belajar/Kerja dan Status Depresi**
<br>
<image src='image/waktu_belajar.png' width= 500/>
<br> 
Responden dengan depresi memiliki rata-rata jam kerja/belajar lebih tinggi (7,81 jam) dibandingkan yang tidak depresi (6,24 jam). Ini mengindikasikan bahwa semakin banyak jam kerja/belajar, potensi mengalami depresi cenderung meningkat.

## Data Preparation

Pada tahap ini dilakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Beberapa tahap persiapan data yang dilakukan adalah:

### 1. Menghapus Kolom & Kategori Values yang Tidak Penting

Pertama akan dilakukakn penghapusan pada kolom - kolom yang tidak akan digunakan lebih lanjut seperti kolom `Id`, `City`, `Profession`, `Job Satisfaction`, `Work Pressure` dan juga mengubah kategori nilai pada kolom `Financial Stress` yang bertipe data **object** menjadi **float**. dan tidak lupa untuk menghapus baris data yang memiliki nilai "Others" pada kolom `Sleep Duration`, `Dietary Habits`, dan `Degree` karana nilai "Others" tidak merepresentasikan informasi yang jelas.
<br>
<image src='image/menghapus_kolom_id.png' width= 500/>
<image src='image/mengubah_kategori_nilai.png' width= 500/>
<br> 

### 2. Menangani Missing Values

Pada dataset terdapat missing value pada kolom `Financial Stress` sebanyak 3 data. Dikarenakan jumlahnya yang sedikit dan untuk menjaga keaslian data, maka diputuskan untuk menghapus baris-baris tersebut dari dataset.
<img src='image/null_value.png' align="center"><br>

### 3. Menghapus Outlier Values

Untuk menangani outlier, dilakukan penghapusan outlier pada kolom `Age` dan `CGPA` menggunakan metode IQR (Interquartile Range). Metode ini digunakan untuk menghilangkan nilai-nilai yang berada di luar batas bawah dan batas atas yang ditentukan, sehingga data menjadi lebih bersih dan representatif.
<img src='image/outlier.png' align="center"><br>

### 4. Encoding Fitur Kategori

Pada bagian ini, dilakukan transformasi data kategori (yang berbentuk teks atau label) menjadi format numerik agar dapat diproses oleh algoritma machine learning. Encoding fitur kategorikal dilakukan dalam 2 bagian:

1. **Label Encoding**: mengonversi nilai kategori menjadi angka integer (`0` dan `1`) untuk variabel biner seperti:
   - `Gender`
   - `Have you ever had suicidal thoughts ?`
   - `Family History of Mental Illness`

2. **One Hot Encoding**: mengubah setiap kategori menjadi kolom biner terpisah untuk data tidak terurut seperti:
   - `Sleep Duration`
   - `Dietary Habits`
   - `Degree`

<img src="image/encoding.png" align="center"><br>

### 5. Train-Test-Split

Data dibagi dengan proporsi 80:20, dimana 80% digunakan untuk training model dan 20% digunakan untuk testing model, untuk memastikan evaluasi yang objektif terhadap performa model.
<img src="image/spliting_data.png" align="center"><br>

### 6. Transformasi Values

Dilakukan scaling value dengan MinMaxScaler untuk menyamaratakan skala dari setiap fitur, sehingga tidak ada fitur yang mendominasi karena memiliki skala nilai yang lebih besar.
<img src="image/transformation_value.png" align="center"><br>

### 7. Menangani Data Imbalance

SMOTE (Synthetic Minority Over-sampling Technique) digunakan untuk mengatasi ketidakseimbangan kelas pada data latih. Pengujian juga dilakukan pada data tanpa SMOTE untuk membandingkan akurasi dan menilai efektivitas metode tersebut.
<img src="image/imbalance_data.png" align="center"><br>

## Modeling

Pada project kali ini akan dilakukan percobaan terhadap beberapa algoritma machine learning yaitu:

### 1. Logistic Regression
Logistic Regression adalah algoritma machine learning yang digunakan untuk klasifikasi, terutama klasifikasi biner. Algoritma ini memodelkan probabilitas suatu data masuk ke kelas tertentu menggunakan fungsi logistik (sigmoid). Logistic Regression mencoba menemukan garis pemisah (decision boundary) linear antara kelas.
**Kelebihan:**
* Sederhana dan cepat untuk dilatih.
* Mudah diinterpretasikan melalui nilai koefisien fitur.
* Cocok untuk kasus klasifikasi biner.

**Kekurangan:**
* Sensitif terhadap multikolinearitas.
* Kurang efektif jika banyak outlier.

### 2. Decision Tree
Decision Tree adalah algoritma klasifikasi yang bekerja dengan membagi dataset berdasarkan fitur menjadi cabang-cabang seperti struktur pohon. Setiap node dalam pohon mewakili fitur, dan setiap daun mewakili kelas. Algoritma ini menggunakan pemisahan berdasarkan kriteria tertentu (seperti Gini atau Entropy) untuk membuat keputusan.

**Kelebihan:**
* Bisa menangani data numerik dan kategorikal.
* Tidak memerlukan normalisasi atau scaling data.
* Dapat menangkap hubungan non-linear antara fitur.

**Kekurangan:**
* Rentan terhadap overfitting, terutama pada data kompleks.
* Sensitif terhadap perubahan kecil pada data.
* Bisa membuat model yang terlalu dalam atau kompleks.

### 3. XGBoost
XGBoost adalah algoritma boosting yang sangat efisien dan akurat. Algoritma ini bekerja dengan membangun banyak pohon keputusan secara bertahap, di mana setiap pohon mencoba memperbaiki kesalahan dari pohon sebelumnya. XGBoost dilengkapi dengan teknik regularisasi dan optimisasi untuk mencegah overfitting dan meningkatkan performa.

**Kelebihan:**
* Mampu menangani missing value secara otomatis.
* Dilengkapi dengan regularisasi untuk menghindari overfitting.
* Cepat dalam pelatihan dan prediksi.

**Kekurangan:**
* Lebih kompleks dan sulit untuk dipahami secara menyeluruh.
* Membutuhkan waktu tuning hyperparameter yang tidak sedikit.
* Mengonsumsi memori dan waktu lebih banyak dibanding model sederhana.

## Evaluation

Untuk mengevaluasi kinerja model dalam mendeteksi risiko depresi pada mahasiswa, digunakan metrik **F1 Score**. Pemilihan metrik ini didasarkan pada karakteristik masalah yang memiliki distribusi kelas tidak seimbang dan dampak serius apabila terjadi kesalahan klasifikasi.

### F1 Score

**F1 Score** merupakan metrik yang menggabungkan **Precision** dan **Recall** dalam satu nilai harmonis, dengan rumus:

<img src="image/f1score_image.png" align="center"><br>

di mana:
- **Precision**: Persentase prediksi positif yang benar-benar positif.  
<img src="image/precision_formulas.png" align="center"><br>
- **Recall**: Persentase kasus positif yang berhasil diprediksi sebagai positif.  
<img src="image/recall_formulas.png" align="center"><br>

Penggunaan F1 Score sangat sesuai untuk situasi di mana keseimbangan antara **False Positive** dan **False Negative** penting untuk dipertahankan, seperti dalam kasus deteksi risiko depresi pada mahasiswa. Berdasarkan hasil evaluasi, model **Logistic Regression** memperoleh nilai F1 Score tertinggi sebesar **0,87**, menunjukkan bahwa model ini mampu menjaga keseimbangan terbaik antara ketepatan dalam mendeteksi depresi dan kepekaan dalam menjangkau kasus positif dibandingkan model lainnya.

<img src="image/perbandingan_score.png" align="center"><br>

## Referensi
1. World Health Organization. (2020). Depression. Retrieved from: https://www.who.int/news-room/fact-sheets/detail/depression
2. American College Health Association. (2021). ACHA-National College Health Assessment III: Undergraduate Student Reference Group Executive Summary Spring 2021.Retrieved from: https://www.sjsu.edu/wellness/docs/ncha-spring-2021-executive-summary.pdf
3. Kementerian Kesehatan RI. (2018). Laporan Nasional Riskesdas 2018. Badan Penelitian dan Pengembangan Kesehatan, Kemenkes RI. Retrieved from: https://repository.badankebijakan.kemkes.go.id/id/eprint/3514/1/Laporan%20Riskesdas%202018%20Nasional.pdf
4. https://medium.com/@andimrinaldisaputraa/memahami-dan-menerapkan-matriks-evaluasi-roc-auc-dalam-machine-learning-4468e5fcb9a