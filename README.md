# Laporan Proyek Machine Learning - Muhammad Ilham Malik

## Project Overview

Industri animasi Jepang berkembang dengan sangat cepat dan telah menghasilkan animasi film yang sangat banyak [1]. Animasi ini biasa disebut dengan istilah "anime" juga menarik perhatian dari berbagai kalangan. Hal ini menyebabkan pertumbuhan industri animasi Jepang yang sangat cepat, bahkan pada tahun 2018, market untuk industri ini sudah melebihi 1 triliun yen [2]. Namun, dengan tingginya variasi jenis animasi film Jepang tersebut akan menyebabkan sebuah masalah yaitu akan menyulitkan user untuk menemukan sebuah _anime_ yang menyamai referensi mereka.

Masalah tersebut dapat diselesaikan dengan memanfaatkan sistem rekomendasi menggunakan pendekatan _Collaborative Filtering_ [3]. Sistem rekomendasi ini dapat memberikan keuntungan baik kepada pengguna dan industri animasi Jepang. Pengguna akan dapat menemukan sebuah _anime_ yang cocok dengan preferensi mereka secara efisien dan akurat. Industri animasi akan mendapat keuntungan lebih karena target genre animasi yang mereka buat akan dapat lebih mudah ditemukan melalui sistem rekomendasi tersebut. Selain itu, dengan diselesaikannya masalah ini akan ditemukan model yang paling optimal pada domain ini.

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah:
- Pengguna akan kesulitan menemukan _anime_ yang sesuai dengan preferensi mereka dengan jumlah _anime_ yang sangat banyak dan juga beragam.
- Penulis belum mengetahui model yang optimal dalam merekomendasikan top-K _anime_.

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Memudahkan pengguna menemukan _anime_ yang sesuai dengan preferensi mereka secara efisien dan akurat.
- Menemukan sebuah model yang optimal pada dataset ini yang ditunjukan dengan evaluasi metrik RMSE.

### Solution statements
- Membangun sistem rekomendasi menggunakan pendekatan _Collaborative Filtering_ dengan algoritma _Deep Learning_ RecommenderNet [5] dan Matriks Faktorisasi menggunakan metode _Deep Learning_ yang bernama _Probabilistic Matrix Factorization_ [4].
- Model paling optimal akan diketahui setelah dilakukan analisis dan visualisasi hasil evaluasi dari setiap model.

## Data Understanding
Dataset _anime_ yang digunakan merupakan dataset yang ditemukan pada platform _public dataset_ Kaggle. Dataset ini memiliki informasi data terkait preferensi pengguna sebanyak 73.516 penilaian pengguna terhadap 12.294 film _anime_ dengan total keseluruhan sebanyak 7.813.737 baris data. Dataset ini diperoleh dari laman _Website_ myanimelist.net dengan memanfaatkan API pada _website_ tersebut. Informasi lebih lanjut terkait dataset yang digunakan dapat diakses pada [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database?resource=download)

Pada file Anime.csv, terdapat variabel-variabel sebagai berikut:
- anime_id : Merupakan _identifier_ unik film anime.
- name : Merupakan judul lengkap _anime._
- genre : Merupakan genre _anime_.
- type : Merupakan jenis _anime_ (OVA, TV, Movie).
- episodes : Merupakan jumlah episode _anime_ tersebut.
- rating : Merupakan rerata nilai yang didapatkan _anime_ tersebut.
- members : Jumlah anggota yang berada pada "grup" _anime_ tersebut.

Pada file Rating.csv, terdapat variabel-variabel sebagai berikut:
- user_id : Merupakan _identifier_ yang dibuat secara random sebagai user_id.
- anime_id : Merupakan _identifier_ anime yang diberikan nilai oleh pengguna.
- rating : Merupakan nilai yang diberikan oleh pengguna, memiliki skala hingga 10 serta nilai -1 jika user menonton dan tidak memberikan rating.

### Exploratory Data Analysis: Data Description
Pada bagian ini, akan digunakan method-method yang berasal dari library Pandas DataFrame untuk memahami skema data. Method-method tersebut meliputi `head()` dan `info()`. Secara singkat method `head()` digunakan untuk melihat beberapa baris data. Selain itu, method `info()` digunakan untuk memeriksa tipe data dan memeriksa keberadaan _missing value_.

Untuk membaca dataset, dapat menggunakan `read_csv()` method untuk membaca file yang memiliki format CSV.

```
anime_df = pd.read_csv("dataset/anime.csv")
rating_df = pd.read_csv("dataset/rating.csv")
```

Berikut adalah merupakan kode yang digunakan untuk melihat 10 data teratas dari dataset. Kode di bawah ini menggunakan head() method yang ada pada objek DataFrame.

```
anime_df.head(10)
```

![anime_df_head](https://github.com/ilhamMalik51/DicodingAppliedML/blob/09e3b2d6195b79a2ecc378130fd2465dbd046c44/Proyek-Kedua/asset/df_head_anime_df.JPG)

Seperti yang sudah disinggung sebelumnya, dataset ini terdiri dari 7 fitur. Terdapat fitur yang perlu diperhatikan, contohnya fitur **rating**. Berdasarkan metadata fitur **rating** ini merupakan rerata rating yang diterima oleh film tersebut. Hal ini penting diketahui karena apabila terdapat sebuah data yang tidak memiliki rating, dapat dilakukan imputasi menggunakan mean yang berasal dari dataset tersebut.

```
rating_df.head(10)
```

![rating_df_head](https://github.com/ilhamMalik51/DicodingAppliedML/blob/09e3b2d6195b79a2ecc378130fd2465dbd046c44/Proyek-Kedua/asset/df_head_rating_df.JPG)

Berdasarkan gambar di atas, dapat diketahui bahwa dataset tersebut terdiri dari 3 fitur. Fitur yang perlu diperhatikan pada dataset ini adalah fitur **rating**, dikarenakan berdasar dari informasi metadata, terdapat nilai rating -1 yang berarti orang tersebut yang sudah menonton namun tidak menilai film tersebut. Hal ini sebenarnya sama saja dengan _missing value_ terhadap fitur **rating**.

#### Memeriksa Jumlah Data Total, user_id, dan anime_id

Berdasarkan hipotesis solusi yang diajukan menggunakan pendekatan *Collaborative Filtering* maka fitur-fitur yang akan diperiksa adalah sebagai berikut.
1. Jumlah seluruh data penilaian terhadap *anime*
2. Jumlah user identifier
3. Jumlah anime identifier

```
print("Jumlah seluruh data penilaian anime: {}".format(len(rating_df["user_id"])))
print("Jumlah identifier unik user: {}".format(len(rating_df["user_id"].unique())))
print("Jumlah identifier unik anime: {}".format(len(anime_df["anime_id"].unique())))
```

![len_data](https://github.com/ilhamMalik51/DicodingAppliedML/blob/09e3b2d6195b79a2ecc378130fd2465dbd046c44/Proyek-Kedua/asset/panjang_total_anime_id_user_id.JPG)

#### Memeriksa Informasi Dataset

```
anime_df.info()
```

![df_anime_info](https://github.com/ilhamMalik51/DicodingAppliedML/blob/6314c91ee8adb0323f3623e41e534277299339f9/Proyek-Kedua/asset/df_info_anime_df.JPG)

```
rating_df.info()
```

![df_rating_info](https://github.com/ilhamMalik51/DicodingAppliedML/blob/6314c91ee8adb0323f3623e41e534277299339f9/Proyek-Kedua/asset/df_info_rating_df.JPG)

Berdasarkan kedua gambar di atas, dataset _Anime_ memiliki 7 fitur yang meliputi _anime_id_, _name_, _genre_, _type_, _episodes_, _rating_, dan _members_. Sedangkan untuk dataset _Rating_ memiliki 3 fitur meliputi _user_id_, _anime_id_, dan _rating._ Jumlah data yang terdapat pada dataset _Rating_ adalah 7.813.737. Jumlah ini sangat banyak, maka dari itu rasio pembagian data sudah dapat ditentukan.

### Exploratory Data Analysis: Data Cleaning
Berdasarkan informasi yang telah didapat pada bagian sebelumnya yaitu terdapat nilai rating -1. Seluruh baris data yang memiliki nilai tersebut tidak dapat digunakan jika tidak diubah atau dihapus. 

Berikut kode yang digunakan untuk memeriksa nilai rating -1 tersebut.

```
neg_rating = rating_df[rating_df["rating"] == -1]
pos_rating = rating_df[rating_df["rating"] != -1]

print("Jumlah data yang memiliki positif rating: {}".format(len(pos_rating)))
print("Jumlah data yang memiliki negatif rating: {}".format(len(neg_rating)))
print("Jumlah identifier unik pada data negatif rating: {}".format(len(neg_rating["anime_id"].unique())))
```

![jumlah_pos_neg](https://github.com/ilhamMalik51/DicodingAppliedML/blob/6314c91ee8adb0323f3623e41e534277299339f9/Proyek-Kedua/asset/jumlah_pos_rating_neg_rating.JPG)

Berdasarkan gambar di atas dan setelah dipertimbangkan, maka nilai rating -1 akan diimputasikan menggunakan fitur **rating** yang berada pada dataset *Anime*. Hal ini dilakukan karena jika data tersebut dihapus memiliki kemungkinan menghapus film anime tertentu. Selain itu, jika data dihapus, dataset Rating kehilangan sekitar 1.476.496 atau hampir 19% dari keseluruhan data.

Berikut kode yang digunakan untuk mengimputasikan nilai mean terhadap film _anime_ yang memiliki rating -1.

```
merge_rating = neg_rating.merge(anime_df, on="anime_id")

merge_rating = merge_rating[["user_id", "anime_id", "rating_y"]]
print("Panjang sebelum menghilangkan missing value: {}".format(len(merge_rating["anime_id"].unique())))

merge_rating = merge_rating.dropna(subset=["rating_y"])
print("Panjang setelah menghilangkan missing value: {}".format(len(merge_rating["anime_id"].unique())))

merge_rating["rating_y"] = merge_rating["rating_y"].astype("int64")
print("Merubah tipe data rating_y dari float menjadi int!")
```

Kode mengimputasikan nilai mean ini dimulai dengan menggunakan method `merge()`. Metode ini menyatukan dataset negatif rating dengan dataset anime. Dengan ini, setiap baris data akan memiliki fitur **rating_y** yang merupakan nilai rerata rating film _anime_ tersebut. Setelah itu,  kolom yang tidak digunakan dapat dihilangkan dan apabila terdapat film _anime_ yang tidak memiliki nilai rerata rating setelah diimputasikan dapat di-_drop_. Terakhir, karena nilai rerata ini merupakan nilai float, maka dapat di-_cast_ menjadi integer dan secara otomatis nilai desimal tersebut akan dibulatkan ke bawah.

Setelah data dengan nilai negatif rating diimputasikan, maka dataset kembali digabungkan. Hal tersebut dapat dilakukan dengan kode berikut.

```
prepared_df = pd.concat([pos_rating, merge_rating], ignore_index=True)

print("Jumlah data setelah dilakukan penggabungan data {}".format(len(prepared_df)))
```

Jumlah data setelah dilakukan penggabungan data 7.813.728.

### Exploratory Data Analysis: Univariate Analysis
Pada bagian ini akan memvisualisasikan distribusi data terhadap rating yang diberikan. Berikut kode yang digunakan untuk melakukan hal tersebut.

```
prepared_df.groupby("rating").size().plot(kind="bar", title="Distribution of Rating")
```

![bar_chart_rating](https://github.com/ilhamMalik51/DicodingAppliedML/blob/667019b25f754f39217e0c2d9475dfe3e3c87144/Proyek-Kedua/asset/distribution_of_Rating.png)

Berdasarkan gambar di atas, sumbu-x menjelaskan rating dari skala 1-10. Hal ini berarti rating ini nantinya akan di-_scale_. Selanjutnya pada sumbu-y menunjukan jumlah data terhadap nilai rating. Angka 1e6 menunjukan 10 pangkat 6, hal ini berarti angka desimal tersebut memiliki satuan 10 pangkat 6. Nilai rating 7 merupakan nilai terbanyak yaitu sekitar lebih dari 2 juta buah data. Hasil distribusi ini akan menjadi dasar pembagian data.

## Data Preparation
Pada bagian ini akan teknik data preparation antara lain:
1. Meng-_encode_ fitur user_id dan _anime_id_
2. Membuat _dictionary_ sebagai lookup table
3. Split dataset menjadi train dan test set

### Encode user_id dan anime_id
_Encoding_ fitur user_id dan anime_id menjadi indeks integer. Hal ini dilakukan karena dua hal:

1. Sebagai look up dictionary nantinya untuk kebutuhan _inference_ model.
2. Indeks row embedding nantinya akan mengikuti indeks tersebut dan diperlukan lookup supaya tidak tertukar.

Berikut kode yang digunakan untuk menerapkan hal tersebut.

```
encoded_df = prepared_df.copy()

encoded_df["user_id"] = encoded_df["user_id"].astype("category") 
encoded_df["user_id_encoded"] = encoded_df["user_id"].cat.codes

encoded_df["anime_id"] = encoded_df["anime_id"].astype("category")
encoded_df["anime_id_encoded"] = encoded_df["anime_id"].cat.codes
```

Kode di atas bekerja dengan merubah tipe _identifier_ yang sebelumnya bertipekan integer menjadi _category_. Setelah itu metode `.cat.codes` akan memberikan nilai indeks dari 0 hingga banyak data unik dari setiap _identifier_.

### Membangun Dictionary Sebagai LookUp Table
Berikut kode yang digunakan untuk menerapkan hasil encode tadi menjadi dictionary.

```
user_id_decoder = dict(zip(encoded_df["user_id_encoded"], encoded_df["user_id"]))
anime_id_decoder = dict(zip(encoded_df["anime_id_encoded"], encoded_df["anime_id"]))

user_id_encoder = dict(zip(encoded_df["user_id"], encoded_df["user_id_encoded"]))
anime_id_encoder = dict(zip(encoded_df["anime_id"], encoded_df["anime_id_encoded"]))
```

Kode tersebut akan memasangkan setiap _encoded_ indeks terhadap _identifier_ yang asli. Setelah dipasangkan, pasangan tersebut akan dijadikan tipe struktur data dictionary. Hal ini bertujuan sebagai _lookup table_ untuk menampilkan item-item tersebut.

### Splitting Data dan Feature Scaling
Berikut kode yang digunakan untuk menerapkan proses ini.

```
from sklearn.model_selection import StratifiedShuffleSplit

encoded_prep_df = encoded_df[["user_id_encoded", "anime_id_encoded", "rating"]]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in split.split(encoded_prep_df, encoded_prep_df["rating"]):
    strat_train_set = encoded_prep_df.loc[train_index]
    strat_test_set = encoded_prep_df.loc[test_index]
```

Pada kode di atas, digunakan metode pembagian data secara strata agar sampel _Training set_ merepresentasikan populasi/seluruh dataset. Hal ini dapat diperoleh dengan memanfaatkan library Scikit Learn menggunakan fungsi `StratifiedShuffleSplit()`. Rasio yang digunakan adalah 90:10 karena jumlah dataset sangat banyak.

Untuk bagian terakhir ini adalah dengan membagi fitur data dan label data serta menerapkan _feature scaling_ terhadap fitur rating.

```
X_train = strat_train_set[["user_id_encoded", "anime_id_encoded"]].to_numpy()
y_train = strat_train_set["rating"].to_numpy()
y_train = (y_train - min(y_train)) / (max(y_train) - min(y_train))

X_test = strat_test_set[["user_id_encoded", "anime_id_encoded"]].to_numpy()
y_test = strat_test_set["rating"].to_numpy()
y_test = (y_test - min(y_test)) / (max(y_test) - min(y_test))
```

## Modeling
Tahapan ini membahas mengenai model sistem rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Terdapat dua buah model yang diajukan untuk menyelesaikan masalah. Model tersebut antara lain RecommenderNet[5] dan PMF[4]. Berikut adalah library yang digunakan untuk membangun model-model tersebut.

```
import tensorflow as tf
from tensorflow import keras
```

### RecommenderNet Model
RecommenderNet ini dipilih karena merupakan salah satu model dengan arsitektur lebih baru dibandingkan dengan model PMF. Selain itu, model ini memiliki arsitektur yang lebih kompleks dibandingkan dengan model PMF. Berikut kode yang digunakan untuk membangun RecommenderNet. Berikut kode yang digunakan untuk membangun model ini.

```
class RecommenderNet(tf.keras.Model):
    def __init__(self, n_users, n_items, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        
        self.user_embedding = keras.layers.Embedding(n_users,
                                                     embedding_size,
                                                     embeddings_initializer = 'he_normal',
                                                     embeddings_regularizer = keras.regularizers.l2(1e-6))
        
        self.user_bias = keras.layers.Embedding(n_users, 1)
        
        self.items_embedding = keras.layers.Embedding(n_items,
                                                      embedding_size,
                                                      embeddings_initializer = 'he_normal',
                                                      embeddings_regularizer = keras.regularizers.l2(1e-6))
        self.items_bias = keras.layers.Embedding(n_items, 1)
    
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:,0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.items_embedding(inputs[:, 1])
        item_bias = self.items_bias(inputs[:, 1])

        dot_user_resto = tf.tensordot(user_vector, item_vector, 2) 

        x = dot_user_resto + user_bias + item_bias

        return tf.nn.sigmoid(x)
```

```
recommender_net = RecommenderNet(NUM_USERS, NUM_ITEMS, EMBEDDING_SIZE)

recommender_net.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        metrics=[tf.keras.metrics.RootMeanSquaredError()])
```

```
recommender_net_hist = recommender_net.fit(X_train, y_train,
                                           epochs=5,
                                           validation_data=(X_test, y_test))
```

### Probabilistic Matrix Factorization
Probabilistic Matrix Factorization ini dipilih karena biasa digunakan sebagai _baseline_ pada paper [6][7] untuk model sistem rekomendasi atau sebagai _environment_ untuk melatih model sistem rekomendasi yang lebih kompleks. Berikut kode yang digunakan untuk membangun RecommenderNet. Berikut kode yang digunakan untuk membangun model ini.

```
class PMF(tf.keras.Model):
    def __init__(self, n_users, n_items, n_dim):
        super(PMF, self).__init__()      
        self.w_u_i_init = tf.keras.initializers.RandomUniform(minval=-1., maxval=1., seed=1)
        
        self.user_embedding = tf.keras.layers.Embedding(n_users,
                                                        n_dim,
                                                        embeddings_initializer='uniform',
                                                        embeddings_regularizer=tf.keras.regularizers.L2(0.1))
      
        self.item_embedding = tf.keras.layers.Embedding(n_items,
                                                        n_dim,
                                                        embeddings_initializer='uniform',
                                                        embeddings_regularizer=tf.keras.regularizers.L2(0.1))
        
        ## users embedding
        self.ub = tf.keras.layers.Embedding(n_users, 
                                            1, 
                                            embeddings_initializer=self.w_u_i_init, 
                                            embeddings_regularizer=tf.keras.regularizers.L2(0.1)) 
        ## items embedding
        self.ib = tf.keras.layers.Embedding(n_items, 
                                            1, 
                                            embeddings_initializer=self.w_u_i_init, 
                                            embeddings_regularizer=tf.keras.regularizers.L2(0.1))
        
    def call(self, inputs):
        self.user_index = inputs[:, 0]
        self.item_index = inputs[:, 1]
        
        user_h1 = self.user_embedding(self.user_index)
        item_h1 = self.item_embedding(self.item_index)
        
        r_h = tf.math.reduce_sum(user_h1 * item_h1, axis=1 if len(user_h1.shape) > 1 else 0)
        r_h += tf.squeeze(self.ub(self.user_index))
        r_h += tf.squeeze(self.ib(self.item_index))
        
        return r_h
```

```
pmf = PMF(NUM_USERS, NUM_ITEMS, EMBEDDING_SIZE)

pmf.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer='adam',
            metrics=[tf.keras.metrics.RootMeanSquaredError()])
```

```
pmf_hist = pmf.fit(X_train, y_train,
                   epochs=5,
                   validation_data=(X_test, y_test))
```

Model akhir yang terpilih menjadi solusi adalah model RecommenderNet dikarenakan model tersebut memiliki loss RMSE lebih rendah dibandingkan dengan model PMF. Berikut merupakan _output_ Top-10 film _anime_ yang dihasilkan dari model tersebut.

![Top-K anime_recommendation](https://github.com/ilhamMalik51/DicodingAppliedML/blob/ac3fe0dff91c74bb7877ca056b80a40ee67c7767/Proyek-Kedua/asset/Top-N%20Recommendation.JPG)

## Evaluation
Pada bagian ini akan membahas metrik yang digunakan dan hasil _training_ serta evaluasi dari setiap model.

### Formula RMSE

Pada proyek ini metrik evaluasi yang digunakan adalah _Root Mean Squared Error_. Alasan menggunakan RMSE adalah karena metrik ini memiliki presisi yang cukup tinggi sehingga dapat menggunakan RMSE sebagai metrik evaluasi. Rumus RMSE dapat diekspresikan sebagai berikut.

![Formula RMSE](https://miro.medium.com/max/966/1*lqDsPkfXPGen32Uem1PTNg.png)

Keterangan:

- n     : jumlah banyak data
- y_hat : merupakan prediksi model
- y     : merupakan nilai target

### Evaluasi Model
Berikut merupakan hasil loss terhadap _Training set_ dan _Test set_ yang diperoleh dari setiap model.

#### RecommenderNet
![Recommender_Net](https://github.com/ilhamMalik51/DicodingAppliedML/blob/ac3fe0dff91c74bb7877ca056b80a40ee67c7767/Proyek-Kedua/asset/recommender_net_hist.png)

#### PMF Model
![PMF_Net](https://github.com/ilhamMalik51/DicodingAppliedML/blob/d5bd536f3eab926281c93b587bf2f9dadc69e932/Proyek-Kedua/asset/pmf_hist.JPG)

Berdasarkan kedua gambar di atas dapat disimpulkan bahwa:
1. Model RecommenderNet mengalami _overfitting_ pada setiap epoch, hal ini dikarenakan model ini lebih kompleks dibandingkan dengan model PMF. Meskipun demikian, model ini memiliki unjuk kerja yang lebih baik dibandingkan dengan model PMF
2. Model PMF mengalami _underfitting_ yang berarti model ini terlalu sederhana. Setelah 1 epcohs, unjuk kerja model PMF tidak meningkat secara signifikan, bahkan pada epochs 3, 4, dan 5 memiliki nilai RMSE yang sama.

### Kesimpulan
Untuk menjawab permasalahan pada bagian Problem Statement, dapat diurutkan sebagai berikut:
1. Model RecommenderNet dapat menyelesaikan permasalahan pengguna yang kesulitan dengan memberikan bantuan dalam bentuk daftar rekomendasi _anime_ yang cocok sesuai dengan pengguna tersebut.
2. Model yang optimal digunakan untuk domain ini adalah RecommenderNet
3. Dari pernyataan di atas, setelah didapatkan model terbaik, model tersebut dapat dilakukan _fine-tuning_ untuk meningkatkan unjuk kerja dari model tersebut contohnya seperti menambahkan _Weight Decay_.

## Referensi
[1]. Wibowo, Agung Toto. "Leveraging side information to anime recommender system using deep learning." 2020 3rd International Seminar on Research of Information Technology and Intelligent Systems (ISRITI). IEEE, 2020.

[2]. Hiromichi Masuda, Tadashi Sudo, Kazuo Rikukawa, Yuji Mori, Naofumi Ito, Yasuo Kameyama, and Megumi Onouchi. Anime industry report 2019, 2019.

[3]. Lu, Jie, et al. "Recommender system application developments: a survey." Decision Support Systems 74 (2015): 12-32. 

[4]. Mnih, Andriy, and Russ R. Salakhutdinov. "Probabilistic matrix factorization." Advances in neural information processing systems 20 (2007).

[5]. He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th international conference on world wide web. 2017.

[6]. Liu, Feng, et al. "Deep reinforcement learning based recommendation with explicit user-item interactions modeling." arXiv preprint arXiv:1810.12027 (2018).

[7]. Liu, Feng, et al. "End-to-end deep reinforcement learning based recommendation with supervised embedding." Proceedings of the 13th International Conference on Web Search and Data Mining. 2020.

**---Ini adalah bagian akhir laporan---**
