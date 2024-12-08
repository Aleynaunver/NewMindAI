# **Kullanıcı Yorumları Analizi ve Sonuç Üretimi**
  
Bu proje, kullanıcı yorumlarını analiz ederek, konuları ve görüşleri sınıflandırmayı ve ardından bu yorumlardan anlamlı sonuçlar üretmeyi amaçlamaktadır. Proje, makine öğrenimi modelleri, cümle benzerliği algoritmaları ve açık kaynaklı bir LLM kullanılarak gerçekleştirilmiştir.


## **İçindekiler**
- Proje Hakkında
- Veri Seti
- Kullanılan Teknolojiler
- Adımlar
- Sonuçlar
- API ve Arayüz Entegrasyonu
- Nasıl Çalıştırılır
- Dosya Yapısı




## **Proje Hakkında**

  Bu projede, kullanıcıların Spotify uygulaması hakkındak, yorumları analiz edilmiştir. Yorumlar, sınıflandırma ve cümle benzerliği teknikleri ile işlenmiş, ardından bir LLM modeli kullanılarak anlamlı sonuçlar üretilmiştir. Sonuçlar, yorumlardan çıkarılan genel fikirleri yansıtmaktadır. Yeni gelen kullanıcı yorumu veri seti ile birleştirilerek yorumlama yapılmıştır.




## **Veri Seti**

Toplam satır sayısı: 14,000
Toplam sütun sayısı: 2
Sütunların özellikleri:
Review: Kullanıcıların Spotify uygulaması hakkındaki yorumları içerir.
Label: Yorumların "POSITIVE" (olumlu) veya "NEGATIVE" (olumsuz) olarak etiketlendiği sütun.




## **Kullanılan Teknolojiler**

Proje kapsamında aşağıdaki teknolojiler ve araçlar kullanılmıştır:

- Programlama Dilleri: Python
- Metin Temizliği, Analizi ve Görselleştirme: Pandas, SpaCy, Matplotlib
- Makine Öğrenimi Modelleri:
  - Denenen modeller: Random Forest, SVM, Decision Tree, Gradient Boosting, K-Nearest Neighbors, Naive Bayes
  - Seçilen model: Logistic Regression (Diğer modellere göre daha hızlı çalışma süresi sağladı ve yorum sınıflandırmasında yeterli performansı gösterdi)
- NLP Teknikleri: TF-IDF, Sentence Similarity
- LLM: Llama3.2-1B-Inference
- API Framework: FastAPI
- Arayüz: Streamlit
- Değerlendirme Metrikleri: Accuracy, Precision, Recall, F1 score, Rouge, Blue
- Platform: VS Code




## **Adımlar**


### **1) Veri Analizi ve Temizleme**

Veri kümesinin detaylı analizi yapıldı. Etiket incelemesi, null değerlerin kontrolü, tekrar eden değerlerin olup olmadığı gibi kontroller yapıldı. Emojiler kaldırıldı. Gereksiz karakterler temizlendi. Tokenizasyon, stopword kaldırılması, pos taggin ve lematizasyon yapılarak preprocessing aşaması tamamlandı.


### **2) Özellik Çıkarımı**

TF-IDF yöntemi kullanılarak metinlerden özellikler çıkarıldı.


### **3) Makine Öğrenimi ile Sınıflandırma**

Logistic Regression modeli kullanılarak yeni yorumların "olumlu" ya da "olumsuz" olduğunu tahmin edecek bir model oluşturuldu.
Model performansı accuracy, precision, recall ve F1 skorları ile değerlendirildi.


### **4) Cümle Benzerliği ile Gruplandırma**

Sentence Similarity algoritması ile benzer yorumlar gruplandırıldı ve veri boyutu küçültüldü.


### **5) Sonuç Üretimi**
Llama3.2-1B-Inference modeli kullanılarak gruplandırılmış yorumlardan anlamlı sonuçlar üretildi.


### **6) API ve UI Entegrasyonu**

API: FastAPI kullanılarak API geliştirmesi yapıldı.
Arayüz:  Streamlit ile, kullanıcıların yorum gönderebildiği ve modelin cevaplarını görebildiği bir web arayüzü geliştirildi.
Streamlit ile arayüz tasarlanarak API ile entegre edildi.


## **SONUÇLAR**

Yorumlardan elde edilen örnek bir sonuç metni:

  _Spotify is a popular music streaming service that offers a vast library of songs, playlists, and radio stations. With a user-friendly interface and a wide range of features, it's no wonder that millions of users worldwide have fallen in love with it. Spotify's selection is vast, with millions of songs available to stream, including popular and obscure tracks. Users can create and manage their own playlists, discover new music through Discover Weekly and Release Radar, and even create custom playlists with specific moods or activities. Spotify's great app is available for both desktop and mobile devices, making it easy to access and enjoy music anywhere, anytime. However, Spotify has faced several issues in recent times, including being unable to play certain songs, having playlist shuffle and playback control issues, and sometimes disappearing songs from the playlist. Despite these problems, Spotify remains one of the best music apps available, offering a truly amazing music experience with its vast library, personalized recommendations, and social features."_


## **PROJENİN SON HALİ**



![image](https://github.com/user-attachments/assets/2a5d35d7-af80-4b24-9868-a481e7ef013d)


