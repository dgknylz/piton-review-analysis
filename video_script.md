# 2-3 Dakikalik Turkce Demo Script

Merhaba, bu videoda Piton Technology teknik case'i icin hazirladigim Customer Review Analysis System projesini, bir satici paneli senaryosu uzerinden anlatacagim.

Bu projede kendimizi bir e-ticaret saticisi olarak dusunuyoruz. Amacimiz, urunlerimize gelen musteri yorumlarini tek tek okumadan genel resmi gorebilmek, riskli urunleri belirlemek, sikayet konularini ozetlemek ve egitilmis NLP modelini is kararlarina destek olacak sekilde kullanmak.

Uygulamanin arayuzu varsayilan olarak Turkce geliyor. Yan menuden English secenegiyle ayni dashboard Ingilizce olarak da kullanilabiliyor. Bu sayede proje teknik case gereksinimlerini karsilarken, satici tarafinda da daha okunabilir ve sunuma uygun bir deneyim veriyor.

Arayuzde iki veri kaynagi secenegi bulunuyor. Varsayilan olarak Kaggle'dan hazirlanmis proje verisi, yani data/processed/clean_reviews.csv kullaniliyor. Istersem kucuk bir satici test datasi yukleyip, ana veri setini bozmadan bu dosya uzerinde de model analizini calistirabiliyorum. Yuklenen test dosyasi sadece Streamlit oturumu icinde gecici olarak kullaniliyor; egitilmis model, clean_reviews.csv, raporlar ve figurler degistirilmiyor.

Veri seti olarak Kaggle Amazon US Customer Reviews veri setinin Electronics kategorisini kullandim. Bu kategori, batarya, sarj, ekran, ses kalitesi, arizali urun, iade ve kargo gibi somut sikayet temalari icerdigi icin teknik case icin uygun bir secim.

Ilk adimda ham TSV dosyasindan temiz ve tekrarlanabilir 30.000 satirlik bir orneklem olusturuyorum. Bu orneklemde yorum metni, yildiz puani, urun bilgisi, yorum tarihi, yorum uzunlugu ve helpful vote gibi alanlar hazirlaniyor. Yildiz puanlarindan pozitif, notr ve negatif etiketler uretiliyor.

Dashboard'un ana ekrani Overview. Burada toplam yorum sayisi, yorum almis urun sayisi, ortalama yildiz puani, sentiment dagilimi ve ortalama fuzzy reliability score goruluyor. Ayrica sentiment, rating ve yorum uzunlugu grafiklerine bakarak veri setinin genel yapisini anlayabiliyoruz.

Ikinci bolum Reviewed Products. Burada yorum almis urunler product_id ve product_title bazinda gruplanmis durumda. Her urun icin yorum sayisi, ortalama rating, pozitif, notr ve negatif yorum adetleri, negatif yorum orani ve ortalama reliability score hesaplaniyor. Ayrica negatif yorum oranina gore basit bir risk etiketi uretiliyor: High Risk, Medium Risk veya Low Risk. Bu satici icin cok onemli, cunku hangi urunlerin daha fazla musteri problemi yarattigini hizlica gosteriyor.

Product Detail Analysis bolumunde tek bir urunu secip daha detayli inceliyoruz. Secilen urun icin yorum sayisi, ortalama rating, sentiment dagilimi, negatif yorum orani, reliability score ve negatif yorumlardan cikan sikayet kelimeleri gosteriliyor. Ayrica yorum tablosunda modelin tahmin ettigi sentiment, confidence, fuzzy reliability score ve weighted confidence de yer aliyor.

Top Complaints bolumu is odakli sikayet ozetidir. Burada secilen kategori icin negatif yorumlar filtreleniyor ve sik gecen kelime veya bigram'lar cikariliyor. Veri seti tarihsel oldugu icin bu ekran "bu haftanin sikayetleri"ni gercek zamanli olarak uretmiyor; ama canli haftalik veri geldigi durumda ayni pipeline bu haftanin one cikan sikayetlerini otomatik uretmek icin kullanilabilir.

Products Without Reviews bolumunde onemli bir veri sinirini acikca gosteriyorum. Amazon Reviews veri seti sadece yorum almis urunleri iceriyor. Bu yuzden sadece bu veriyle yorum almamis urunleri bulamayiz. Default proje verisinde bunun icin saticinin tum urun katalogunu yuklemesi gerekir. Uploaded Test Dataset modunda ise review_body alani bos olan satirlar yorumsuz urun olarak ayriliyor ve dashboard'da ayri tabloda gosteriliyor.

Son tab Single Review Test. Bu bolum artik ana deneyim degil; sadece egitilmis modeli hizli test etmek icin var. Kullanici yeni bir yorum metni, rating ve yorum yasini giriyor. Sistem sentiment tahmini, model confidence, fuzzy reliability score ve weighted confidence gosteriyor.

Modelleme tarafinda TF-IDF ile Logistic Regression ve Random Forest karsilastirildi. Logistic Regression bu tip sparse metin verisinde guclu oldugu icin iyi performans gosterdi. GridSearchCV ile Logistic Regression optimize edildi ve en iyi model kaydedildi.

Fuzzy reliability sistemi rating, yorum uzunlugu ve yorum yasi girdilerini kullanarak 0-100 arasi bir guvenilirlik skoru uretiyor. Bu skor mutlak dogru degil; saticinin model tahminini yorum kalitesiyle birlikte degerlendirmesine yardim eden aciklanabilir bir is kurali katmani.

GitHub reposundaki README dosyasinda kurulum, dataset hazirlama, model egitimi, Streamlit calistirma adimlari ve ekran goruntuleri bulunuyor. Bu sayede projeyi ilk kez indiren biri once gerekli kutuphaneleri kurup, sonra veriyi hazirlayip modeli egiterek ayni dashboard'u kendi bilgisayarinda calistirabilir.

Ozetle bu proje, manuel yorum tahmininden daha fazlasini yapiyor. Bir saticinin yorum almis urunlerini analiz ediyor, riskli urunleri belirliyor, musteri sikayetlerini ozetliyor, katalog yuklenirse yorum almamis urunleri buluyor ve teknik olarak NLP modeli ile fuzzy logic entegrasyonunu tek bir dashboard'da gosteriyor.
