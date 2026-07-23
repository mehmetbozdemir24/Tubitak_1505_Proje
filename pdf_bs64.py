import base64

metin = """
AMAÇ

Bu dokümanın amacı, kurumsal bilgi yönetimi süreçlerinde doküman yükleme ve erişim mekanizmalarını test etmektir. İçerik örnek amaçlı hazırlanmıştır ve sistem entegrasyonunun doğrulanmasına yardımcı olur.

HEDEF

Hedef, bir metin dosyasının başarıyla yüklenmesi, indekslenmesi ve yetkili kullanıcılar tarafından erişilebilir hale gelmesidir. Ayrıca yüklenen içeriğin arama ve sorgulama işlemlerinde kullanılabilir olması beklenmektedir.

KAPSAM

Bu çalışma yalnızca test ortamında gerçekleştirilecek temel doküman yönetimi işlemlerini kapsamaktadır. Üretim ortamına yönelik herhangi bir işlem veya gerçek veri kullanımı planlanmamaktadır.
"""

base64_icerik = base64.b64encode(metin.encode("utf-8")).decode("utf-8")
print(base64_icerik)