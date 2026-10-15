# Doğrulanabilir ödül: GRPO ile Sigma kuralı üretimi

Küçük bir dil modeline, açıklamasını verdiğim saldırı için Sigma tespit
kuralı yazdırmayı GRPO ile öğretmeye çalıştım. İlk denemede model hiçbir şey
öğrenmedi. Sebebi bana GRPO'nun nasıl çalıştığını öğretti; bu depo o yolun
kodu, ölçümleri ve başarısız denemeleri.

**Kısaca**

- **Problem:** SFT ile eğittiğim model kuralları kusursuz biçimde yazıyor ama
  saldırı tekniğini (ATT&CK) yanlış etiketliyordu. Biçimi kopyalamak kolay,
  doğru tekniği bulmak değil.
- **İlk deneme çöktü:** "doğru teknik 1, yanlış 0" diye puanladım. 390 teknik
  arasında model hiç tutturamadı, bütün cevaplar 0 aldı ve GRPO öğrenecek
  bir fark bulamadı.
- **Düzelten:** puanı basamaklara bölmek (tam isabet / doğru aile / doğru
  taktik) ve modeli başlangıç hâline bağlayan küçük bir ceza (KL).
- **Sonuç:** ATT&CK isabeti **0,070 → 0,194**, çalışan kural **72 → 87 / 100**.
- **Ders:** ödülün doğrulanabilir olması yetmiyor, modelin ona adım adım
  ulaşabilmesi gerekiyor.

Her şey NVIDIA DGX Spark üzerinde (GB10, 128 GB) Eylül 2026'da koşturuldu.
Model Qwen3-0.6B. Ortamın kurulumu ayrı bir depoda:
[emr4h/dgx-spark-ortam](https://github.com/emr4h/dgx-spark-ortam).

## Önce: GRPO nedir?

SFT'de modele doğru cevabı gösterirsiniz, o kopyalar.

GRPO'da doğru cevap **yoktur**. Bir puanlayıcı vardır: model aynı soruya
sekiz cevap üretir, hepsi puanlanır, model iyi puan alanlara doğru kayar.
Açılımı Group Relative Policy Optimization; "relative" o sekiz cevabın
birbirlerine göre sıralanması demek. Ortalamadan iyi olan cevaplar
güçlendirilir, kötüler zayıflatılır. Kısacası öğretmen yok, sınav var.

Bundan önemli bir sonuç çıkıyor: **GRPO cevaplar arasındaki farktan öğrenir.**
Sekiz cevabın hepsi aynı puanı alırsa (örneğin hepsi 0) öğrenecek bir şey
kalmaz. Aşağıdaki ilk başarısız koşu tam olarak bu.

Tek şartı ödülün **hesaplanabilir** olması. Buna doğrulanabilir ödül deniyor.

**Sıra bozulmaz: önce SFT, sonra GRPO.** Eğitimsiz model Sigma kuralı değil
düz metin üretir, ödül sürekli 0 gelir ve karşılaştıracak bir şey kalmaz.
Bu depoda ölçüldü: eğitimsiz model 100 denemede **0** geçerli kural üretti.

## Neden Sigma, neden Splunk SPL değil?

İlk plan SPL'di. Ödül fonksiyonunu tasarlarken şu soru çıktı: modelin
ürettiği sorgunun geçerli olduğunu nasıl anlayacağım?

**SPL'in çevrimdışı doğrulayıcısı yok.** Resmi yol, çalışan bir Splunk
örneğindeki `/services/search/parser` uç noktasına sorguyu göndermek. GRPO
için bu ölümcül: ödül eğitim boyunca binlerce kez çalışıyor (tam bir epoch
1.488 prompt × 8 üretim ≈ 12.000 çağrı). Her biri ağ üzerinden lisanslı bir
sunucuya gidemez. Ödül **hızlı, yerel ve deterministik** olmalı.

Sigma'da bu sorun yok: `pySigma` kuralı yerel olarak doğruluyor.

Karar ödül tasarımına doğrudan yansıdı. Kesin bileşen sayısı 1'den 3'e çıktı:

| Bileşen | SPL ile | Sigma ile |
|---|---|---|
| Çıktı ayrıştırılıyor mu | sezgisel (parantez say) | **Kesin**, YAML ayrıştırıcısı |
| Kural geçerli mi | yapılamıyor | **Kesin**, `pySigma` |
| ATT&CK kimliği | kesin | **Kesin**, `tags` alanında |
| logsource örtüşmesi | kısmi | kısmi |

Bedava gelen: `pysigma-backend-splunk` ile Sigma çıktısı SPL'e çevrilebiliyor.
Birini seçmek zorunda değilim. Sigma hedef, Splunk türev.

## Veri

[SigmaHQ](https://github.com/SigmaHQ/sigma), Detection Rule License 1.1.
Kullan, değiştir, yayınla serbest; şart atıf. `NC`/`ND` yok.

Üçlü aynı dosyada duruyor ve tam ihtiyacım olanı veriyor:

```
description → modele verilecek SORU
detection   → modelin üreteceği CEVAP
tags        → cevabı DOĞRULAYACAĞIMIZ ATT&CK kimliği
```

Veri üretmedim; soru da cevap da doğrulayıcı da hazır.

```
rules/ altında ayrıştırılan   3.144
ATT&CK kimliği olan           2.798  (%89)
uzunluk filtresinden          2.687  → 2.488 eğitim / 199 test
farklı teknik                   390
```

Test seti **iki eksende** dengeleniyor: logsource ailesi (veri Windows'a
yığılmış, 2.060/2.687) ve `condition` zorluğu (tek / toplu / mantıksal).
Düz rastgele bölme test setini Windows'a kaydırır ve Linux'taki zayıflığı
gizler.

## Ödül

```
ayristir   pySigma YAML+şemayı kabul ediyor mu    kesin    0,5
donustur   backend condition'ı çözebiliyor mu     kesin    1,0
attack     ATT&CK isabeti (kademeli)              kesin    1,0
ortusme    logsource tuttu mu                     kısmi    0,3
```

`python odul.py --test` ile eğitimden önce sınanıyor. Yanlış ödülle saatlerce
eğitim koşup sonunda modelin puanlayıcıyı kandırdığını görmek istemezsiniz.

### Doğrulama neden iki adımlı?

`SigmaCollection.from_yaml()` tek başına yetmiyor. Var olmayan bir seçime
atıf yapan `condition: olmayan_secim` **ayrıştırmadan geçiyor**; yalnızca
backend dönüşümü yakalıyor.

```
bozuk YAML, detection yok, logsource yok, geçersiz değiştirici → 1. adım
tanımsız condition, boş detection                              → 2. adım
```

Tek adıma güvenseydik model "geçerli görünen ama çalışmayan" kurallar
üretmeyi öğrenebilirdi.

## Sonuç

100 kayıtlık test setinde:

```
                eğitimsiz     SFT      GRPO    değişim
ayristir          0,000     0,360     0,435    +0,075
donustur          0,000     0,720     0,870    +0,150
attack            0,000     0,070     0,194    +0,124
ortusme           0,000     0,160     0,195    +0,035
TOPLAM            0,000     1,310     1,694    +0,384
çalışan kural      0/100    72/100    87/100      +15
```

Zorluk kırılımı, üçünde de kazanç var:

```
            SFT     GRPO
tek        1,236   1,594
toplu      1,451   1,801
mantiksal  1,326   1,771
```

**SFT biçimi öğretti, GRPO bilgiyi.** `attack` neredeyse üç katına çıktı.
Bu, bir önceki paketin bulgusunun devamı: orada SFT'nin ATT&CK kimliklerini
öğretemediğini ölçmüştüm (0,44 → 0,50). Burada onu kimin öğretebileceğini.

## Üç başarısız GRPO koşusu

Her biri bir öncekinin gizlediği sorunu açığa çıkardı.

### 1. İkili ödül hiçbir şey öğretmedi

ATT&CK kimliği kusursuz doğrulanabilir bir hedefti. Ama 390 teknik arasından
doğruyu bulma şansı %0,26. Sekiz üretimin hepsi 0 aldı, `std = 0` oldu ve
**GRPO o bileşenden hiçbir yön bilgisi çıkaramadı.**

GRPO yalnızca *farktan* öğrenir. Fark yoksa gradyan yok.

Geriye kalan gradyan sadece biçim bileşenlerinden geliyordu; model orada
zaten %87'deydi, kazanacak yeri yoktu. 20 adımda ödül 1,575'ten 1,327'ye
düştü. Model SFT'de öğrendiğini kaybediyor, karşılığında hiçbir şey
kazanmıyordu.

**Ders:** doğrulanabilirlik gerekli ama yeterli değil. Ödülün **ulaşılabilir**
de olması gerekiyor.

### 2. Kademeli ödül gradyanı geri getirdi, erozyonu durdurmadı

ATT&CK zaten hiyerarşik. Ödülü ona göre kademelendirdim:

```
t1059.001 = t1059.001   tam isabet      1,0
t1059     ~ t1059.001   doğru aile      0,5
execution ~ execution   doğru taktik    0,2
alakasız                                0,0
```

`attack` tırmanmaya başladı: 0,048 → 0,104. Ama toplam ödül hâlâ düşüyordu.
Kazanç kaybı karşılamıyordu:

```
adım 5 → 10
  attack    +0,056
  ayristir  −0,025
  donustur  −0,050   } −0,075
```

### 3. KL cezası çıpayı koydu

`beta = 0.0` idi. Modeli SFT başlangıç noktasına bağlayan hiçbir şey yoktu,
RL serbestçe sürükleniyordu. `beta = 0.02` ile:

```
EĞİM (ölçüm başına)   ayristir  donustur   attack  ortusme   TOPLAM
1 ikili, KL yok        −0,0169   −0,0287  +0,0025  −0,0101  −0,0534
2 kademeli, KL yok     −0,0249   −0,0500  +0,0563  −0,0150  −0,0330
3 kademeli, KL 0,02    +0,0003   +0,0015  +0,0013  +0,0002  +0,0032
```

Üçüncü koşuda bütün bileşenler pozitif. Erozyon durdu.

> **Dürüstlük notu:** ilk iki koşuyu bozulduklarını görünce erken durdurdum;
> 4 ve 2 ölçüm noktaları var, üçüncüde 30. Eğimleri bu yüzden ölçüm değil
> gözlem sayılmalı. Sağlam olan: koşu 3 bozulmadı, diğer ikisi bozuldu.

Ve eğitim ödülü düz kalmasına rağmen (1,600 → 1,599) test skoru 1,310'dan
1,694'e çıktı. Çelişki değil: eğitim ödülü GRPO promptları üzerinde, test
ayrı bir set üzerinde. Model eğitim dağılımında tavana yakındı; öğrendiğini
görünmeyen veriye taşıdı.

## Nasıl tekrarlanır

Ortam: [emr4h/dgx-spark-ortam](https://github.com/emr4h/dgx-spark-ortam)
konteyneri (torch, transformers, TRL, PEFT, datasets hazır). Üstüne
`pip install pysigma pysigma-backend-splunk pyyaml`. `veri_sec.py` SigmaHQ'yu
`git` ile çeker; ağ gerekir.

```bash
python veri_sec.py                    # SigmaHQ'yu çeker, havuzu kurar
python odul.py --test                 # ÖDÜLÜ ÖNCE SINA
python sft_hazirla.py                 # 1000 SFT + 1488 GRPO promptu

python egit_lora.py --veri veri/sft_1000.jsonl --cikti ckpt/sft_1000 --epoch 3
python test_kos.py --cikti cikti/temel.jsonl                    # taban
python test_kos.py --adapter ckpt/sft_1000 --cikti cikti/sft.jsonl

python grpo_kos.py --adapter ckpt/sft_1000 --adim 150 --beta 0.02
python test_kos.py --taban-adapter ckpt/sft_1000 --adapter ckpt/grpo \
                   --cikti cikti/grpo.jsonl

python kosu_karsilastir.py grpo1.log grpo2.log grpo3.log   # eğim karşılaştırması
```

Bu depodaki GRPO koşusu üretimi `transformers.generate` ile yaptı (adım
başına ~14 sn). `vllm_servis.sh` modeli ayrı bir konteynerde OpenAI uyumlu
uç nokta olarak servis eder; aşağıdaki hız ölçümü o servisle alındı. GRPO
eğitimine bağlanmadı: TRL'in `vllm_mode="server"` modu ağırlık eşitlemek için
`trl vllm-serve` bekliyor, düz `vllm serve` yetmiyor. Sıradaki iş bu.

## Dosyalar

| Dosya | Ne işe yarar |
|---|---|
| `veri_sec.py` | SigmaHQ'dan havuzu çıkarır, iki eksende dengeli böler |
| `odul.py` | Dört bileşenli ödül + `--test` ile sınama |
| `sft_hazirla.py` | SFT ve GRPO veri setlerini kurar (kesişmez) |
| `egit_lora.py` | LoRA + SFT eğitimi |
| `grpo_kos.py` | GRPO eğitimi |
| `test_kos.py` | Test setinde koşturup puanlar |
| `kosu_karsilastir.py` | Koşuları **eğimle** karşılaştırır, tek noktayla değil |
| `vllm_servis.sh` | vLLM'i ayrı konteynerde servis eder (üretim hızı ölçümü; GRPO'ya bağlı değil) |

## Yeniden kuracaklar için: dört teknik engel

Sonucu anlamak için bu bölüm gerekmiyor; depoyu kendi makinesinde
çalıştıracaklar için. Her biri bir kez yolumu kesti.

### Arrow şema hatası

`datasets` tüm satırların aynı tipte olmasını istiyor. Sigma'nın `detection`
alanı serbest biçimli. Bir kuralda `EventID: 4104` (sayı), başkasında
`[4104, 4105]` (dizi):

```
ArrowInvalid: Column(/detection/selection/EventID)
              changed from number to array in row 7
```

Eğitim başlamadan çöktü. Üstelik o alanı ödül hiç kullanmıyordu.

**Ders: RL veri setinde ödülün okumadığı hiçbir alan bulunmasın.**

### TRL ile transformers uyumsuzluğu

```
grpo_trainer.py:501  model.warnings_issued["estimate_tokens"] = True
AttributeError: 'Qwen3ForCausalLM' object has no attribute 'warnings_issued'
```

`warnings_issued` transformers 4.x'te `PreTrainedModel`'in alanıydı, 5.x'te
kaldırılmış; TRL 0.26.1 hâlâ arıyor. Tek satırlık yamayla geçildi.

Yan bulgu: ortam doğrulama scriptim beş trainer'ı da "OK" gösteriyordu ama
yalnızca **import** ediyordu, **kurmayı** denemiyordu. İkisi farklı şey.
`from trl import GRPOTrainer` sorunsuz çalışıyor, hata ancak `GRPOTrainer(...)`
çağrılınca çıkıyor.

### vLLM: ilk sonucum yanlıştı

Plan vLLM'i colocate modunda kullanmaktı. Üç engel çıktı:

1. İmajda vLLM yok (`weave`/`mergekit` boşluğuyla aynı sınıf)
2. `pip install vllm` torch'u 2.13'e yükseltmek istiyor. NGC'nin 2.10.0a0
   derlemesi Blackwell desteğini taşıyor, gitmesi GPU'yu kullanılamaz yapar
3. `--no-deps` ile kurulunca:
   `undefined symbol: torch_get_mutable_data_ptr`

Buraya kadar doğru. Sonra **"vLLM bu cihazda kullanılamıyor" diye
genelledim ve yanıldım.**

NVIDIA kendi torch'una karşı derlenmiş bir vLLM imajı yayınlıyor:
`nvcr.io/nvidia/vllm`, arm64 dahil. İçindeki torch eğitim imajıyla birebir
aynı sürüm. Tek kurulum yolu denemek yetmiyormuş.

**Doğru mimari iki konteyner:** vLLM kendi torch'uyla servis eder, eğitim
konteyneri HTTP üzerinden konuşur. ABI çakışması ortadan kalkar.

Ama ayırmak yeni bir sınır yaratıyor: eğitim konteynerindeki transformers
5.12.1'in yazdığı tokenizer'ı vLLM'deki 4.57.1 okuyamıyor
(`extra_special_tokens` 5.x'te liste, 4.x'te sözlük). Çözüm tokenizer'ı
kaydetmeyip `--tokenizer Qwen/Qwen3-0.6B` ile kaynaktan okutmak. Model
ağırlıkları sürümden bağımsız, tokenizer değil.

**Ölçülen kazanç.** Aynı model, aynı 100 istem, aynı GPU:

```
                süre      token    token/s
transformers   212,7 sn   29.493     139
vLLM            63,8 sn   49.868     782
                ────────────────────────────
                3,3 kat             5,6 kat
```

İki oran farklı çünkü vLLM daha uzun çıktı üretti. `transformers` tarafında
batch içindeki kısa cevaplar uzunları beklerken dolgu harcanıyor; vLLM'in
sürekli batch'lemesinde o israf yok. Gerçek kazanç token/s oranında.

### Adaptör üstüne adaptör: konfigürasyon yalan söylüyor

GRPO, SFT adaptörü **eritilmiş** model üstünde eğitiliyor. Yani `ckpt/grpo`
adaptörü "taban + SFT" ağırlıklarına göreli. Ama PEFT bunu bilmiyor ve
`adapter_config.json`'a `base_model: Qwen/Qwen3-0.6B` yazıyor, çünkü
`from_pretrained`'e verilen ad buydu.

Çıplak tabana takınca model Sigma değil, "tespit kuralı herhalde böyledir"
tahmini üretiyor:

```yaml
name: "Load Response File..."       # Sigma'da title:
mitre_attack_technique: "1022"      # Sigma'da tags:
```

Test **0,000** verdi, eğitim ödülü 1,6 iken. İmkânsız bir çelişki olduğu
için yakalandı. Doğru yol eğitimdeki tabanı aynen yeniden kurmak:
`--taban-adapter ckpt/sft_1000`.

## Sınırlar

- **Tek model, tek boyut.** Qwen3-0.6B. Daha büyük modelde ATT&CK isabeti
  çok daha yüksek başlayabilir ve GRPO'nun katkısı farklı görünebilir.
- **150 adım az.** Ödül eğrisi düz kalmıştı; daha uzun koşu daha fazla
  kazanç verebilir.
- **İlk iki koşu erken durduruldu**, eğimleri gözlem düzeyinde.
- **Kademeli ödülün basamak değerleri (1,0 / 0,5 / 0,2) ölçülmedi**, makul
  seçildi. Başka oranlar daha iyi olabilir.
- **Türev veri ve model yayınlanmıyor**; scriptler kaynağı kendileri indirir.

## Kaynaklar

- [SigmaHQ](https://github.com/SigmaHQ/sigma), DRL 1.1
- [pySigma](https://pypi.org/project/pysigma/), çevrimdışı doğrulayıcı
- `nvcr.io/nvidia/vllm`, NVIDIA'nın kendi torch'una karşı derlenmiş vLLM
- Bir önceki çalışmam: [emr4h/sft-bicim-mi-bilgi-mi](https://github.com/emr4h/sft-bicim-mi-bilgi-mi)

## Lisans

Kod MIT. Kaynak kural seti DRL 1.1. Türev kullanımda `author` alanı ve
kaynağa link korunmalı.
